#%%
import fitz
import re
from collections import defaultdict
import pandas as pd
#%%
#DR pattern that should be met under DR column
DR_pattern = re.compile(
    r"\b(?:ESRS\s*)?(?:[EGS]\d+|GOV|SBM|BP|SMB|IRO|MDR|S\d+|S\d+-\d+)"
    r"(?:[\s\-–]+(?:ESRS\s*)?(?:[EGS]\d+|GOV|SBM|BP|SMB|IRO|MDR|S\d+))*"
    r"(?:[\s\-–]+\d+)?\b",
    re.IGNORECASE
)

#in addition to DR list there might be entity specific DR list
ENTITY_pattern = re.compile(
    r"\b(entity[\s\-]*specific|entity[\s\-]*related[\s\-]*disclosure|"
    r"custom[\s\-]*disclosure|entity[\s\-]*level\s+esrs)\b",
    re.IGNORECASE
)

#keywords to find the relavant pages where the DR table is located
keywords = ["disclosure requirement", "location", "standard section", 
           "reference table", "content index", "ESRS indices", "reference in the report", "index of", "section", "page number", "reference to", "list of ESRS disclosure requirements"]

#the DR table is usually followed by EU legislation table hence we need to distinguish then
EU_pattern = re.compile(
    r"\b("
    r"eu\s+legislation|"
    r"eu\s+taxonomy|"
    r"eu\s+regulation|"
    r"sfdr|"
    r"sfdr\s+reference|"
    r"pillar\s+3\s+reference|"
    r"directive\s+\d+/\d+/EC|"
    r"regulation\s+\(EU\)\s+\d+/\d+|"
    r"due\s+diligence|"
    r"elements\s+of\s+due\s+diligence"
    r")\b",
    re.IGNORECASE
)

#%%
def looks_like_table(text):
    lines = text.splitlines()
    table_like_lines = sum(1 for line in lines if len(re.findall(r'\s{2,}', line)) >= 1)
    return table_like_lines >= 3

def extract_text_and_score_pages(pdf_path):
    doc = fitz.open(pdf_path)
    results = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        lowered = text.lower()

        # Count ESRS codes and entity specific codes
        dr_matches = DR_pattern.findall(lowered)
        entity_matches = ENTITY_pattern.findall(lowered)
        all_dr_matches = dr_matches+entity_matches
        esrs_count = len(all_dr_matches)
        unique_esrs = set(all_dr_matches)


        # Count keyword hits
        keyword_count = sum(kw in lowered for kw in keywords)

        # Detect table structure
        has_table_structure = looks_like_table(text)

        #count eu legislation mentionings
        eu_hits = bool(EU_pattern.search(lowered))
        eu_penalty = 1 if eu_hits else 0  

        #total scoring system. the most weight is to the keyword and eu legislation penalty
        score = (esrs_count*1.5 +len(unique_esrs) * 1.5 + keyword_count * 3 + (1 if has_table_structure else 0)*0.1 - eu_penalty*15)

        results.append({
        "page_num": page_num,
        "total_score": score,
        "keyword_hits": keyword_count,
        "unique_esrs": len(unique_esrs),
        "total_esrs": esrs_count,
        "has_table": has_table_structure,
        "eu_legislation_hit": eu_hits,
        "eu_penalty": eu_penalty
    })

    df = pd.DataFrame(results)
    df_sorted = df.sort_values("total_score", ascending=False).reset_index(drop=True)
    return df_sorted


#%%
def get_expanded_page_range(df, pdf_path, esrs_thresh=5, unique_thresh=3):
    """Takes the top one page candidate and looks through its neighbours to find a range of table pages. Note: the logic is that the page range should cover the tables beginning and the end, hence the might +/- 1 not relevant page but that insures the extraction of the whole table later"""
    df = df.copy()
    numeric_cols = ['page_num', 'total_score', 'keyword_hits', 'unique_esrs', 'total_esrs']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    top_page = int(df.iloc[0]['page_num'])
    assessed_pages = []
    final_pages = [top_page]

    doc = fitz.open(pdf_path)

    def assess_page(page_num):
        if page_num not in df['page_num'].values:
            return None
        row = df[df['page_num'] == page_num].iloc[0]

        text = doc[page_num].get_text().lower()
        eu_hits = EU_pattern.findall(text)
        eu_penalty = len(eu_hits)

        meets_criteria = (
            (row['total_esrs'] >= esrs_thresh and row['unique_esrs'] >= unique_thresh)
            and eu_penalty <= 6
        )

        return {
            'page': page_num,
            'total_score': row['total_score'],
            'total_esrs': row['total_esrs'],
            'unique_esrs': row['unique_esrs'],
            'eu_penalty': eu_penalty,
            'meets_criteria': meets_criteria,
            'in_final_range': False
        }

    # Check backward neighbours
    high_eu_seen = False
    for offset in range(1, 6): #check 5 neighbours backward
        i = top_page - offset
        if i < 0:
            break
        result = assess_page(i)
        if result:
            assessed_pages.append(result)
            if not (row := result)['total_esrs'] >= esrs_thresh and row['unique_esrs'] < unique_thresh:
                break

            if result['eu_penalty'] > 5:
                if high_eu_seen:
                    break  # second high-EU page → stop
                #one high eu penalty is allowed to ensure that we capture all table pages
                high_eu_seen = True
            else:
                high_eu_seen = False

            final_pages.insert(0, i)
            result['in_final_range'] = True
        else:
            break

    # Check forward neighbours
    high_eu_seen = False
    for offset in range(1, 6): #check 5 pages forward 
        i = top_page + offset
        if i >= len(doc):
            break
        result = assess_page(i)
        if result:
            assessed_pages.append(result)
            if not (row := result)['total_esrs'] >= esrs_thresh and row['unique_esrs'] < unique_thresh:
                break

            if result['eu_penalty'] > 5:
                if high_eu_seen:
                    break  # second high-EU page → stop
                high_eu_seen = True
            else:
                high_eu_seen = False

            final_pages.append(i)
            result['in_final_range'] = True
        else:
            break

    doc.close()

    assessment_df = pd.DataFrame(assessed_pages).sort_values("page")
    return {
        "final_page_range": sorted(final_pages),
        "assessment_details": assessment_df
    }


# %%
#RUN only this if you already ran the functions
#prints the top one candidate page where the table might be located
pdf_path = "2024-Acerinox-Group-Consolidated-Management-Report.pdf"
df_result = extract_text_and_score_pages(pdf_path)
top_page_number = df_result.iloc[0]["page_num"]
print("Top page number:", top_page_number)

#returns the range of the pages where the whole table is present + the scoring table
result = get_expanded_page_range(df_result, pdf_path)
print("Final page range:", result["final_page_range"])
print(result["assessment_details"][[
    "page", "total_score", "total_esrs", "unique_esrs", "eu_penalty", "meets_criteria", "in_final_range"
]])















#%%
#additionally run:
#to check the content of the specific page
def show_pdf_page_text(pdf_path, page_number, char_limit=None):
    """Prints the text of a specific page from a PDF."""
    with fitz.open(pdf_path) as doc:
        if page_number < 0 or page_number >= len(doc):
            print(f" Page {page_number} is out of range (0 to {len(doc)-1})")
            return
        text = doc[page_number].get_text()
        if char_limit:
            text = text[:char_limit] + "..." if len(text) > char_limit else text
        print(f"📄 Content of page {page_number}:\n")
        print(text)

show_pdf_page_text(pdf_path, 166, char_limit=500)
#%%
#to check with eu penalty keywords were detected
def count_eu_penalty(pdf_path, page_number, show_matches=False):
    with fitz.open(pdf_path) as doc:
        if page_number < 0 or page_number >= len(doc):
            print(f" Page {page_number} is out of range (0 to {len(doc) - 1})")
            return
        text = doc[page_number].get_text().lower()
        matches = EU_pattern.findall(text)
        print(f"🔍 Page {page_number} EU keyword penalty: {len(matches)}")
        if show_matches:
            print("Matched terms:", matches)

count_eu_penalty(pdf_path, 120, show_matches=True)