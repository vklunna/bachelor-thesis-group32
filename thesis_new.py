#%%
import fitz
import re
from collections import defaultdict
import pandas as pd
#%%
DR_pattern = re.compile(
    r"\b(?:ESRS\s*)?(?:[EGS]\d+|GOV|SBM|BP|SMB|IRO|MDR|S\d+|S\d+-\d+)"
    r"(?:[\s\-–]+(?:ESRS\s*)?(?:[EGS]\d+|GOV|SBM|BP|SMB|IRO|MDR|S\d+))*"
    r"(?:[\s\-–]+\d+)?\b",
    re.IGNORECASE
)
ENTITY_pattern = re.compile(
    r"\b(entity[\s\-]*specific|entity[\s\-]*related[\s\-]*disclosure|"
    r"custom[\s\-]*disclosure|entity[\s\-]*level\s+esrs)\b",
    re.IGNORECASE
)
keywords = ["disclosure requirement", "location", "standard section", 
           "reference table", "content index", "ESRS indices", "reference in the report", "index of", "section", "page number", "reference to", "list of ESRS disclosure requirements"]
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

        # Count ESRS codes
        dr_matches = DR_pattern.findall(lowered)
        entity_matches = ENTITY_pattern.findall(lowered)
        all_dr_matches = dr_matches+entity_matches
        esrs_count = len(all_dr_matches)
        unique_esrs = set(all_dr_matches)


        # Count keyword hits
        keyword_count = sum(kw in lowered for kw in keywords)

        # Detect table structure
        has_table_structure = looks_like_table(text)

        eu_hits = bool(EU_pattern.search(lowered))
        eu_penalty = 1 if eu_hits else 0  

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

# %% #keep
# def get_expanded_page_range(df, pdf_path, esrs_thresh=5, unique_thresh=3, max_eu_penalty=4):
#     """Expands page range around top candidate and returns assessment details."""
#     df = df.copy()
#     numeric_cols = ['page_num', 'total_score', 'keyword_hits', 'unique_esrs', 'total_esrs']
#     df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
#     top_page = int(df.iloc[0]['page_num'])
#     assessed_pages = []
#     final_pages = [top_page]

#     doc = fitz.open(pdf_path)  # Open PDF once

#     def assess_page(page_num):
#         if page_num not in df['page_num'].values:
#             return None
#         row = df[df['page_num'] == page_num].iloc[0]

#         text = doc[page_num].get_text().lower()
#         eu_hits = EU_pattern.findall(text)
#         eu_penalty = len(eu_hits)

#         meets_criteria = (
#             eu_penalty <= max_eu_penalty and
#             (row['total_esrs'] >= esrs_thresh or row['unique_esrs'] >= unique_thresh)
#         )

#         return {
#             'page': page_num,
#             'total_score': row['total_score'],
#             'total_esrs': row['total_esrs'],
#             'unique_esrs': row['unique_esrs'],
#             'eu_penalty': eu_penalty,
#             'meets_criteria': meets_criteria,
#             'in_final_range': meets_criteria
#         }

#     # Check backward
#     for offset in range(1, 6):
#         prev_page = top_page - offset
#         if prev_page < 0:
#             break
#         result = assess_page(prev_page)
#         if result:
#             assessed_pages.append(result)
#             if result['meets_criteria']:
#                 final_pages.insert(0, prev_page)
#             else:
#                 break

#     # Check forward
#     for offset in range(1, 6):
#         next_page = top_page + offset
#         if next_page >= len(doc):
#             break
#         result = assess_page(next_page)
#         if result:
#             assessed_pages.append(result)
#             if result['meets_criteria']:
#                 final_pages.append(next_page)
#             else:
#                 break

#     doc.close()

#     assessment_df = pd.DataFrame(assessed_pages).sort_values("page")
#     assessment_df["in_final_range"] = assessment_df["page"].isin(final_pages)

#     return {
#         "final_page_range": sorted(final_pages),
#         "assessment_details": assessment_df
#     }

# %%
pdf_path = "Annual Report 2024_Zalando SE_EN_250503_s.pdf"
df_result = extract_text_and_score_pages(pdf_path)
top_page_number = df_result.iloc[0]["page_num"]
print("Top page number:", top_page_number)


def get_expanded_page_range(df, pdf_path, esrs_thresh=5, unique_thresh=3):
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
            (row['total_esrs'] >= esrs_thresh or row['unique_esrs'] >= unique_thresh)
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

    # Check backward
    consecutive_high_eu = 0
    for offset in range(1, 6):
        i = top_page - offset
        if i < 0:
            break
        result = assess_page(i)
        if result:
            assessed_pages.append(result)
            if not result['meets_criteria']:
                break

            if result['eu_penalty'] > 6:
                consecutive_high_eu += 1
                if consecutive_high_eu >= 7:
                    break
            else:
                consecutive_high_eu = 0

            final_pages.insert(0, i)
            result['in_final_range'] = True

    # Check forward
    consecutive_high_eu = 0
    for offset in range(1, 6):
        i = top_page + offset
        if i >= len(doc):
            break
        result = assess_page(i)
        if result:
            assessed_pages.append(result)
            if not result['meets_criteria']:
                break

            if result['eu_penalty'] > 6:
                consecutive_high_eu += 1
                if consecutive_high_eu >= 7:
                    break
            else:
                consecutive_high_eu = 0

            final_pages.append(i)
            result['in_final_range'] = True

    doc.close()

    assessment_df = pd.DataFrame(assessed_pages).sort_values("page")
    return {
        "final_page_range": sorted(final_pages),
        "assessment_details": assessment_df
    }

result = get_expanded_page_range(df_result, pdf_path)
print("Final page range:", result["final_page_range"])
print(result["assessment_details"][[
    "page", "total_score", "total_esrs", "unique_esrs", "eu_penalty", "meets_criteria", "in_final_range"
]])

#%%
def show_pdf_page_text(pdf_path, page_number, char_limit=None):
    """Prints the text of a specific page from a PDF."""
    with fitz.open(pdf_path) as doc:
        if page_number < 0 or page_number >= len(doc):
            print(f"❌ Page {page_number} is out of range (0 to {len(doc)-1})")
            return
        text = doc[page_number].get_text()
        if char_limit:
            text = text[:char_limit] + "..." if len(text) > char_limit else text
        print(f"📄 Content of page {page_number}:\n")
        print(text)

show_pdf_page_text(pdf_path, 281, char_limit=500)
#%%
def count_eu_penalty(pdf_path, page_number, show_matches=False):
    with fitz.open(pdf_path) as doc:
        if page_number < 0 or page_number >= len(doc):
            print(f"❌ Page {page_number} is out of range (0 to {len(doc) - 1})")
            return
        text = doc[page_number].get_text().lower()
        matches = EU_pattern.findall(text)
        print(f"🔍 Page {page_number} EU keyword penalty: {len(matches)}")
        if show_matches:
            print("Matched terms:", matches)
count_eu_penalty(pdf_path, 246, show_matches=True)