#%%
import re, pdfplumber
from typing import List, Tuple, Set, Optional
from collections import defaultdict 
import pandas as pd
from openai import OpenAI
#%%
#Method 1
client = OpenAI(api_key="your api")

keywords=["ersr", "disclosure requirement", "location", "standard section", "index of", "chapter", "section", "page", "page number", "reference to", "reference table", "content index", "ESRS indices"]
DR_pattern=re.compile(r"\b(?:E|S|G|GOV|BP|SMB|IRO|MDR|ESRS)-\d+\b", re.IGNORECASE)
EU_pattern = re.compile(r"european union|eu legislation|eu taxonomy| eu regulation", re.IGNORECASE)

def extract_text_and_tables(pdf_path):
    pages_data=[]
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            pages_data.append((text, tables))
    return pages_data

def score_pages_by_content(pages_data):
    results = []
    for i, (text, tables) in enumerate(pages_data):
        text_lower=text.lower()
        keyword_score=sum(kw in text_lower for kw in keywords)
        matched_drs = [m.group(0).upper() for m in DR_pattern.finditer(text)]
        found_drs: Set[str]=set(m.group(0).upper() for m in DR_pattern.finditer(text))
        unique_dr_count=len(found_drs)
        total_dr_mentions = len(matched_drs)
        table_bonus = int(len(tables)>0)
        total_score=keyword_score + unique_dr_count + table_bonus+total_dr_mentions
        results.append((i+1, total_score, keyword_score, unique_dr_count, total_dr_mentions, table_bonus))
    df = pd.DataFrame(results, columns=["page", "total_score", "keyword_hits", "unique_dr_codes", "total_dr_mentions", "has_table"])
    df = df[df.total_score>0].sort_values("total_score", ascending=False).reset_index(drop=True)
    return df

def extract_candidate_page(score_df):
    top_page = int(score_df.iloc[0]["page"])
    neighbors = [top_page-1, top_page, top_page+1]
    return [p for p  in neighbors if p>0]

def extract_all_tables_from_page(pdf_path, page_number):
    all_tables=[]
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number-1]
        for raw_table in page.extract_tables() or []:
            if raw_table and len(raw_table)>=2:
                all_tables.append((raw_table[0], raw_table[1:]))
        try:
            horiz = page.extract_table({"vertical_strategy": "text", "horizontal_strategy":"lines", "snap_tolerance": 3})
            if horiz and len(horiz)>=2:
                all_tables.append((horiz[0], horiz[1:]))
        except:
            pass
        words = page.extract_words() or []
        rows = defaultdict(list)
        for word in words:
            y = round(word["top"], -1)
            rows[y].append((word["x0"], word["text"]))
        structured = []
        for y in sorted(rows.keys()):
            line = [text for _, text in sorted(rows[y], key = lambda x: x[0])]
            structured.append(line)
        if len(structured)>=2:
            all_tables.append((structured[0], structured[1:]))
    return all_tables

def table_to_text(headers, rows):
    if not headers or not isinstance(headers, list) or not rows:
        return ""  
    full_text = "\n".join([" ".join(map(str, row)) for row in rows])
    if EU_pattern.search(full_text):
        return ""
    output = " | ".join(map(str, headers)) + "\n"
    output += " | ".join(["---"] * len(headers)) + "\n"
    for row in rows:
        if not isinstance(row, list):
            continue  # skip malformed row
        output += " | ".join(map(str, row)) + "\n"
    return output.strip()

def ask_llm_to_pick_disclosure_table(table_texts, model = "gpt4o"):
    prompt = """You are sustainability analyst going through annual financial reports. You receive several extracted tables, your task is to identify the table(s) (if any) that maps ESRS disclosure requirements to where they are found in the report (such as page numbers or section references). There miight be several cases:
    1. One whole table in one page - then return that table
    2. One whole table that is spread in several consecutice pages - then return all those pages.
    3. There might be small table in one page (example page 443 has table for climate change requirement, page 443 has table for pollution) - return the page.
    4. The mix of previous scenarios
    
    When you find a table(s) please check the neighbouring table whether they have the same structure and context and also report them. And then again, check neighbouring pages (before and after the foun ones) just to make sure.
    
    Return only the table(s) that best fit for this task. If no table identified return \"No reference table\"."""
    for i, text in enumerate(table_texts):
        if text.strip():
            prompt+=f"\n\nTable {i+1}:{text}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant for ESG disclosure detection"},
            {"role": "user", "content": prompt.strip()}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

pdf_path = "Annual Report 2024_Zalando SE_EN_250503_s.pdf"
pages_data = extract_text_and_tables(pdf_path)
scores_df = score_pages_by_content(pages_data)
print("\nTop Candidate Pages for Disclosure Table:")
print(scores_df.head())

top_and_neighbor_pages = extract_candidate_page(scores_df)
table_texts = []
for page in top_and_neighbor_pages:
    tables = extract_all_tables_from_page(pdf_path, page)
    table_texts.extend([f"Page {page}:{table_to_text(header, rows)}" for header, rows in tables if header and isinstance(header, list)])

print(f"\nSending {len(table_texts)} tables")

chosen_table = ask_llm_to_pick_disclosure_table(table_texts)
print("\n LLM Response:")
print(chosen_table)

# %%
selected_table_text1 = table_texts[3]
print(selected_table_text1)

#%%
#%%






#Method 2

from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from unstructured.partition.pdf import partition_pdf
from collections import defaultdict
from pypdf import PdfReader
import pdfplumber
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor
import time
#%%

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

keywords = ["disclosure requirement", "location", "standard section", 
           "reference table", "content index", "ESRS indices", "reference in the report"]
DR_pattern = re.compile(r"\b(?:E|S|G|GOV|BP|SMB|IRO|MDR|ESRS)[-\s]\d+\b", re.IGNORECASE)
EU_pattern = re.compile(r"\b(european union|eu legislation|eu taxonomy| eu regulation| datapoint| sfdr| indicator| delegated regulation)\b", re.IGNORECASE)

def get_accurate_page_count(pdf_path):
    """Get page count using the most reliable method"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception as e:
        print(f"PDFPlumber page count failed: {e}")
        try:
            return len(PdfReader(pdf_path).pages)
        except Exception as e:
            print(f"PyPDF page count failed: {e}")
            return None

def extract_text_with_proper_pagenums(pdf_path):
    """Extract text with guaranteed proper page numbers"""
    page_texts = []
    
    # Method 1: Try pdfplumber first for accurate text and page numbers
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                page_texts.append((page_num, text))
        return page_texts
    except Exception as e:
        print(f"PDFPlumber extraction failed: {e}")
    
    # Method 2: Fallback to PyPDF
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            page_texts.append((page_num, text))
        return page_texts
    except Exception as e:
        print(f"PyPDF extraction failed: {e}")
    
    # Method 3: Final fallback to Unstructured
    print("Falling back to Unstructured IO...")
    elements = partition_pdf(filename=pdf_path, strategy="fast", languages=["eng"])
    page_dict = defaultdict(str)
    for el in elements:
        if hasattr(el.metadata, 'page_number'):
            page_num = el.metadata.page_number + 1  # Convert to 1-based
            page_dict[page_num] += el.text + "\n"
    return sorted(page_dict.items())

def score_page_by_content(page_texts):
    """Score pages based on content relevance with proper page numbers"""
    results = []
    for page_num, text in page_texts:
        if not text.strip():
            continue
            
        text_lower = text.lower()
        keyword_score = sum(kw in text_lower for kw in keywords)
        drs = DR_pattern.findall(text)
        unique_drs = set(drs)
        score = 1.5 * keyword_score + len(unique_drs) + len(drs)
        
        results.append({
            "page": page_num,
            "total_score": score,
            "text": text[:500] + "..." if len(text) > 500 else text,
            "keyword_hits": keyword_score,
            "unique_dr_codes": len(unique_drs),
            "total_dr_mentions": len(drs)
        })
    
    df = pd.DataFrame(results)
    return df.sort_values("total_score", ascending=False).reset_index(drop=True)

def is_likely_disclosure_table(text):
    """Determine if text is likely an ESRS disclosure table with multiple checks"""
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    if not text:
        return False
    
    lines = text.split('\n')
    if len(lines) < 3: 
        return False
    
    has_dr_codes = bool(DR_pattern.search(text))
    
    has_tabular_format = any('\t' in line or 
                           (len(line.split()) > 3 and len(line) > 20) 
                           for line in lines)
    
    keywords_present = any(
        kw in text.lower() 
        for kw in ["disclosure", "requirement", "reference", "location", "index"]
    )
    
    has_numbering = any(
        re.search(r"\b\d+\.\d+\b", line) or  # E.g., "1.1", "2.3"
        re.search(r"\b[ESG]-\d+\b", line)    # E.g., "E-1", "S-2"
        for line in lines
    )
    
    is_eu_content = bool(EU_pattern.search(text))
    is_page_header = any(
        line.lower().strip() in ["page", "page number", "continued"] 
        for line in lines[:2]
    )
    
    return (
        (has_dr_codes or (has_tabular_format and keywords_present)) and
        not is_eu_content and
        not is_page_header and
        (has_numbering or has_tabular_format)
    )
    

def extract_tables_from_pdf(pdf_path, candidate_pages):
    """Robust table extraction with proper page numbers"""
    all_tables = []
    # Method 1: PDFPlumber table extraction
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num in candidate_pages:
                if page_num > len(pdf.pages):
                    continue
                    
                page = pdf.pages[page_num - 1]
                
                tables = page.extract_tables()
                for table in tables:
                    table_text = "\n".join(["\t".join(filter(None, row)) for row in table])
                    if is_likely_disclosure_table(table_text):
                        all_tables.append({
                            "page": page_num,
                            "source": "PDFPlumber Table",
                            "text": table_text
                        })
                
                words = page.extract_words() or []
                rows = defaultdict(list)
                for word in words:
                    y = round(word["top"], -1)
                    rows[y].append((word["x0"], word["text"]))
                
                structured_text = []
                for y in sorted(rows.keys()):
                    line = [text for _, text in sorted(rows[y], key=lambda x: x[0])]
                    structured_text.append(" ".join(line))
                
                visual_text = "\n".join(structured_text)
                if is_likely_disclosure_table(visual_text):
                    all_tables.append({
                        "page": page_num,
                        "source": "Visual Reconstruction", 
                        "text": visual_text
                    })
    except Exception as e:
        print(f"PDFPlumber table extraction failed: {e}")
    
    # Method 2: Unstructured IO as fallback
    if not all_tables:
        print("Attempting Unstructured IO extraction...")
        try:
            elements = partition_pdf(filename=pdf_path, strategy="fast", languages=["eng"])
            for el in elements:
                if hasattr(el.metadata, 'page_number') and el.category == "Table":
                    page_num = el.metadata.page_number + 1
                    if page_num in candidate_pages and is_likely_disclosure_table(el.text):
                        all_tables.append({
                            "page": page_num,
                            "source": "Unstructured IO",
                            "text": el.text
                        })
        except Exception as e:
            print(f"Unstructured IO failed: {e}")
    
    return all_tables


def rerank_with_llm(table_texts, model="gpt4o"):
    """Final verification step with LLM"""
    if not table_texts:
        return "No tables found for LLM analysis"
    
    prompt = """Analyze these table candidates and identify which is most likely the 
    ESRS disclosure requirements reference table. Consider:
    - Presence of disclosure codes (E/S/G-XXX)
    - Table structure (rows/columns)
    - Completeness of information
    - Presence of standard section references
    
    Options:\n"""
    
    for i, table in enumerate(table_texts, 1):
        preview = "\n".join(table["text"].split("\n")[:10])  # First 10 lines
        prompt += f"\n\nOption {i} (Page {table['page']}):\n{preview}\n..."
    
    prompt += "\n\nRespond with: 'Best option: X (Page Y)' and a 1-sentence explanation."
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You're an ESG reporting expert analyzing disclosure tables."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM analysis failed: {str(e)}"

def identify_table_pages(pdf_path, initial_page, max_pages):
    """Identify all consecutive pages belonging to the same table"""
    table_pages = [initial_page]
    
    next_page = initial_page + 1
    while next_page <= max_pages:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[next_page - 1]
            text = page.extract_text() or ""
            
            if (DR_pattern.search(text) and 
                any(kw in text.lower() for kw in ["disclosure", "reference", "location"])):
                table_pages.append(next_page)
                next_page += 1
            else:
                break
    
    return table_pages
def main(pdf_path):
    """Expands detection from top candidate both forward and backward, stopping at EU-heavy pages"""
    print("\n=== Starting Focused Table Detection ===")
    start_time = time.time()

    try:
        with pdfplumber.open(pdf_path) as pdf:
            max_pages = len(pdf.pages)
            print(f"Processing {max_pages} pages...")

            top_candidate = None
            for page_num in range(1, max_pages + 1):
                page = pdf.pages[page_num - 1]
                text = page.extract_text() or ""

                dr_count = len(DR_pattern.findall(text))
                kw_score = sum(kw in text.lower() for kw in keywords)
                table_score = 1 if ('\t' in text or len(page.extract_tables()) > 0) else 0
                total_score = dr_count + kw_score * 1.5 + table_score

                if not top_candidate or total_score > top_candidate[1]:
                    top_candidate = (page_num, total_score)

            if not top_candidate:
                return ["No suitable candidates found"]

            core_page = top_candidate[0]
            print(f"\nTop candidate found on page {core_page} (score: {top_candidate[1]})")

            table_pages = set([core_page])

            for page_num in range(core_page + 1, max_pages + 1):
                page = pdf.pages[page_num - 1]
                text = page.extract_text() or ""

                eu_hits = len(EU_pattern.findall(text))
                if eu_hits > 3:
                    print(f"Page {page_num} skipped due to {eu_hits} EU keyword hits")
                    break

                table_pages.add(page_num)
                print(f"Connected page {page_num} (forward)")

            for page_num in range(core_page - 1, max(core_page - 5, 0), -1):
                page = pdf.pages[page_num - 1]
                text = page.extract_text() or ""

                eu_hits = len(EU_pattern.findall(text))
                if eu_hits > 3:
                    print(f"Page {page_num} skipped due to {eu_hits} EU keyword hits")
                    break

                if (DR_pattern.search(text) or 
                    any(kw in text.lower() for kw in ["continued", "table", "disclosure", "requirement", "location", "reference"])):
                    table_pages.add(page_num)
                    print(f"Connected page {page_num} (backward)")
                else:
                    break

            table_pages = sorted(table_pages)
            ranges = []
            start = table_pages[0]
            for i in range(1, len(table_pages)):
                if table_pages[i] != table_pages[i - 1] + 1:
                    ranges.append(f"{start}-{table_pages[i - 1]}" if start != table_pages[i - 1] else str(start))
                    start = table_pages[i]
            ranges.append(f"{start}-{table_pages[-1]}" if start != table_pages[-1] else str(start))

            print(f"\nCompleted in {time.time() - start_time:.1f} seconds")
            return ranges

    except Exception as e:
        print(f"Error: {str(e)}")
        return ["Processing failed"]


if __name__ == "__main__":
    pdf_path = "2024 Adyen Annual Report.pdf"
    print("\n=== Focused Table Detection ===")
    results = main(pdf_path)

    print("\n=== FINAL TABLE RANGE ===")
    print("Pages:", ", ".join(results))

#%%
with pdfplumber.open(pdf_path) as pdf:
    page = pdf.pages[113]  
    text = page.extract_text()
    print(text)