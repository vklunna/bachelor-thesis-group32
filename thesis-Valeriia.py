#%%
import re, pdfplumber
from typing import List, Tuple, Set, Optional
from collections import defaultdict 
import pandas as pd
from openai import OpenAI
#%%
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

def ask_llm_to_pick_disclosure_table(table_texts, model = "gpt-3.5-turbo"):
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