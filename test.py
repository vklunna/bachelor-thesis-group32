import re
import pdfplumber
import camelot
import pandas as pd
import glob
import os

# --- Global Variables & Configuration ---
DR_KEYWORDS_NORMALIZED = []
# Ensure this CSV file is in the same directory as the script, or provide the full absolute path.
DR_LIST_CSV_PATH = "2025-04-06T06-46_export_DR_List.csv" 
ESRS_CODE_RE = re.compile(r"\b[A-Z]{2,4}-\d+\b")

# --- Helper Function for Normalization ---
def normalize_text_for_matching(text_input):
    """Normalizes text for keyword matching: lowercase, strip, remove specific punctuation."""
    if not isinstance(text_input, str):
        return ""
    text_input = text_input.strip().lower()
    text_input = re.sub(r'[.,;:]', '', text_input) # Remove common punctuation
    text_input = re.sub(r'\s+', ' ', text_input)   # Consolidate multiple spaces
    return text_input

# --- Keyword Loading Function ---
def load_dr_keywords(csv_path):
    """Loads and normalizes keywords from the first column of a CSV file."""
    global DR_KEYWORDS_NORMALIZED
    DR_KEYWORDS_NORMALIZED = []
    try:
        if not os.path.exists(csv_path):
            print(f"Warning: Keyword CSV file not found at {csv_path}. Please ensure the path is correct.")
            print(f"Current working directory is: {os.getcwd()}")
            return False # Indicate failure

        # Read only the first column (index 0).
        # The 'squeeze' argument is removed as it's deprecated/removed.
        # If usecols selects a single column, pandas returns a Series by default.
        df_keywords_series = pd.read_csv(csv_path, usecols=[0], header=None)
        
        # The result of usecols=[0] is a DataFrame with one column. Access it by its integer position.
        if df_keywords_series.empty or df_keywords_series.shape[1] == 0:
            print(f"Warning: Keyword CSV file {csv_path} appears to be empty or not parsed correctly into columns.")
            return False

        # Get the first column as a Series
        keywords_series = df_keywords_series.iloc[:, 0]

        raw_keywords = keywords_series.dropna().astype(str).tolist()
        
        for kw in raw_keywords:
            normalized_kw = normalize_text_for_matching(kw)
            if normalized_kw: # Add if not empty after normalization
                DR_KEYWORDS_NORMALIZED.append(normalized_kw)
        
        # Sort keywords by length in descending order for better matching (longer phrases first)
        DR_KEYWORDS_NORMALIZED = sorted(DR_KEYWORDS_NORMALIZED, key=len, reverse=True)
        
        if not DR_KEYWORDS_NORMALIZED:
            print("Warning: No keywords were loaded from the CSV. Please check the CSV file content and path.")
            return False # Indicate failure
        print(f"Successfully loaded and normalized {len(DR_KEYWORDS_NORMALIZED)} keywords from {csv_path}")
        return True # Indicate success

    except pd.errors.EmptyDataError:
        print(f"Warning: Keyword CSV file {csv_path} is empty.")
        return False
    except Exception as e:
        print(f"Error loading or processing keywords from {csv_path}: {e}")
        return False

# --- PDF Processing Functions ---
def find_esrs_pages(pdf_path):
    """Return list of pages (1-based) whose text contains 'ESRS'."""
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").upper() # Keep .upper() for "ESRS" search
                if "ESRS" in text:
                    pages.append(i)
    except Exception as e:
        print(f"Error reading PDF for page finding {pdf_path}: {e}")
    return pages

def extract_tables_on_page(pdf_path, page, flavor):
    """Pull tables from a single page with Camelot flavor."""
    try:
        tables = camelot.read_pdf(pdf_path,
                                pages=str(page),
                                flavor=flavor,
                                strip_text='\n',
                                suppress_stdout=True) # Suppress verbose Camelot output
        return tables
    except Exception as e:
        if "No tables found on page" not in str(e) and "File an issue" not in str(e):
            print(f"  → Camelot {flavor} failed on page {page}: {e}")
        return []

def extract_relevant_rows_from_tables(pdf_path, output_csv="esrs_extracted_rows.csv"):
    """
    Finds pages with "ESRS", extracts tables, and then extracts rows from these tables
    if they contain keywords from the DR_LIST_CSV_PATH.
    """
    if not DR_KEYWORDS_NORMALIZED: # Check if keywords were loaded successfully
        print("Error: DR keywords are not loaded. Cannot proceed with table processing.")
        return None

    esrs_pages = find_esrs_pages(pdf_path)
    if not esrs_pages:
        print(f"PDF {pdf_path}: No pages mentioning ESRS found.")
        return None
    print(f"PDF {pdf_path} has {len(esrs_pages)} pages mentioning ESRS: {esrs_pages}")

    all_matched_rows_data = [] # To store dictionaries of matched rows

    for page_num in esrs_pages:
        print(f"  Processing page {page_num}...")
        tables_lattice = extract_tables_on_page(pdf_path, page_num, "lattice")
        tables_stream = extract_tables_on_page(pdf_path, page_num, "stream")
        
        camelot_tables = []
        if tables_lattice: camelot_tables.extend(tables_lattice)
        if tables_stream: camelot_tables.extend(tables_stream)
        
        unique_tables_dfs = []
        seen_table_hashes = set()
        for tbl in camelot_tables:
            try:
                # Create a simple hash based on shape and first few values
                tbl_hash_content = tuple(tbl.df.head(2).to_string()) # Use more content for hashing
                tbl_hash = (tbl.df.shape, tbl_hash_content)
                if tbl_hash not in seen_table_hashes:
                    unique_tables_dfs.append(tbl.df.copy()) # Store the DataFrame
                    seen_table_hashes.add(tbl_hash)
            except Exception: # If hashing fails, add the table
                unique_tables_dfs.append(tbl.df.copy())


        if not unique_tables_dfs:
            continue # No tables found on this page by either flavor

        print(f"    Found {len(unique_tables_dfs)} unique table(s) on page {page_num}.")
        for table_idx, df_table in enumerate(unique_tables_dfs, start=1):
            # Normalize whitespace in the DataFrame cells
            df_table_cleaned = df_table.applymap(lambda x: re.sub(r"\s+", " ", str(x).strip()) if pd.notna(x) else "")

            found_keywords_in_table = False
            for row_idx, row_series in df_table_cleaned.iterrows():
                # Combine all cell texts in the current row into a single string
                row_text_original = " ".join(cell for cell in row_series.tolist() if cell)
                searchable_row_text = normalize_text_for_matching(row_text_original)

                if not searchable_row_text: # Skip empty rows
                    continue

                matched_keyword_for_this_row = None
                for keyword in DR_KEYWORDS_NORMALIZED:
                    if keyword in searchable_row_text:
                        matched_keyword_for_this_row = keyword
                        break # Found the first (longest) keyword match for this row
                
                if matched_keyword_for_this_row:
                    found_keywords_in_table = True
                    # Create a dictionary from the original row data
                    row_data_dict = df_table.iloc[row_idx].to_dict() 
                    row_data_dict["Matched_Keyword"] = matched_keyword_for_this_row
                    row_data_dict["Source_PDF_Page"] = page_num
                    all_matched_rows_data.append(row_data_dict)
            
            if found_keywords_in_table:
                 print(f"      → Table #{table_idx} on p.{page_num}: Found rows matching DR keywords.")

    if not all_matched_rows_data:
        print(f"⛔ No rows matching DR keywords extracted from {pdf_path}.")
        return None

    result_df = pd.DataFrame(all_matched_rows_data)
    
    if not result_df.empty:
        cols = result_df.columns.tolist()
        # Move new columns to the front
        new_cols_order = ["Source_PDF_Page", "Matched_Keyword"] + [c for c in cols if c not in ["Source_PDF_Page", "Matched_Keyword"]]
        try:
            result_df = result_df[new_cols_order]
        except KeyError:
            print("Note: Could not reorder columns as expected, possibly due to empty initial tables or differing column names.")

    try:
        result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"✅ Extracted {len(result_df)} keyword-matched rows into {output_csv!r}")
    except Exception as e:
        print(f"❌ Error saving CSV to {output_csv}: {e}")
        return None
        
    return result_df

# --- Main Execution ---
def main():
    print("Script started: ESRS Row Extractor based on DR List")
    
    # Load DR keywords globally once. If it fails, halt.
    if not load_dr_keywords(DR_LIST_CSV_PATH):
        print("Halting script as DR keywords could not be loaded. Please check the CSV file and path.")
        return

    pdfs_input_dir = "pdfs" # Directory containing your PDF files
    output_dir = "output_dr_rows" # Directory to save the output CSVs

    # Create directories if they don't exist
    if not os.path.exists(pdfs_input_dir):
        os.makedirs(pdfs_input_dir)
        print(f"Created directory '{pdfs_input_dir}/'. Please place your PDF files here.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory '{output_dir}/' for output files.")

    all_pdf_files_in_dir = sorted(glob.glob(os.path.join(pdfs_input_dir, "*.pdf")))

    if not all_pdf_files_in_dir:
        print(f"No PDF files found in the '{pdfs_input_dir}/' directory.")
        return

    while True: # Loop for processing multiple batches of files
        print(f"\nAvailable PDF files in '{pdfs_input_dir}/':")
        for i, pdf_f_path in enumerate(all_pdf_files_in_dir):
            print(f"  {i+1}. {os.path.basename(pdf_f_path)}")

        user_choice = input(f"\nEnter the number(s) of the PDF(s) to process (e.g., 1 or 1,3,5 or 1-3), "
                            f"'all' to process all listed, or 'q' to quit: ").strip().lower()

        if user_choice == 'q':
            print("Exiting script.")
            break # Exit the main while loop

        selected_files_to_process = []
        if user_choice == 'all':
            selected_files_to_process = all_pdf_files_in_dir
            print("Processing all listed PDF files...")
        else:
            try:
                chosen_indices = set()
                parts = user_choice.split(',')
                for part in parts:
                    part = part.strip()
                    if '-' in part: # Handle ranges like "1-3"
                        start, end = map(int, part.split('-'))
                        if not (1 <= start <= end <= len(all_pdf_files_in_dir)):
                            raise ValueError("Range selection is out of bounds.")
                        chosen_indices.update(range(start - 1, end)) # -1 for 0-based indexing
                    else: # Handle single numbers
                        num = int(part)
                        if not (1 <= num <= len(all_pdf_files_in_dir)):
                            raise ValueError("Number selection is out of bounds.")
                        chosen_indices.add(num - 1) # -1 for 0-based indexing
                
                for index in sorted(list(chosen_indices)):
                     selected_files_to_process.append(all_pdf_files_in_dir[index])

            except ValueError as e:
                print(f"Invalid input: {e}. Please use numbers, ranges (e.g., 1-3), 'all', or 'q'.")
                continue # Go back to asking for input
        
        if not selected_files_to_process:
            if user_choice != 'all': 
                print("No valid files selected with the given input. Please try again.")
            continue 

        print(f"\nProcessing {len(selected_files_to_process)} selected file(s)...")
        for pdf_full_path in selected_files_to_process:
            print(f"\n── Processing PDF: {pdf_full_path}")
            base_name = os.path.basename(pdf_full_path)
            # Sanitize filename for CSV
            csv_filename_base = re.sub(r'\.pdf$', '', base_name, flags=re.IGNORECASE)
            output_csv_name = f"{csv_filename_base}_dr_matched_rows.csv"
            full_output_path = os.path.join(output_dir, output_csv_name)
            
            extract_relevant_rows_from_tables(pdf_full_path, output_csv=full_output_path)
        
        process_more = input("\nProcess another batch of files? (yes/no): ").strip().lower()
        if process_more not in ['yes', 'y']:
            print("Exiting script.")
            break 

    print("\nScript finished.")

if __name__ == "__main__":
    main()
