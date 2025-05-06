import pandas as pd
import os
from PyPDF2 import PdfReader
import re
import logging
from openai import OpenAI
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ESRSTableExtractor:
    def __init__(self, api_key=None, base_url=None):
        """Initialize with optional API key and base URL for proxy"""
        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass key to constructor.")
            
        self.model = "gpt-3.5-turbo"
        self.logger = self.setup_logging()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url or os.getenv('OPENAI_BASE_URL')
        )
        self._verify_model_access()

    def setup_logging(self):
        """Configure logging for the extractor"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _verify_model_access(self):
        """Verify access to the specified model"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            self.logger.info(f"Successfully connected to model: {self.model}")
            return True
        except Exception as e:
            self.logger.error(f"Error accessing model {self.model}: {str(e)}")
            if self.model == "gpt-4":
                self.logger.info("Falling back to gpt-3.5-turbo")
                self.model = "gpt-3.5-turbo"
                return self._verify_model_access()
            return False
    
    def extract_from_directory(self, pdf_directory, output_file="esrs_references.csv"):
        """Process all PDFs in a directory"""
        all_results = []
        
        for filename in os.listdir(pdf_directory):
            if filename.endswith('.pdf'):
                file_path = os.path.join(pdf_directory, filename)
                logging.info(f"Processing {filename}")
                
                try:
                    company_results = self.process_pdf(file_path)
                    if company_results:  # Only add if results were found
                        company_name = os.path.splitext(filename)[0]
                        # Add company name to each result
                        for result in company_results:
                            result['company'] = company_name
                        all_results.extend(company_results)
                except Exception as e:
                    logging.error(f"Error processing {filename}: {str(e)}")
        
        # Create DataFrame only if results were found
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(output_file, index=False)
            return results_df
        else:
            logging.warning("No results were found")
            return pd.DataFrame()  # Return empty DataFrame instead of None
    
    def process_pdf(self, pdf_path):
        """Process a single PDF file to extract ESRS references"""
        # Step 1: Identify potential pages with ESRS references
        potential_pages = self.find_potential_pages(pdf_path)
        logging.info(f"Found {len(potential_pages)} potential pages with ESRS references")
        
        # Step 2: Extract and analyze content from potential pages
        results = []
        
        for page_num in potential_pages:
            page_text = self.extract_page_text(pdf_path, page_num)
            
            # Check if this looks like a cross-reference table
            if self.is_likely_esrs_table(page_text):
                # Extract structured data using LLM
                page_results = self.extract_with_llm(page_text, page_num)
                if page_results:
                    results.extend(page_results)
        
        return results
    
    def find_potential_pages(self, pdf_path):
        """Find pages likely to contain ESRS reference tables"""
        potential_pages = []
        reader = PdfReader(pdf_path)
        
        # Expanded keywords that might indicate ESRS reference tables
        keywords = [
            "ESRS", 
            "cross-reference",
            "CSRD",
            "ESG",
            "E1-",
            "E2-",
            "S1-",
            "G1-"
        ]
        
        # ESRS code pattern
        esrs_pattern = re.compile(r'[ESG]\d{1,2}[-–]\d{1,2}')
        
        # Search through all pages
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text().lower()  # Case-insensitive matching
            
            # Count keyword matches
            keyword_matches = sum(1 for kw in keywords if kw.lower() in text)
            
            # Check for ESRS code patterns
            has_esrs_codes = bool(esrs_pattern.search(text))
            
            # Add page if it has multiple keywords or ESRS codes
            if keyword_matches >= 2 or has_esrs_codes:
                potential_pages.append(page_num)
                logging.info(f"Found potential ESRS content on page {page_num+1}")
        
        return potential_pages
    
    def extract_page_text(self, pdf_path, page_num):
        """Extract text from a specific page"""
        reader = PdfReader(pdf_path)
        if page_num < len(reader.pages):
            return reader.pages[page_num].extract_text()
        return ""

    def is_likely_esrs_table(self, text):
        """Determine if text likely contains an ESRS reference table"""
        patterns = [
            r"Disclosure\s+Requirement\s+[A-Z]{2,3}-?[0-9]*",  # More general
            r"Minimum disclosure requirement",                 # Also in your image
            r"Reference",                                      # Any mention of "Reference"
            r"Assurance\s+level",                              # Found in your table
            r"CSRD\s+Topic",                                   # Found in your table
            r"Double Materiality",                             # Common in these tables
            r"SBM-[0-9]+",                                     # Strategy Business Model codes
            r"GOV-[0-9]+",                                     # Governance-related codes
            r"ESRS\s+[A-Z0-9\-]+",                             # More general ESRS reference
        ]

        # Count how many of these patterns are found
        matches = sum(bool(re.search(p, text, re.IGNORECASE)) for p in patterns)

        return matches >= 2  # Loosened logic: if 2+ patterns found, it's likely a table
    
    def extract_with_llm(self, text, page_num):
        """Use LLM to extract structured data from text"""
        try:
            prompt = f"""
            You are an ESRS disclosure analyzer. Extract ALL ESRS disclosures from the following text from page {page_num + 1}.
            Look for all of these disclosure types:
            - Environmental (E1-1 to E5-7)
            - Social (S1-1 to S4-7) 
            - Governance (G1-1 to G2-10)
            
            For each disclosure found, extract:
            1. The disclosure code (e.g., E1-1, S2-3)
            2. The disclosure title/name
            3. The page reference if available
            
            Return the data in this JSON format:
            {{
                "items": [
                    {{
                        "code": "E1-1",
                        "title": "Climate change mitigation policy",
                        "page_reference": "p. 45"
                    }},
                    // more items
                ]
            }}
            
            Here's the text to analyze:
            {text}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            logging.debug(f"LLM response: {result}")
            
            # Parse JSON response
            parsed_result = json.loads(result)
            
            # Extract the items array
            items = parsed_result.get("items", [])
            
            # Add page number to each item
            for item in items:
                item["source_page"] = page_num + 1
                
            return items
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error on page {page_num + 1}: {str(e)}")
            logging.error(f"Raw response: {result}")
            return []
        except Exception as e:
            logging.error(f"API error on page {page_num + 1}: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    try:
        # Create a .env file with OPENAI_API_KEY=your_api_key or pass it directly
        extractor = ESRSTableExtractor()
        
        # Make sure this directory exists and contains PDF files
        pdf_dir = "./annual_reports"
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
            logging.warning(f"Created directory {pdf_dir}. Please add PDF files to process.")
        
        results_df = extractor.extract_from_directory(pdf_dir)
        if not results_df.empty:
            print(results_df)
            print(f"Results saved to esrs_references.csv")
    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")