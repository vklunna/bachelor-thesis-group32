from KeyW_LLM import ESRSTableExtractor

# Example usage
if __name__ == "__main__":
    extractor = ESRSTableExtractor(
        api_key="sk-vvTzU6czhiPnaLYz2TM9qg",
        base_url="your_proxy_base_url"  # Optional for LiteLLM proxy
    )
    results_df = extractor.extract_from_directory("./annual_reports")