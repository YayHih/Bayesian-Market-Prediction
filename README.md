# Bayesian-Market-Prediction

# 📈 S&P 500 Stock Price Analysis  

## Overview  
This Python script reads a list of S&P 500 companies, fetches historical stock data from Yahoo Finance, analyzes significant price changes, and saves the results to a CSV file.  

## Features  
✅ Reads company details from a tab-separated file (`sandp500.txt`).  
✅ Fetches historical stock data using Yahoo Finance API.  
✅ Identifies significant daily price changes (default: ±5%).  
✅ Saves results to `significant_changes.csv` and logs missing tickers in `not_found.txt`.  

## Requirements  
- **Python 3.x**  
- Install dependencies:  
  ```bash
  pip install requests pandas beautifulsoup4 yfinance
Usage
Ensure sandp500.txt is in the same directory (or provide the file path when prompted).
Run the script:
bash
Copy
Edit
python script.py
Results will be saved in significant_changes.csv.
Configuration
Adjust the price change threshold in analyze_price_changes() (threshold=0.05 for 5%).
Modify test_limit in main() to process only a subset of companies for testing.
Output Files
📂 significant_changes.csv – Records notable stock price movements with company details.
📂 not_found.txt – Lists tickers for which data could not be retrieved.

Example Output
Sample significant_changes.csv:
Ticker	Company Name	Date	Open Price	Close Price	% Change
AAPL	Apple Inc.	2025-03-14	170.50	160.00	-6.15%
TSLA	Tesla, Inc.	2025-03-14	200.00	210.50	+5.25%
Sample not_found.txt:
pgsql
Copy
Edit
XYZ Corp (XYZ) - No data found
ABC Inc. (ABC) - No data found
Error Handling
If a ticker is not found, it is logged in not_found.txt.
If the Yahoo Finance API rate-limits requests, consider adding a delay between API calls.
License
This project is licensed under the MIT License.

Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss the proposed modifications.
