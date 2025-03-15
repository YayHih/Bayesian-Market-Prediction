import requests
import pandas as pd
import time
import random
import os
from bs4 import BeautifulSoup
from datetime import datetime

# Function to read the S&P 500 companies from the file
def read_sp500_companies(filename):
    companies = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                if parts and len(parts) >= 1:
                    # Just extract the ticker symbol (first column)
                    symbol = parts[0].strip()
                    if symbol and symbol != "Symbol" and not symbol.startswith("#"):
                        companies.append({"Symbol": symbol})
        
        print(f"Successfully read {len(companies)} company ticker symbols")
        return companies
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# Function to fetch historical data for a ticker
def fetch_historical_data(ticker, period="1y"):
    print(f"Fetching data for {ticker}...")
    
    # Configure headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }
    
    # Use Yahoo Finance API endpoint
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={period}&interval=1d"
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch data for {ticker}: HTTP {response.status_code}")
            return None
        
        data = response.json()
        
        # Check if we got valid data
        if not data or 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
            print(f"No valid data found for {ticker}")
            return None
        
        # Extract the relevant data
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        # Create a DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': quotes.get('open', []),
            'high': quotes.get('high', []),
            'low': quotes.get('low', []),
            'close': quotes.get('close', []),
            'volume': quotes.get('volume', [])
        })
        
        # Convert timestamp to datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Handle missing values
        df = df.dropna(subset=['close'])
        
        print(f"Successfully fetched {len(df)} data points for {ticker}")
        return df
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Function to analyze price changes and find significant movements
def analyze_price_changes(df, ticker, threshold=0.10):
    if df is None or len(df) < 2:
        return []
    
    significant_changes = []
    
    # Calculate day-over-day changes
    df = df.sort_values('date')
    df['prev_close'] = df['close'].shift(1)
    df['change'] = (df['close'] - df['prev_close']) / df['prev_close']
    
    # Find significant changes
    significant_df = df[abs(df['change']) > threshold].copy()
    
    for _, row in significant_df.iterrows():
        try:
            event = {
                'Date': row['date'].strftime('%Y-%m-%d'),
                'Symbol': ticker,
                'Previous Close': round(row['prev_close'], 2),
                'New Close': round(row['close'], 2),
                'Change (%)': round(row['change'] * 100, 2)
            }
            significant_changes.append(event)
            print(f"Found significant change for {ticker} on {event['Date']}: {event['Change (%)']}%")
        except Exception as e:
            print(f"Error processing significant change for {ticker}: {e}")
    
    return significant_changes

# Main function
def main():
    try:
        # Read company data
        file_path = 'sandp500.txt'
        if not os.path.exists(file_path):
            file_path = input("Enter the path to your S&P 500 companies file: ")
            
        companies = read_sp500_companies(file_path)
        if not companies:
            print("No companies found. Exiting...")
            return
        
        # Process companies
        all_significant_changes = []
        not_found = []
        
        # Limit to first 5 companies for testing
        test_limit = 502
        print(f"Testing with first {test_limit} companies...")
        
        for i, company in enumerate(companies):
            if i >= test_limit:
                print(f"Completed test run with {test_limit} companies")
                break
                
            ticker = company["Symbol"]
            
            # Fetch historical data
            df = fetch_historical_data(ticker)
            
            if df is not None:
                # Analyze for significant changes
                changes = analyze_price_changes(df, ticker)
                if changes:
                    all_significant_changes.extend(changes)
            else:
                not_found.append(ticker)
                
            # Wait between requests to avoid rate limiting
            delay = random.uniform(1, 3)
            print(f"Waiting {delay:.1f} seconds before next request...")
            time.sleep(delay)
        
        # Save results
        if all_significant_changes:
            pd.DataFrame(all_significant_changes).to_csv("significant_changes.csv", index=False)
            print(f"Saved {len(all_significant_changes)} significant changes to CSV")
            
        if not_found:
            with open("not_found.txt", "w") as f:
                for ticker in not_found:
                    f.write(f"{ticker}\n")
            print(f"Saved {len(not_found)} not found tickers to file")
    
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
