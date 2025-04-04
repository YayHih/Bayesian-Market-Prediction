import csv
import time
import random
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urlparse
import re
import logging
import backoff

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of known financial and news domains
FINANCIAL_NEWS_DOMAINS = [
    'bloomberg.com', 'reuters.com', 'cnbc.com', 'wsj.com', 'ft.com', 'marketwatch.com',
    'barrons.com', 'forbes.com', 'fool.com', 'businessinsider.com', 'finance.yahoo.com',
    'investing.com', 'seekingalpha.com', 'money.cnn.com', 'thestreet.com', 'benzinga.com',
    'investors.com', 'zacks.com', 'morningstar.com', 'tradingview.com', 'nytimes.com',
    'washingtonpost.com', 'apnews.com', 'bbc.com', 'aljazeera.com', 'theguardian.com',
    'fool.com', 'investopedia.com', 'nasdaq.com', 'financialtimes.com', 'economist.com'
]

# Keywords that indicate generic stock valuation articles rather than news
STOCK_VALUATION_KEYWORDS = [
    'undervalued', 'overvalued', 'stock quote', 'technical analysis', 
    'buy or sell', 'stock rating', 'price target', 'analyst rating',
    'is it a buy', 'stock grade', 'stock evaluation', 'stock performance',
    'stock analysis', 'buy the dip', 'stock pick'
]

def is_news_site(url):
    """Check if URL is from a news or financial site."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    
    # Remove www. prefix if present
    if domain.startswith('www.'):
        domain = domain[4:]
    
    # Check if domain or any subdomain is in our list
    return any(domain.endswith(news_domain) for news_domain in FINANCIAL_NEWS_DOMAINS)

def is_generic_stock_article(title, snippet):
    """Check if the article is a generic stock valuation piece rather than news."""
    combined_text = f"{title} {snippet}".lower()
    
    # Check for stock valuation keywords
    for keyword in STOCK_VALUATION_KEYWORDS:
        if keyword.lower() in combined_text:
            # If it contains news-specific terms as well, it might still be news
            news_indicators = ['announce', 'report', 'launch', 'unveil', 'release', 
                              'acquire', 'merger', 'earnings', 'dividend', 'quarter']
            
            # If it has news indicators, don't filter it out
            if any(indicator in combined_text for indicator in news_indicators):
                return False
            
            # Otherwise, it's likely just a stock valuation article
            return True
    
    # Check for URL containing "quote" at the end
    if "quote" in combined_text and not ("quote" in combined_text and "said" in combined_text):
        return True
        
    return False

def is_date_relevant(article_date_str, target_date_str, window_days=7):
    """
    Check if the article date is within a specified window of the target date.
    Returns True if we can't determine the date (will check content later).
    """
    try:
        # Try to parse the target date
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
                
        # Try to parse the article date with multiple formats
        for fmt in ['%Y-%m-%d', '%B %d, %Y', '%b %d, %Y']:
            try:
                article_date = datetime.strptime(article_date_str, fmt)
                # Check if the article date is within the window
                delta = abs((article_date - target_date).days)
                return delta <= window_days
            except (ValueError, TypeError):
                continue
                
        # If we can't parse the date, return True and filter later based on content
        return True
    except Exception:
        # If any error occurs, return True and we'll filter based on content later
        return True

@backoff.on_exception(backoff.expo, Exception, max_tries=5, max_time=300)
def search_company_news(company_name, symbol, date_str, sector=None, target_count=8):
    """Search for news articles about a company, with backoff for rate limiting."""
    articles = []
    all_results = []  # Store all results before filtering
    
    # Parse the date for filtering
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        # Format date as string to include in the search
        formatted_date = date.strftime('%B %Y')  # e.g. "March 2023"
    except ValueError:
        formatted_date = date_str
    
    # Try multiple search queries to get more results
    search_queries = [
        f"{company_name} {symbol} stock news {formatted_date}",
        f"{company_name} {symbol} earnings {formatted_date}",
        f"{company_name} {symbol} financial news {formatted_date}",
        f"{company_name} {symbol} press release {formatted_date}"
    ]
    
    # Add sector-specific query if sector is provided
    if sector:
        search_queries.append(f"{company_name} {symbol} {sector} news {formatted_date}")
    
    for query in search_queries:
        if len(all_results) >= 30:  # Collect a good pool of results to filter from
            break
            
        logger.info(f"Searching for: {query}")
        
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=15)  # Get more results per query
                
                if not results:
                    logger.warning(f"No results found for query: {query}")
                    # Add a random delay before next query
                    time.sleep(random.uniform(3, 7))
                    continue
                
                # Add results to our pool
                all_results.extend(results)
                
                # Add a random delay to avoid rate limiting
                time.sleep(random.uniform(3, 7))
            
        except Exception as e:
            logger.error(f"Error searching for {company_name} with query '{query}': {e}")
            # Add a longer delay after an error
            time.sleep(random.uniform(15, 30))
    
    # Filter and process the collected results
    processed_count = 0
    for result in all_results:
        # Check if it's a news site
        if is_news_site(result['href']):
            # Check for duplicates
            if not any(a['url'] == result['href'] for a in articles):
                # Skip generic stock valuation articles rather than news
                if is_generic_stock_article(result['title'], result['body']):
                    logger.info(f"Skipping generic stock valuation article: {result['title']}")
                    continue
                
                # Get the date from the snippet or title if possible
                date_match = re.search(r'(\d{1,2}\s+[A-Za-z]+\s+\d{4}|\d{4}-\d{2}-\d{2}|[A-Za-z]+\s+\d{1,2},\s+\d{4})', 
                                      result['title'] + ' ' + result['body'])
                article_date = date_match.group(1) if date_match else None
                
                # Only add if the date is relevant or we can't determine
                if not article_date or is_date_relevant(article_date, date_str):
                    articles.append({
                        'title': result['title'],
                        'url': result['href'],
                        'snippet': result['body'],
                        'apparent_date': article_date
                    })
                    processed_count += 1
                    
                    # Log the article we're keeping
                    logger.info(f"Found article {processed_count}: {result['title']}")
                    
                    if processed_count >= target_count:
                        break
    
    # If we didn't find enough articles, log this information
    if processed_count < target_count:
        logger.warning(f"Only found {processed_count}/{target_count} articles for {company_name}")
        
    return articles

def verify_date_in_content(content, target_date_str, company_name, symbol, window_days=7):
    """
    Verify if the article content mentions dates close to our target date.
    Returns True if it seems relevant, False otherwise.
    """
    if not content:
        return False
        
    try:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    except ValueError:
        # If we can't parse the target date, we can't verify
        return True
    
    # Look for dates in content
    date_patterns = [
        r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})',  # 15 March 2023
        r'(\d{4}-\d{2}-\d{2})',            # 2023-03-15
        r'([A-Za-z]+\s+\d{1,2},\s+\d{4})'  # March 15, 2023
    ]
    
    found_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            try:
                for fmt in ['%d %B %Y', '%Y-%m-%d', '%B %d, %Y']:
                    try:
                        article_date = datetime.strptime(match, fmt)
                        delta = abs((article_date - target_date).days)
                        if delta <= window_days:
                            return True
                        found_dates.append(article_date)
                    except ValueError:
                        continue
            except Exception:
                continue
    
    # If we found dates but none are close, check if the article mentions the company
    # and relevant financial terms
    if found_dates and (
        company_name.lower() in content.lower() or 
        symbol.lower() in content.lower()
    ):
        financial_terms = ['earnings', 'stock', 'shares', 'investors', 'market', 
                           'trading', 'quarterly', 'financial', 'results']
        if any(term in content.lower() for term in financial_terms):
            # If at least some financial terms are mentioned with the company, it might be relevant
            return True
    
    # Default to false if we couldn't verify relevance
    return False

def extract_article_text(url):
    """Extract the main text content from a news article with error handling."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, nav, header, footer elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
            
            # Try to find the article content using common article containers
            article_content = None
            for selector in ['article', '.article', '.article-content', '.content', '.story', '.post-content', 'main']:
                article = soup.select_one(selector)
                if article:
                    paragraphs = article.find_all('p')
                    if len(paragraphs) >= 3:  # Make sure we have at least 3 paragraphs
                        article_content = ' '.join(p.get_text().strip() for p in paragraphs)
                        break
            
            # If we couldn't find an article container, fall back to all paragraphs
            if not article_content:
                # Get all paragraphs with substantial content
                paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text()) > 50]
                article_content = ' '.join(paragraphs)
            
            # Cleanup: remove extra whitespace and normalize
            if article_content:
                article_content = re.sub(r'\s+', ' ', article_content).strip()
                # Remove advertisements or common non-content text
                article_content = re.sub(r'(advertisement|subscribe now|sign up|you may also like|related stories)', 
                                        '', article_content, flags=re.IGNORECASE)
            
            # Extract publication date if available
            pub_date = None
            
            # Try to find date in meta tags
            date_meta = soup.find('meta', {'property': 'article:published_time'}) or \
                      soup.find('meta', {'name': 'pubdate'}) or \
                      soup.find('meta', {'name': 'date'})
            
            if date_meta and 'content' in date_meta.attrs:
                pub_date = date_meta['content']
            
            # If no meta tag, look for common date elements
            if not pub_date:
                date_classes = ['date', 'time', 'publish-date', 'published-date', 'article-date', 
                               'post-date', 'timestamp', 'article-time']
                for class_name in date_classes:
                    date_elem = soup.find(class_=re.compile(class_name, re.I))
                    if date_elem:
                        pub_date = date_elem.get_text().strip()
                        break
            
            return article_content, pub_date
        else:
            logger.warning(f"Failed to fetch article: HTTP {response.status_code} - {url}")
            return None, None
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {e}")
        return None, None

def analyze_sentiment(text):
    """Perform sentiment analysis on the given text."""
    if not text:
        return {'polarity': 0, 'subjectivity': 0}
    
    try:
        analysis = TextBlob(text)
        return {
            'polarity': round(analysis.sentiment.polarity, 2),
            'subjectivity': round(analysis.sentiment.subjectivity, 2)
        }
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return {'polarity': 0, 'subjectivity': 0}

def process_financial_data(input_file, output_file, target_articles_per_company=8):
    """Process financial data, find news articles, analyze sentiment, and save to CSV."""
    # Read the input CSV
    df = pd.read_csv(input_file)
    
    # Initialize lists to store article data
    all_article_data = []
    
    # Process each company
    for index, row in df.iterrows():
        company_name = row['Company Name']
        symbol = row['Symbol']
        date = row['Date']
        sector = row.get('Sector', None)  # Get sector if available
        
        logger.info(f"\nProcessing {company_name} ({symbol}) for date {date}")
        
        # Search for news articles with exponential backoff for rate limits
        attempt = 0
        max_attempts = 3
        success = False
        
        while attempt < max_attempts and not success:
            try:
                articles = search_company_news(company_name, symbol, date, sector, target_articles_per_company)
                success = True
            except Exception as e:
                attempt += 1
                logger.error(f"Attempt {attempt}/{max_attempts} failed for {company_name}: {e}")
                # Exponential backoff
                wait_time = (2 ** attempt) * 30 + random.uniform(0, 30)
                logger.info(f"Waiting {wait_time:.1f} seconds before retry")
                time.sleep(wait_time)
                
                if attempt == max_attempts:
                    articles = []
                    logger.error(f"All attempts failed for {company_name}")
        
        if not articles:
            logger.warning(f"No news articles found for {company_name}")
            # Add entry with no articles
            all_article_data.append({
                **row.to_dict(),
                'Article_Title': 'No articles found',
                'Article_URL': '',
                'Article_Date': '',
                'Article_Snippet': '',
                'Article_Content': '',
                'Sentiment_Polarity': 0,
                'Sentiment_Subjectivity': 0,
                'Is_Relevant': False
            })
        else:
            # Process each article
            for i, article in enumerate(articles):
                logger.info(f"Processing article {i+1}/{len(articles)}: {article['title']}")
                
                # Extract full article text
                article_text, pub_date = extract_article_text(article['url'])
                
                # Use publication date if available, otherwise use apparent date from search
                article_date = pub_date or article.get('apparent_date', '')
                
                # Check if the article content is relevant to our target date
                is_relevant = verify_date_in_content(article_text, date, company_name, symbol)
                
                # Analyze sentiment if we have content
                sentiment = analyze_sentiment(article_text)
                
                # Add to our results
                all_article_data.append({
                    **row.to_dict(),
                    'Article_Title': article['title'],
                    'Article_URL': article['url'],
                    'Article_Date': article_date,
                    'Article_Snippet': article['snippet'],
                    'Article_Content': article_text[:5000] if article_text else '',  # First 5000 chars
                    'Sentiment_Polarity': sentiment['polarity'],
                    'Sentiment_Subjectivity': sentiment['subjectivity'],
                    'Is_Relevant': is_relevant
                })
                
                # Add a small delay between processing articles
                time.sleep(random.uniform(1, 3))
    
    # Create DataFrame from all the gathered data
    result_df = pd.DataFrame(all_article_data)
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to {output_file}")
    
    # Create a summary of relevant articles
    relevant_df = result_df[result_df['Is_Relevant'] == True]
    relevant_output = output_file.replace('.csv', '_relevant.csv')
    if not relevant_df.empty:
        relevant_df.to_csv(relevant_output, index=False)
        logger.info(f"Relevant articles saved to {relevant_output}")
    else:
        logger.warning("No relevant articles found")

if __name__ == "__main__":
    input_file = "/home/campuslens/WebScrapers/.10_significant_23-24.csv"  # Your input CSV file
    output_file = "/home/campuslens/WebScrapers/small_test.csv"  # Output CSV file
    
    # Set how many articles to find per company
    target_articles_per_company = 8
    
    logger.info(f"Starting analysis of financial data from {input_file}")
    process_financial_data(input_file, output_file, target_articles_per_company)
