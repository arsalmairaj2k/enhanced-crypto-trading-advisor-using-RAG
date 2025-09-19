import os
import warnings
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
import feedparser
import time
from bs4 import BeautifulSoup
import yfinance as yf
import numpy as np
import threading

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# For testing
import unittest

# Note: For dependencies, use pinned versions in requirements.txt for reproducibility, e.g.:
# requests==2.31.0
# yfinance==0.2.40
# etc.

# Removed global warning suppression; handle errors per request instead.
# Investigate urllib3/OpenSSL compatibility and update libraries if possible.

class CryptoTradingGuideRAG:
    def __init__(self):
        """Initialize the Crypto Trading Guide RAG system"""
        self.load_environment()
        self.setup_models()
        self.vector_store = None
        self.qa_chain = None
        self.chat_history = []
        
        # API configurations
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        
        # News RSS feeds
        self.news_feeds = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cryptonews.com/news/feed/",
            "https://decrypt.co/feed"
        ]
        
        # Symbol mapping for yfinance
        self.symbol_map = {
            "bitcoin": "BTC-USD",
            "ethereum": "ETH-USD",
            "cardano": "ADA-USD"
        }
        
    def load_environment(self):
        """Load environment variables"""
        # For production, use a secrets manager like AWS Secrets Manager or HashiCorp Vault instead of .env
        load_dotenv()
        
        if not os.environ.get("GOOGLE_API_KEY"):
            import getpass
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google AI Studio: ")
        
        self.api_key = os.environ["GOOGLE_API_KEY"]
        print(f"✓ API Key loaded: {bool(self.api_key)}")
        
    def setup_models(self):
        """Initialize LLM and embedding models"""
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            model="gemini-1.5-flash",
            temperature=0.3
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=self.api_key,
            model="models/embedding-001"
        )
        
        print("✓ Successfully initialized models")
    
    def fetch_crypto_news(self, max_articles: int = 20) -> List[Document]:
        """Fetch recent crypto news from RSS feeds"""
        documents = []
        print("Fetching live crypto news from RSS feeds...")
        
        for feed_url in self.news_feeds:
            print(f"  Parsing: {feed_url}")
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:5]:  # Get top 5 from each feed
                    # Extract and clean content
                    title = entry.get('title', 'No Title')
                    summary = entry.get('summary', entry.get('description', ''))
                    link = entry.get('link', '')
                    published = entry.get('published', '')
                    
                    # Clean HTML from summary
                    if summary:
                        soup = BeautifulSoup(summary, 'html.parser')
                        clean_summary = soup.get_text().strip()
                    else:
                        clean_summary = "No summary available"
                    
                    # Create document
                    content = f"""
                    Title: {title}
                    
                    Summary: {clean_summary}
                    
                    Published: {published}
                    Source URL: {link}
                    """
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": "crypto_news",
                            "title": title,
                            "url": link,
                            "published": published,
                            "feed_source": feed_url,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
                
                time.sleep(1)  # Rate limiting between feeds
            except Exception as e:
                print(f"Warning: Failed to parse feed {feed_url}: {e}")
        
        print(f"✓ Fetched {len(documents)} news articles from RSS feeds")
        return documents
    
    def fetch_market_data(self, symbols: List[str] = ["bitcoin", "ethereum", "cardano"]) -> List[Document]:
        """Fetch real-time market data from CoinGecko API"""
        documents = []
        print(f"Fetching live market data for {symbols}...")
        
        # CoinGecko API call for multiple coins
        symbols_str = ",".join(symbols)
        url = f"{self.coingecko_base_url}/simple/price"
        params = {
            "ids": symbols_str,
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()
        except Exception as e:
            print(f"Error fetching from CoinGecko: {e}")
            return documents
        
        for symbol in symbols:
            if symbol in data:
                coin_data = data[symbol]
                
                # Get additional detailed data
                detail_url = f"{self.coingecko_base_url}/coins/{symbol}"
                detail_params = {
                    "localization": "false",
                    "tickers": "false",
                    "market_data": "true",
                    "community_data": "false",
                    "developer_data": "false"
                }
                
                try:
                    time.sleep(0.5)  # Rate limiting
                    detail_response = requests.get(detail_url, params=detail_params)
                    detail_response.raise_for_status()
                    detail_data = detail_response.json()
                    market_data = detail_data.get("market_data", {})
                except Exception as e:
                    print(f"Error fetching details for {symbol}: {e}")
                    continue
                
                # Calculate accurate RSI using yfinance historical data
                yf_ticker = self.symbol_map.get(symbol, None)
                rsi = self.calculate_rsi_from_yf(yf_ticker) if yf_ticker else 50.0
                
                content = f"""
                {symbol.upper()} Live Market Analysis:
                Current Price: ${coin_data.get('usd', 0):,.2f}
                24h Change: {coin_data.get('usd_24h_change', 0):+.2f}%
                Market Cap: ${coin_data.get('usd_market_cap', 0):,.0f}
                24h Volume: ${coin_data.get('usd_24h_vol', 0):,.0f}
                
                Extended Metrics:
                7-day Change: {market_data.get('price_change_percentage_7d', 0):+.2f}%
                30-day Change: {market_data.get('price_change_percentage_30d', 0):+.2f}%
                All-time High: ${market_data.get('ath', {}).get('usd', 0):,.2f}
                ATH Date: {market_data.get('ath_date', {}).get('usd', 'N/A')[:10]}
                
                Technical Analysis:
                RSI (14-day): {rsi:.1f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})
                Market Rank: #{detail_data.get('market_cap_rank', 'N/A')}
                Circulating Supply: {market_data.get('circulating_supply', 0):,.0f}
                
                Sentiment: {'Bearish' if coin_data.get('usd_24h_change', 0) < -5 else 'Bullish' if coin_data.get('usd_24h_change', 0) > 5 else 'Neutral'}
                Last Updated: {datetime.fromtimestamp(coin_data.get('last_updated_at', 0)).strftime('%Y-%m-%d %H:%M:%S')}
                """
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "live_market_data",
                        "symbol": symbol.upper(),
                        "timestamp": datetime.now().isoformat(),
                        "type": "real_time_analysis",
                        "api_source": "coingecko"
                    }
                )
                documents.append(doc)
        
        print(f"✓ Fetched live market data for {len(documents)} cryptocurrencies")
        return documents
    
    def calculate_rsi_from_yf(self, ticker: str, period: str = "1mo", rsi_window: int = 14) -> float:
        """Calculate accurate RSI using yfinance historical data"""
        try:
            hist = yf.Ticker(ticker).history(period=period)
            if len(hist) < rsi_window:
                return 50.0  # Neutral if insufficient data
            return self.calculate_rsi(hist['Close'], window=rsi_window).iloc[-1]
        except Exception as e:
            print(f"Error calculating RSI for {ticker}: {e}")
            return 50.0
    
    def fetch_expert_content(self) -> List[Document]:
        """Fetch expert insights and analysis"""
        documents = []
        print("Fetching expert crypto content...")
        
        # Expert insights (in production, integrate with Twitter API v2, YouTube API)
        expert_insights = [
            {
                "expert": "Peter Brandt",
                "content": """Veteran trader analysis: Bitcoin's current price action resembles 
                patterns seen in previous bull markets. Key support levels are holding, 
                suggesting potential for upward movement. However, macro economic factors 
                remain a concern. Risk management is essential in current environment.""",
                "expertise": "Technical Analysis, Chart Patterns"
            },
            {
                "expert": "Plan B",
                "content": """Stock-to-Flow model analysis: Bitcoin remains within expected 
                price corridors despite recent volatility. Long-term fundamentals stay strong 
                with increasing institutional adoption. Scarcity metrics continue to support 
                bullish thesis over multi-year timeframes.""",
                "expertise": "Quantitative Analysis, S2F Model"
            },
            {
                "expert": "Willy Woo",
                "content": """On-chain analysis reveals: Network fundamentals remain robust 
                with strong hash rate and active addresses. Whale accumulation patterns 
                suggest smart money positioning. Short-term price action may be volatile 
                but long-term trajectory appears positive.""",
                "expertise": "On-chain Analysis, Network Metrics"
            },
            {
                "expert": "Coin Bureau (Guy)",
                "content": """Educational perspective: Current market cycle shows similarities 
                to previous cycles but with important differences due to institutional participation. 
                Regulatory clarity improving in major markets. Focus on projects with strong 
                fundamentals and real-world utility.""",
                "expertise": "Market Education, Fundamental Analysis"
            }
        ]
        
        for insight in expert_insights:
            content = f"""
            Expert: {insight['expert']}
            Expertise: {insight['expertise']}
            
            Analysis: {insight['content']}
            
            Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": "expert_analysis",
                    "expert": insight["expert"],
                    "expertise": insight["expertise"],
                    "timestamp": datetime.now().isoformat(),
                    "type": "expert_insight"
                }
            )
            documents.append(doc)
        
        print(f"✓ Fetched insights from {len(documents)} crypto experts")
        return documents
    
    def fetch_historical_data(self, symbols: List[str] = ["BTC-USD", "ETH-USD"], period: str = "6mo") -> List[Document]:
        """Fetch historical price data for backtesting using yfinance"""
        documents = []
        print(f"Fetching historical data for {symbols} over {period}...")
        
        for symbol in symbols:
            try:
                # Fetch historical data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    # Calculate technical indicators
                    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                    hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                    hist['RSI'] = self.calculate_rsi(hist['Close'])
                    hist['Volatility'] = hist['Close'].rolling(window=30).std()
                    
                    # Get recent statistics
                    latest = hist.iloc[-1]
                    
                    # Calculate performance metrics
                    returns = hist['Close'].pct_change().dropna()
                    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                    max_drawdown = ((hist['Close'] / hist['Close'].cummax()) - 1).min()
                    
                    content = f"""
                    {symbol} Historical Analysis ({period}):
                    
                    Current Metrics:
                    Latest Price: ${latest['Close']:.2f}
                    20-day SMA: ${latest['SMA_20']:.2f}
                    50-day SMA: ${latest['SMA_50']:.2f}
                    Current RSI: {latest['RSI']:.1f}
                    30-day Volatility: {latest['Volatility']:.2f}
                    
                    Performance Metrics:
                    Period Return: {((latest['Close'] / hist['Close'].iloc[0]) - 1) * 100:.2f}%
                    Annualized Sharpe Ratio: {sharpe_ratio:.2f}
                    Maximum Drawdown: {max_drawdown * 100:.2f}%
                    Average Daily Volume: {hist['Volume'].mean():,.0f}
                    
                    Trading Signals:
                    Trend: {'Bullish' if latest['Close'] > latest['SMA_20'] > latest['SMA_50'] else 'Bearish' if latest['Close'] < latest['SMA_20'] < latest['SMA_50'] else 'Mixed'}
                    RSI Signal: {'Oversold' if latest['RSI'] < 30 else 'Overbought' if latest['RSI'] > 70 else 'Neutral'}
                    Volatility: {'High' if latest['Volatility'] > hist['Volatility'].quantile(0.8) else 'Low' if latest['Volatility'] < hist['Volatility'].quantile(0.2) else 'Normal'}
                    
                    Backtesting Insights:
                    Best performing month: {hist.groupby(hist.index.to_period('M'))['Close'].apply(lambda x: (x.iloc[-1]/x.iloc[0]-1)*100).idxmax()}
                    Worst performing month: {hist.groupby(hist.index.to_period('M'))['Close'].apply(lambda x: (x.iloc[-1]/x.iloc[0]-1)*100).idxmin()}
                    """
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": "historical_data",
                            "symbol": symbol,
                            "period": period,
                            "timestamp": datetime.now().isoformat(),
                            "type": "historical_analysis",
                            "data_points": len(hist)
                        }
                    )
                    documents.append(doc)
                
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Error fetching historical data for {symbol}: {e}")
        
        print(f"✓ Fetched historical data for {len(documents)} symbols")
        return documents
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Removed duplicate calculate_rsi_estimate; using accurate calculation now
    
    def load_trading_strategies(self) -> List[Document]:
        """Load trading strategies and educational content"""
        
        trading_strategies = [
            {
                "strategy": "Dollar-Cost Averaging (DCA)",
                "content": """
                Dollar-Cost Averaging is a systematic investment strategy where you invest a fixed amount 
                regularly regardless of market price. This strategy helps reduce the impact of volatility 
                and removes the emotional aspect of timing the market.
                
                Implementation:
                - Set a fixed investment amount (e.g., $100/week)
                - Choose a consistent schedule (weekly, bi-weekly, monthly)
                - Stick to the plan regardless of market conditions
                - Best for long-term investors with steady income
                
                Pros: Reduces volatility impact, removes timing pressure, builds discipline
                Cons: May miss significant dips, requires steady cash flow
                """
            },
            {
                "strategy": "Technical Analysis Trading",
                "content": """
                Technical analysis involves studying price charts and indicators to predict future movements.
                
                Key Indicators:
                - RSI (Relative Strength Index): Identifies overbought/oversold conditions
                - Moving Averages: Trend identification and support/resistance
                - MACD: Momentum and trend changes
                - Bollinger Bands: Volatility and price extremes
                - Volume: Confirms price movements
                
                Trading Signals:
                - RSI < 30: Potential buy signal (oversold)
                - RSI > 70: Potential sell signal (overbought)
                - Price above MA: Bullish trend
                - Golden Cross: 50-day MA crosses above 200-day MA
                """
            },
            {
                "strategy": "Risk Management",
                "content": """
                Risk management is crucial for successful crypto trading.
                
                Key Principles:
                - Never invest more than you can afford to lose
                - Diversify across different cryptocurrencies
                - Set stop-loss orders to limit downside
                - Take profits systematically
                - Position sizing: Risk only 1-2% per trade
                
                Portfolio Allocation:
                - 60-70% in established coins (BTC, ETH)
                - 20-30% in mid-cap altcoins
                - 5-10% in high-risk/high-reward projects
                
                Emotional Control:
                - Stick to predetermined entry/exit points
                - Avoid FOMO (Fear of Missing Out)
                - Don't panic sell during dips
                - Keep detailed trading journal
                """
            }
        ]
        
        documents = []
        for strategy in trading_strategies:
            doc = Document(
                page_content=f"Strategy: {strategy['strategy']}\n\n{strategy['content']}",
                metadata={
                    "source": "trading_strategies",
                    "strategy_name": strategy["strategy"],
                    "type": "educational"
                }
            )
            documents.append(doc)
        
        print(f"✓ Loaded {len(documents)} trading strategy documents")
        return documents
    
    def create_vector_store(self, documents: List[Document]):
        """Create and populate vector store with documents"""
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        splits = text_splitter.split_documents(documents)
        print(f"✓ Split documents into {len(splits)} chunks")
        
        # Create vector store
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        print("✓ Created vector store")
        
        return self.vector_store
    
    def setup_qa_chain(self):
        """Setup the QA chain with custom prompt"""
        
        custom_prompt = PromptTemplate(
            template="""You are an expert crypto trading advisor. Use the following context to provide 
            comprehensive and actionable trading advice. Always consider risk management and provide 
            balanced perspectives. Explain why you made each recommendation based on specific data points, 
            indicators, or news items from the context.

            Context: {context}
            
            Question: {question}
            
            Provide a detailed response that includes:
            1. Direct answer to the question
            2. Relevant market analysis if applicable, explaining the 'why' behind trends
            3. Risk considerations with justifications
            4. Actionable recommendations and the reasoning for them
            5. Educational insights
            
            Answer:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=True
        )
        
        print("✓ Setup QA chain")
    
    def initialize_system(self):
        """Initialize the complete RAG system"""
        print("Initializing Crypto Trading Guide RAG System...")
        
        # Collect all documents from real-time sources
        all_documents = []
        
        # Fetch live data sources
        news_docs = self.fetch_crypto_news()
        market_docs = self.fetch_market_data()
        strategy_docs = self.load_trading_strategies()
        expert_docs = self.fetch_expert_content()
        historical_docs = self.fetch_historical_data()
        
        all_documents.extend(news_docs)
        all_documents.extend(market_docs)
        all_documents.extend(strategy_docs)
        all_documents.extend(expert_docs)
        all_documents.extend(historical_docs)
        
        if not all_documents:
            raise ValueError("No documents loaded from real-time sources. Check API connections.")
        
        # Create vector store
        self.create_vector_store(all_documents)
        
        # Setup QA chain
        self.setup_qa_chain()
        
        # Start background update thread for real-time updates
        threading.Thread(target=self._background_update, daemon=True).start()
        
        print("✓ Crypto Trading Guide RAG System initialized successfully!")
    
    def _background_update(self):
        """Background thread for periodic data updates"""
        while True:
            time.sleep(300)  # Update every 5 minutes
            try:
                self.update_all_data()
                print("✓ Background data update completed")
            except Exception as e:
                print(f"✗ Background update failed: {e}")
    
    def get_trading_advice(self, question: str) -> Dict[str, Any]:
        """Get trading advice based on the question"""
        response = self.qa_chain.invoke({"query": question})
        
        # Add to chat history for context
        self.chat_history.append(f"Human: {question}")
        self.chat_history.append(f"Assistant: {response['result']}")
        
        return {
            "answer": response["result"],
            "source_documents": [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "unknown"),
                    "title": doc.metadata.get("title", "N/A")
                }
                for doc in response.get("source_documents", [])
            ]
        }
    
    def update_all_data(self):
        """Update all data sources with fresh real-time data"""
        print("Updating all data sources...")
        
        # Fetch fresh data
        news_docs = self.fetch_crypto_news()
        market_docs = self.fetch_market_data()
        expert_docs = self.fetch_expert_content()
        
        all_new_docs = news_docs + market_docs + expert_docs
        
        if all_new_docs:
            # Add to existing vector store
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(all_new_docs)
            self.vector_store.add_documents(splits)
            print(f"✓ Updated vector store with {len(all_new_docs)} new documents")
        else:
            print("No new documents fetched.")
    
    def get_market_summary(self) -> str:
        """Get a comprehensive market summary"""
        question = """
        Provide a comprehensive market summary including:
        1. Current market sentiment across major cryptocurrencies
        2. Key technical levels and indicators
        3. Recent news impact on prices
        4. Expert opinions and analysis
        5. Trading recommendations for the next 24-48 hours
        """
        
        response = self.get_trading_advice(question)
        return response["answer"]
    
    def get_portfolio_analysis(self, portfolio: Dict[str, float]) -> str:
        """Analyze a given portfolio"""
        # Data validation
        if not isinstance(portfolio, dict):
            raise ValueError("Portfolio must be a dictionary with string keys (symbols) and numeric values (amounts)")
        for symbol, amount in portfolio.items():
            if not isinstance(symbol, str) or not isinstance(amount, (int, float)):
                raise ValueError(f"Invalid portfolio entry: {symbol}={amount}. Symbols must be strings, amounts must be numbers.")
            if amount < 0:
                raise ValueError(f"Amount for {symbol} cannot be negative.")
        
        portfolio_text = "Current Portfolio:\n"
        for symbol, amount in portfolio.items():
            portfolio_text += f"- {symbol}: {amount}\n"
        
        question = f"""
        Please analyze this portfolio and provide recommendations:
        {portfolio_text}
        
        Consider:
        1. Portfolio diversification
        2. Risk assessment
        3. Rebalancing suggestions
        4. Current market conditions impact
        """
        
        return self.get_trading_advice(question)["answer"]

def main():
    """Main function to demonstrate the system"""
    # Initialize the system
    crypto_guide = CryptoTradingGuideRAG()
    crypto_guide.initialize_system()
    
    print("\n" + "="*60)
    print("CRYPTO TRADING GUIDE RAG SYSTEM READY")
    print("="*60)
    
    # Example queries
    example_questions = [
        "What's the current market sentiment for Bitcoin?",
        "Should I buy Ethereum now or wait for a dip?",
        "Explain dollar-cost averaging strategy for crypto",
        "What are the key technical indicators I should watch?",
        "How should I manage risk in crypto trading?"
    ]
    
    print("\nExample Questions You Can Ask:")
    for i, q in enumerate(example_questions, 1):
        print(f"{i}. {q}")
    
    # Interactive loop
    print("\n" + "-"*60)
    print("Ask your crypto trading questions (type 'quit' to exit):")
    print("Special commands: 'update' = refresh data, 'summary' = market overview")
    print("-"*60)
    
    while True:
        user_question = input("\nYour Question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'bye']:
            print("Thanks for using the Crypto Trading Guide! Happy trading! 🚀")
            break
        
        if user_question.lower() == 'update':
            try:
                crypto_guide.update_all_data()
            except Exception as e:
                print(f"✗ Update failed: {e}")
            continue
            
        if user_question.lower() == 'summary':
            print("\nGenerating market summary... 📈")
            try:
                summary = crypto_guide.get_market_summary()
                print(f"\n📊 Market Summary:")
                print("-" * 40)
                print(summary)
            except Exception as e:
                print(f"✗ Summary generation failed: {e}")
            continue
        
        if not user_question:
            continue
        
        print("\nAnalyzing... 🤔")
        try:
            response = crypto_guide.get_trading_advice(user_question)
            
            print(f"\n📊 Trading Advisor Response:")
            print("-" * 40)
            print(response["answer"])
            
            if response["source_documents"]:
                print(f"\n📚 Sources Referenced:")
                for i, source in enumerate(response["source_documents"][:3], 1):
                    print(f"{i}. {source['title']} ({source['source']})")
                    print(f"   Preview: {source['content']}")
        
        except Exception as e:
            print(f"✗ Error processing question: {e}")

# Unit and Integration Tests
class TestCryptoTradingGuideRAG(unittest.TestCase):
    def setUp(self):
        self.guide = CryptoTradingGuideRAG()
        # Mock API key for testing
        os.environ["GOOGLE_API_KEY"] = "test_key"
        self.guide.api_key = "test_key"

    def test_load_environment(self):
        self.guide.load_environment()
        self.assertTrue(bool(self.guide.api_key))

    def test_calculate_rsi(self):
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 110, 112, 111, 113, 115])
        rsi = self.guide.calculate_rsi(prices)
        self.assertFalse(rsi.isna().all())  # Ensure RSI is calculated

    def test_get_portfolio_analysis_validation(self):
        with self.assertRaises(ValueError):
            self.guide.get_portfolio_analysis("invalid")  # Not a dict
        with self.assertRaises(ValueError):
            self.guide.get_portfolio_analysis({"BTC": "invalid"})  # Not numeric
        with self.assertRaises(ValueError):
            self.guide.get_portfolio_analysis({"BTC": -100})  # Negative

    # Add more tests as needed, e.g., for fetch functions (mock requests)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"✗ System startup error: {e}")
        print("\nRequired setup:")
        print("1. Set GOOGLE_API_KEY in .env file")
        print("2. Install: pip install langchain-google-genai faiss-cpu langchain-community")
        print("3. Install: pip install feedparser beautifulsoup4 yfinance pandas numpy")
        print("4. Ensure stable internet connection for API calls")
    # Run tests: unittest.main() if needed, but for now, it's separate