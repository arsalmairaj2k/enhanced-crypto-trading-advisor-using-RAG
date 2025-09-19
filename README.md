Crypto Trading Guide RAG System
Overview
The Crypto Trading Guide RAG System is a Retrieval-Augmented Generation (RAG) application designed to provide real-time cryptocurrency trading advice. It integrates live market data, news, expert insights, historical data, and trading strategies to deliver comprehensive, actionable recommendations. The system uses Google Generative AI models for natural language processing and FAISS for efficient document retrieval, making it a powerful tool for crypto traders.
Key features include:

Real-time market data from CoinGecko
News aggregation from multiple RSS feeds
Historical price analysis using yfinance
Expert insights and trading strategies
Interactive chat interface for user queries
Background data updates for real-time information
Robust error handling and input validation

Installation
Prerequisites

Python 3.8 or higher
Stable internet connection for API calls
Google API key for Google Generative AI models

Dependencies
Install the required Python libraries using pip:
pip install python-dotenv requests pandas feedparser beautifulsoup4 yfinance numpy langchain-google-genai faiss-cpu langchain-community langchain

For reproducibility, consider using a requirements.txt file with pinned versions:
python-dotenv==1.0.1
requests==2.31.0
pandas==2.2.2
feedparser==6.0.10
beautifulsoup4==4.12.2
yfinance==0.2.40
numpy==1.26.4
langchain-google-genai==1.0.8
faiss-cpu==1.8.0
langchain-community==0.2.16
langchain==0.2.16

Install with:
pip install -r requirements.txt

Environment Setup

Create a .env file in the project root directory.
Add your Google API key:

GOOGLE_API_KEY=your_google_api_key_here

You can obtain a Google API key from Google AI Studio. If the key is not provided, the system will prompt for it during execution.
Usage

Clone or download the project files.
Ensure all dependencies are installed and the .env file is configured.
Run the main script:

python lc.py


The system will initialize by fetching data from various sources (CoinGecko, RSS feeds, yfinance) and setting up the RAG pipeline.
Once initialized, you’ll enter the chat mode with the following prompt:

============================================================
CRYPTO TRADING ADVISOR - CHAT MODE
============================================================

Example Questions You Can Ask:
1. What's the current market sentiment for Bitcoin?
2. Should I buy Ethereum now or wait for a dip?
...

Type your question below (commands: 'quit' to exit, 'update' to refresh data, 'summary' for market overview)
------------------------------------------------------------
You:


Interact with the system by:

Asking crypto-related questions (e.g., "Should I buy Bitcoin now?").
Using special commands:
summary: Get a comprehensive market overview.
update: Refresh data from live sources.
quit, exit, or bye: Exit the application.




The system responds with detailed advice, including market analysis, risk considerations, recommendations, and referenced sources.


Features

Real-Time Data:

Fetches live market data for cryptocurrencies like Bitcoin, Ethereum, and Cardano from CoinGecko.
Aggregates news from trusted sources (CoinTelegraph, CoinDesk, CryptoNews, Decrypt).
Updates data every 5 minutes in the background for real-time insights.


Historical Analysis:

Retrieves historical price data using yfinance.
Calculates technical indicators like RSI, SMA, volatility, and Sharpe ratio.


Expert Insights:

Includes curated expert analyses from industry figures (e.g., Peter Brandt, Plan B).
Provides insights on technical analysis, on-chain metrics, and market fundamentals.


Trading Strategies:

Offers educational content on strategies like Dollar-Cost Averaging, Technical Analysis, and Risk Management.


Interactive Chat Interface:

Modular chat() method for conversational interaction.
Supports natural language queries with detailed, source-referenced responses.
Maintains chat history for context.


Robust Error Handling:

Validates inputs (e.g., portfolio dictionary, question strings).
Handles API errors and network issues gracefully.


Testing:

Includes unit tests for key functionalities (e.g., RSI calculation, portfolio validation).
Run tests with: python -m unittest lc.py.


Explainability:

Responses include explanations of why recommendations are made, based on specific data points, indicators, or news.



Project Structure

lc.py: Main script containing the CryptoTradingGuideRAG class, chat interface, and tests.
.env: Environment file for storing the Google API key (not included in version control).
requirements.txt (optional): Recommended for pinning dependency versions.

Example Questions

"What's the current market sentiment for Bitcoin?"
"Should I buy Ethereum now or wait for a dip?"
"Explain dollar-cost averaging strategy for crypto."
"What are the key technical indicators I should watch?"
"How should I manage risk in crypto trading?"

Notes

API Key Security: For production use, consider a secrets manager (e.g., AWS Secrets Manager, HashiCorp Vault) instead of a .env file.
Library Compatibility: Ensure urllib3 and OpenSSL versions are compatible to avoid warnings. Update libraries if needed.
Rate Limiting: The system includes delays to respect API rate limits (e.g., CoinGecko, yfinance).
Extensibility: The chat() method is modular and can be extended for GUI integration (e.g., Streamlit) or WebSocket-based real-time notifications.

Troubleshooting

API Errors: Ensure a stable internet connection and valid Google API key.
Missing Dependencies: Verify all libraries are installed with correct versions.
Data Fetch Issues: Check RSS feed URLs and CoinGecko API status.
Startup Errors: Review the error message and ensure setup steps are followed (see error message in the console).

Future Improvements

Add a web-based UI using Streamlit or Flask for a graphical chat interface.
Implement WebSocket support for real-time data notifications.
Expand expert insights with live data from X API or YouTube API.
Add more technical indicators (e.g., MACD, Bollinger Bands) for deeper analysis.

License
This project is for educational purposes and provided as-is. Always conduct your own research before making trading decisions. Cryptocurrency investments carry high risk.

Happy trading! 🚀 For questions or contributions, contact the project maintainer.
