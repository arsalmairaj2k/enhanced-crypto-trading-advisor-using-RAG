Crypto Trading Guide RAG System
A Retrieval-Augmented Generation (RAG) system designed to provide real-time cryptocurrency trading advice by combining live market data, news, expert insights, and trading strategies. The system uses LangChain with Google Generative AI models to deliver actionable recommendations, technical analysis, and portfolio insights.
Features

Real-Time Market Data: Fetches live price data and metrics for cryptocurrencies like Bitcoin, Ethereum, and Cardano via the CoinGecko API.
Crypto News Aggregation: Collects and summarizes recent news from RSS feeds (e.g., CoinTelegraph, CoinDesk).
Historical Data Analysis: Analyzes historical price data using yfinance for technical indicators like RSI, SMA, and volatility.
Expert Insights: Incorporates expert analysis from crypto thought leaders (mock data; extendable to Twitter/YouTube APIs).
Trading Strategies: Provides educational content on strategies like Dollar-Cost Averaging (DCA) and risk management.
Interactive Chat Interface: Allows users to ask trading-related questions and receive detailed responses with source citations.
Background Updates: Automatically refreshes data every 5 minutes for real-time insights.
Portfolio Analysis: Evaluates user portfolios for diversification and risk, offering rebalancing suggestions.
Robust Error Handling: Includes input validation and per-request error management for reliability.
Unit Testing: Includes basic unit tests to ensure core functionality (e.g., RSI calculation, portfolio validation).

Prerequisites

Python: Version 3.8 or higher.
Google API Key: Required for Google Generative AI models. Obtain from Google AI Studio.
Internet Connection: Needed for API calls to CoinGecko, yfinance, and RSS feeds.
Git: For cloning the repository (optional).

Installation

Clone the Repository:
git clone https://github.com/your-username/crypto-trading-guide-rag.git
cd crypto-trading-guide-rag


Install Dependencies:Create a virtual environment (optional but recommended) and install the required libraries:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Recommended requirements.txt with pinned versions for reproducibility:
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


Set Up Environment Variables:Create a .env file in the project root with your Google API key:
echo "GOOGLE_API_KEY=your-google-api-key" > .env

Alternatively, the system will prompt for the key during execution if not set.


Usage

Run the Script:
python lc.py


Interact with the Chat Interface:

After initialization, you’ll see the chat prompt:============================================================
CRYPTO TRADING ADVISOR - CHAT MODE
============================================================

Example Questions You Can Ask:
1. What's the current market sentiment for Bitcoin?
2. Should I buy Ethereum now or wait for a dip?
...

Type your question below (commands: 'quit' to exit, 'update' to refresh data, 'summary' for market overview)
------------------------------------------------------------
You:


Enter a question (e.g., "Should I buy Bitcoin now?") or a command:
summary: Get a comprehensive market overview.
update: Refresh data from APIs.
quit: Exit the program.


The system responds with detailed advice, including market analysis, risk considerations, and source citations.


Example Commands:

Ask about market sentiment: What's the current market sentiment for Bitcoin?
Request portfolio analysis: Analyze my portfolio: BTC=0.5, ETH=2.0
Learn about strategies: Explain dollar-cost averaging strategy for crypto


Run Unit Tests:To verify core functionality, run the tests:
python -m unittest lc.py



Project Structure

lc.py: Main script containing the CryptoTradingGuideRAG class and chat interface.
.env: Environment file for storing the Google API key (not tracked in git).
requirements.txt: List of dependencies with pinned versions (create based on the installation section).
README.md: This file, providing project documentation.

Notes

API Key Security: For production, consider using a secrets manager (e.g., AWS Secrets Manager, HashiCorp Vault) instead of .env.
Error Handling: The system handles API failures and invalid inputs gracefully, with detailed error messages.
Real-Time Updates: Data is updated every 5 minutes in the background. Use the update command for immediate refreshes.
Extensibility: The chat interface is modular and can be extended to a web-based UI (e.g., Streamlit) or WebSocket-based notifications.
Library Compatibility: Ensure requests and urllib3 are compatible with your system’s OpenSSL version to avoid warnings.

Troubleshooting

API Key Issues: If prompted for a key, ensure it’s valid and set in .env or entered correctly.
API Failures: Check your internet connection and API rate limits (CoinGecko, yfinance).
Library Errors: Verify all dependencies are installed with the correct versions.
Warnings: If you encounter urllib3 warnings, update the library or check OpenSSL compatibility.

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Powered by LangChain and Google Generative AI.
Data sources: CoinGecko, yfinance, CoinTelegraph, CoinDesk, CryptoNews, Decrypt.
Inspired by the need for reliable, data-driven crypto trading advice.
