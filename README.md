# Enhanced Crypto RAG (Retrieval-Augmented Generation)

A sophisticated cryptocurrency trading assistant that combines real-time market data, news analysis, and AI-powered insights using Retrieval-Augmented Generation (RAG) technology.

## Features

- Real-time cryptocurrency market data integration via CoinGecko API
- Live news aggregation from major crypto news sources (CoinTelegraph, CoinDesk)
- Advanced RAG system using Google's Generative AI
- Historical price analysis using yfinance
- Intelligent question-answering system about crypto markets and trends
- Vector store-based knowledge management using FAISS
- Comprehensive chat history tracking

## Prerequisites

- Python 3.12+
- Google API credentials properly configured
- Internet connection for real-time data fetching

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd Enhanced-Crypto-Rag
```

2. Create and activate a virtual environment:
```bash
python -m venv ECR
source ECR/bin/activate  # On Windows: ECR\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your API keys:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

## Dependencies

Key dependencies include:
- python-dotenv==1.0.1
- requests==2.31.0
- pandas==2.2.2
- feedparser==6.0.10
- beautifulsoup4==4.12.2
- yfinance==0.2.40
- numpy==1.26.4
- langchain-google-genai==1.0.8
- faiss-cpu==1.8.0
- langchain-community==0.2.16
- langchain==0.2.16

## Usage

```python
from ECR import CryptoTradingGuideRAG

# Initialize the RAG system
rag = CryptoTradingGuideRAG()

# Use the system for crypto analysis and insights
response = rag.query("What's the current market sentiment for Bitcoin?")
```

## Features in Detail

1. **Market Data Integration**
   - Real-time price data from CoinGecko
   - Historical price analysis
   - Market trends and indicators

2. **News Analysis**
   - Live news aggregation from major crypto sources
   - Sentiment analysis
   - Trend identification

3. **AI-Powered Insights**
   - RAG-based question answering
   - Context-aware responses
   - Historical data consideration

4. **Vector Store Management**
   - Efficient information retrieval
   - Dynamic knowledge base updates
   - FAISS-based similarity search

## Testing

The project includes a comprehensive test suite. To run the tests:

```bash
python -m unittest ECR.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Disclaimer

This tool is for informational purposes only and should not be considered financial advice. Always do your own research before making investment decisions.