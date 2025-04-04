# AI-Powered-SEC-Filings-Analyser


An AI-powered tool for extracting, analyzing, and visualizing insights from SEC filings using natural language processing and machine learning techniques.

## Features

- **Filing Retrieval**: Automatically download SEC filings for any publicly traded company
- **Section Extraction**: Parse and extract key sections from filings (Risk Factors, MD&A, etc.)
- **Advanced Analysis**:
  - Sentiment Analysis
  - Entity Recognition
  - Topic Modeling
  - Text Classification
  - Anomaly Detection
- **Interactive Visualizations**: Charts, graphs, and network diagrams
- **Agentic AI**: Ask questions about filings and get AI-powered answers
- **PDF Report Generation**: Export analysis results as PDF reports

## Setup

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/sfy45/AI-Powered-SEC-Filings-Analyser.git
cd AI-Powered-SEC-Filings-Analyser
```

2. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Download required NLTK and spaCy resources:
```
python -m nltk.downloader punkt stopwords wordnet
python -m spacy download en_core_web_sm
```

### Environment Variables

Create a `.env` file in the project root with the following variables:
```
OPENAI_API_KEY=your_openai_api_key  # Optional, for AI-powered insights
SEC_API_EMAIL=[your.email@example.com](mailto:your.email@example.com)  # Required for SEC EDGAR API
SEC_API_USER_AGENT="Your Name ([your.email@example.com](mailto:your.email@example.com))"  # Required for SEC EDGAR API
```

## Usage

1. Start the Streamlit application:
```
streamlit run app.py
```

2. Access the application in your web browser at `http://localhost:8501`

3. Enter a company ticker symbol, select filing type, year, and Quarter, and click "Process SEC Filing"

4. Navigate through the tabs to explore different analyses and visualizations

## Dependencies

Major dependencies include:

- **Data Processing**: pandas, numpy
- **NLP & ML**: nltk, spacy, scikit-learn, transformers, sentence-transformers
- **Visualization**: matplotlib, seaborn, plotly, pyvis
- **Web Interface**: streamlit
- **PDF Generation**: fpdf
- **SEC Filing Retrieval**: sec-edgar-downloader
- **Database**: sqlite3
- **AI Integration**: openai, langchain

See `requirements.txt` for a complete list of dependencies.

## Project Structure

```

AI-Powered-SEC-Filings-Analyzer/
├── app.py       # Main application file
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
├── data/                     # Downloaded SEC filings
├── mock_filings/             # Mock filing data for testing
└── sec-edgar-filings/        # SEC EDGAR downloaded filings

```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SEC EDGAR for providing access to filing data
- OpenAI for AI capabilities
- All open-source libraries used in this project
  
## Contributing

If you have a contribution to make, feel free to submit issues or pull requests. PRs are more than welcome!

## Contact

For any issues or suggestions, feel free to reach out: 📧 sophiasad1421@gmail.com
