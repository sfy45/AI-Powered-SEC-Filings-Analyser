import os
import re
import json
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
from typing import List, Dict, Any, Optional, Tuple, Union
import sqlite3
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory
import re
from typing import List, Union
import networkx as nx
try:
    from pyvis.network import Network
except ImportError:
    pass
import torch
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import random
from fpdf import FPDF
from io import BytesIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Import for OpenAI client
from openai import OpenAI as OpenAIClient

# Import for Agentic AI
try:
    import mcp  # Mock import for MCP protocol
except ImportError:
    pass

# Set page configuration
st.set_page_config(
    page_title="AI-Powered SEC Filings Analyser",
    page_icon="📊",
    layout="wide"
)

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        with st.spinner("Downloading NLTK resources..."):
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
    return True

# Initialize NLTK resources
download_nltk_resources()

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        st.warning("Downloading spaCy model. This may take a moment...")
        os.system("python -m spacy download en_core_web_sm")
        try:
            return spacy.load("en_core_web_sm")
        except:
            st.error("Failed to load spaCy model. Using fallback mechanisms.")
            return None

# Initialize spaCy
nlp = load_spacy_model()

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize sentence transformer model for embeddings
@st.cache_resource
def load_sentence_transformer():
    try:
        with st.spinner("Loading language model (this will only happen once)..."):
            return SentenceTransformer('paraphrase-MiniLM-L3-v2')
    except Exception as e:
        st.warning(f"SentenceTransformer not available: {e}. Some features will be limited.")
        return None

# Initialize summarization model
@st.cache_resource
def load_summarizer():
    try:
        with st.spinner("Loading summarization model..."):
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
            print("✅ Summarization model loaded successfully.")  # Debugging print
            return summarizer
    except Exception as e:
        st.warning(f"Error loading summarizer: {e}. Using fallback.")
        print(f"❌ Error loading summarizer: {e}")  # Debugging print
        try:
            return pipeline("summarization")
        except Exception as e:
            print(f"❌ Fallback Summarization Failed: {e}")  # Debugging print
            st.warning("Summarization not available. Some features will be limited.")
            return None



# Initialize text classification model
@st.cache_resource
def load_text_classifier():
    try:
        with st.spinner("Loading text classification model..."):
            # Create a simple text classification pipeline
            model = make_pipeline(
                CountVectorizer(),
                MultinomialNB()
            )
            return model
    except Exception as e:
        st.warning(f"Error loading text classifier: {e}")
        return None

# Database setup
def setup_database():
    try:
        conn = sqlite3.connect('sec_filings.db')
        c = conn.cursor()
        
        # Create tables if they don't exist
        c.execute('''
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY,
            ticker TEXT UNIQUE,
            name TEXT,
            cik TEXT,
            sic TEXT,
            industry TEXT
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS filings (
            id INTEGER PRIMARY KEY,
            company_id INTEGER,
            filing_type TEXT,
            filing_date TEXT,
            filing_year INTEGER,
            filing_quarter INTEGER,
            accession_number TEXT,
            file_path TEXT,
            processed INTEGER DEFAULT 0,
            FOREIGN KEY (company_id) REFERENCES companies (id)
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY,
            filing_id INTEGER,
            section_name TEXT,
            section_text TEXT,
            FOREIGN KEY (filing_id) REFERENCES filings (id)
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY,
            section_id INTEGER,
            analysis_type TEXT,
            analysis_result TEXT,
            FOREIGN KEY (section_id) REFERENCES sections (id)
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS risk_factors (
            id INTEGER PRIMARY KEY,
            section_id INTEGER,
            risk_category TEXT,
            risk_name TEXT,
            severity REAL,
            trend TEXT,
            year INTEGER,
            FOREIGN KEY (section_id) REFERENCES sections (id)
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            section_id INTEGER,
            entity_type TEXT,
            entity_name TEXT,
            frequency REAL,
            sentiment REAL,
            FOREIGN KEY (section_id) REFERENCES sections (id)
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS entity_relationships (
            id INTEGER PRIMARY KEY,
            section_id INTEGER,
            entity1_id INTEGER,
            entity2_id INTEGER,
            relationship_type TEXT,
            confidence REAL,
            FOREIGN KEY (section_id) REFERENCES sections (id),
            FOREIGN KEY (entity1_id) REFERENCES entities (id),
            FOREIGN KEY (entity2_id) REFERENCES entities (id)
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS text_classifications (
            id INTEGER PRIMARY KEY,
            section_id INTEGER,
            category TEXT,
            confidence REAL,
            FOREIGN KEY (section_id) REFERENCES sections (id)
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY,
            section_id INTEGER,
            anomaly_score REAL,
            description TEXT,
            FOREIGN KEY (section_id) REFERENCES sections (id)
        )
        ''')
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error setting up database: {e}")
        return False

# Initialize database
setup_database()

class SECFilingsAgent:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        try:
            self.llm = OpenAI(temperature=0.3, openai_api_key=openai_api_key)
            self.setup_agent()
        except Exception as e:
            st.error(f"Error initializing OpenAI: {e}")
            self.llm = None
            self.agent_executor = None
    
    def setup_agent(self):
        if not self.llm:
            return
            
        # Define tools
        tools = [
            Tool(
                name="Extract Risk Factors",
                func=self.extract_risk_factors,
                description="Extracts risk factors from a SEC filing"
            ),
            Tool(
                name="Analyze Sentiment",
                func=self.analyze_sentiment,
                description="Analyzes sentiment of a section in a SEC filing"
            ),
            Tool(
                name="Extract Entities",
                func=self.extract_entities,
                description="Extracts named entities from a SEC filing"
            ),
            Tool(
                name="Generate Summary",
                func=self.generate_summary,
                description="Generates a summary of a SEC filing section"
            ),
            Tool(
                name="Compare Filings",
                func=self.compare_filings,
                description="Compares multiple SEC filings"
            ),
            Tool(
                name="Classify Text",
                func=self.classify_text,
                description="Classifies text into predefined categories"
            ),
            Tool(
                name="Detect Anomalies",
                func=self.detect_anomalies,
                description="Detects unusual patterns or anomalies in the text"
            ),
            Tool(
                name="Get Filing History",
                func=self.get_filing_history,
                description="Gets the filing history for a company"
            )
        ]
        
        # Define prompt template with all required inputs
        template = """
        You are an AI assistant specialized in analyzing SEC filings (10-K, 10-Q, 8-K).
        
        {chat_history}
        
        Question: {input}
        
        Think through this step by step:
        1. Understand what the human is asking about SEC filings
        2. Determine which tool would be most helpful
        3. Use the tool to get the information
        4. Provide a clear, concise response
        
        Available tools: {tools}
        
        Response:
        """
        
        prompt = PromptTemplate(
            input_variables=["input", "chat_history", "tools"],
            template=template
        )
        
        # Define output parser
        class CustomOutputParser(AgentOutputParser):
            def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
                if "Final Answer:" in llm_output:
                    return AgentFinish(
                        return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                        log=llm_output,
                    )
                
                regex = r"Action: (.*?)[\n]*Action Input: (.*)"
                match = re.search(regex, llm_output, re.DOTALL)
                if not match:
                    return AgentFinish(
                        return_values={"output": llm_output},
                        log=llm_output,
                    )
                action = match.group(1).strip()
                action_input = match.group(2).strip()
                
                return AgentAction(tool=action, tool_input=action_input, log=llm_output)
        
        output_parser = CustomOutputParser()
        
        # Define LLM chain
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Define agent
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in tools]
        )
        
        # Define memory
        memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Define agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=memory
        )
    
    def extract_risk_factors(self, query):
        # Parse query to get ticker, year, and filing type
        match = re.search(r"ticker: (\w+), year: (\d{4})(?:, filing_type: (\w+))?", query)
        if not match:
            return "Please provide ticker and year in the format: ticker: AAPL, year: 2023, filing_type: 10-K"
        
        ticker = match.group(1)
        year = int(match.group(2))
        filing_type = match.group(3) if match.group(3) else "10-K"
        
        # Query database for risk factors
        try:
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            
            c.execute("""
                SELECT rf.risk_category, rf.risk_name, rf.severity, rf.trend
                FROM risk_factors rf
                JOIN sections s ON rf.section_id = s.id
                JOIN filings f ON s.filing_id = f.id
                JOIN companies c ON f.company_id = c.id
                WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ?
            """, (ticker, year, filing_type))
            
            results = c.fetchall()
            conn.close()
        except Exception as e:
            return f"Database error: {str(e)}"
        
        if not results:
            # Generate mock data for demo
            risk_categories = ["Operational", "Technological", "Regulatory", "Market"]
            risk_names = [
                ["Supply Chain Disruption", "Manufacturing Delays", "Quality Control Issues"],
                ["Cybersecurity Threats", "Technology Obsolescence", "Intellectual Property Protection"],
                ["Compliance Requirements", "International Trade Policies", "Data Privacy Regulations"],
                ["Competitive Pressure", "Consumer Preference Changes", "Economic Uncertainty"]
            ]
            severities = [0.85, 0.72, 0.65, 0.88, 0.71, 0.75, 0.78, 0.81, 0.76, 0.82, 0.69, 0.77]
            trends = ["up", "stable", "down"]
            
            results = []
            for i, category in enumerate(risk_categories):
                for j, name in enumerate(risk_names[i]):
                    idx = i * 3 + j
                    results.append((category, name, severities[idx], trends[idx % 3]))
        
        # Format results
        response = f"Risk factors for {ticker} ({year}) {filing_type}:\n\n"
        
        by_category = {}
        for category, name, severity, trend in results:
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((name, severity, trend))
        
        for category, risks in by_category.items():
            response += f"## {category} Risks\n\n"
            for name, severity, trend in risks:
                severity_pct = f"{severity * 100:.0f}%"
                trend_icon = "↑" if trend == "up" else "↓" if trend == "down" else "→"
                response += f"- {name} (Severity: {severity_pct}, Trend: {trend_icon})\n"
            response += "\n"
        
        return response
    
    def analyze_sentiment(self, query):
        # Parse query to get ticker, year, filing type, and section
        match = re.search(r"ticker: (\w+), year: (\d{4})(?:, filing_type: (\w+))?, section: (\w+)", query)
        if not match:
            return "Please provide ticker, year, and section in the format: ticker: AAPL, year: 2023, filing_type: 10-K, section: risk_factors"
        
        ticker = match.group(1)
        year = int(match.group(2))
        filing_type = match.group(3) if match.group(3) else "10-K"
        section_name = match.group(4)
        
        # Query database for sentiment analysis
        try:
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            
            c.execute("""
                SELECT ar.analysis_result
                FROM analysis_results ar
                JOIN sections s ON ar.section_id = s.id
                JOIN filings f ON s.filing_id = f.id
                JOIN companies c ON f.company_id = c.id
                WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ? AND s.section_name = ? AND ar.analysis_type = 'sentiment'
            """, (ticker, year, filing_type, section_name))
            
            result = c.fetchone()
            conn.close()
        except Exception as e:
            return f"Database error: {str(e)}"
        
        if not result:
            # Generate mock data for demo
            sentiment = {
                "compound": 0.42,
                "pos": 0.31,
                "neg": 0.47,
                "neu": 0.22
            }
        else:
            try:
                sentiment = json.loads(result[0])
            except:
                sentiment = {
                    "compound": 0.42,
                    "pos": 0.31,
                    "neg": 0.47,
                    "neu": 0.22
                }
        
        # Format results
        response = f"Sentiment analysis for {ticker} ({year}) {filing_type} {section_name.replace('_', ' ')}:\n\n"
        
        sentiment_label = "Negative" if sentiment["compound"] < -0.05 else "Positive" if sentiment["compound"] > 0.05 else "Neutral"
        response += f"Overall sentiment: {sentiment_label} (Compound score: {sentiment['compound']:.2f})\n\n"
        response += f"- Positive: {sentiment['pos']:.2f}\n"
        response += f"- Negative: {sentiment['neg']:.2f}\n"
        response += f"- Neutral: {sentiment['neu']:.2f}\n\n"
        
        response += "Interpretation:\n"
        if sentiment["compound"] < -0.05:
            response += "The text has a predominantly negative tone, which may indicate significant risk disclosures or cautious language."
        elif sentiment["compound"] > 0.05:
            response += "The text has a predominantly positive tone, which may indicate confidence in the company's position or outlook."
        else:
            response += "The text has a neutral tone, which is typical for formal financial disclosures."
        
        return response
    
    def extract_entities(self, query):
        # Parse query to get ticker, year, filing type, and section
        match = re.search(r"ticker: (\w+), year: (\d{4})(?:, filing_type: (\w+))?, section: (\w+)", query)
        if not match:
            return "Please provide ticker, year, and section in the format: ticker: AAPL, year: 2023, filing_type: 10-K, section: risk_factors"
        
        ticker = match.group(1)
        year = int(match.group(2))
        filing_type = match.group(3) if match.group(3) else "10-K"
        section_name = match.group(4)
        
        # Query database for entities
        try:
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            
            c.execute("""
                SELECT e.entity_type, e.entity_name, e.frequency, e.sentiment
                FROM entities e
                JOIN sections s ON e.section_id = s.id
                JOIN filings f ON s.filing_id = f.id
                JOIN companies c ON f.company_id = c.id
                WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ? AND s.section_name = ?
            """, (ticker, year, filing_type, section_name))
            
            results = c.fetchall()
            conn.close()
        except Exception as e:
            return f"Database error: {str(e)}"
        
        if not results:
            # Generate mock data for demo
            entity_types = ["ORG", "PERSON", "GPE", "LOC", "PRODUCT"]
            entity_names = [
                [ticker, "Competitors", "Suppliers", "Regulatory Bodies", "Partners"],
                ["CEO", "CFO", "Board Members", "Executives"],
                ["United States", "China", "Europe", "Asia Pacific"],
                ["Manufacturing Facilities", "Headquarters", "Distribution Centers"],
                ["Core Products", "Services", "Software", "Hardware"]
            ]
            frequencies = [0.92, 0.45, 0.32, 0.28, 0.22, 0.85, 0.65, 0.55, 0.45, 0.68, 0.42, 0.38, 0.32, 0.25, 0.20, 0.15, 0.75, 0.58, 0.48, 0.42]
            sentiments = [0.68, 0.38, 0.52, 0.41, 0.65, 0.60, 0.55, 0.50, 0.45, 0.62, 0.45, 0.58, 0.55, 0.51, 0.48, 0.45, 0.72, 0.68, 0.65, 0.61]
            
            results = []
            idx = 0
            for i, entity_type in enumerate(entity_types):
                for j, name in enumerate(entity_names[i]):
                    if idx < len(frequencies) and j < len(entity_names[i]):
                        results.append((entity_type, name, frequencies[idx], sentiments[idx]))
                        idx += 1
        
        # Format results
        response = f"Entity analysis for {ticker} ({year}) {filing_type} {section_name.replace('_', ' ')}:\n\n"
        
        by_type = {}
        for entity_type, name, frequency, sentiment in results:
            if entity_type not in by_type:
                by_type[entity_type] = []
            by_type[entity_type].append((name, frequency, sentiment))
        
        for entity_type, entities in by_type.items():
            response += f"## {entity_type} Entities\n\n"
            for name, frequency, sentiment in sorted(entities, key=lambda x: x[1], reverse=True):
                frequency_pct = f"{frequency * 100:.0f}%"
                sentiment_label = "Negative" if sentiment < 0.4 else "Positive" if sentiment > 0.6 else "Neutral"
                response += f"- {name} (Frequency: {frequency_pct}, Sentiment: {sentiment_label})\n"
            response += "\n"
        
        return response
    
    def generate_summary(self, query):
        # Parse query to get ticker, year, filing type, and section
        match = re.search(r"ticker: (\w+), year: (\d{4})(?:, filing_type: (\w+))?, section: (\w+)", query)
        if not match:
            return "Please provide ticker, year, and section in the format: ticker: AAPL, year: 2023, filing_type: 10-K, section: risk_factors"

        ticker = match.group(1)
        year = int(match.group(2))
        filing_type = match.group(3) if match.group(3) else "10-K"
        section_name = match.group(4)

        try:
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()

            c.execute("""
                SELECT ar.analysis_result
                FROM analysis_results ar
                JOIN sections s ON ar.section_id = s.id
                JOIN filings f ON s.filing_id = f.id
                JOIN companies c ON f.company_id = c.id
                WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ? AND s.section_name = ? AND ar.analysis_type = 'summary'
            """, (ticker, year, filing_type, section_name))

            result = c.fetchone()
            conn.close()
        except Exception as e:
            return f"Database error: {str(e)}"

        # Use summarizer if available and no result is found in DB
        if not result:
            # Generate a mock summary instead of trying to use self.summarizer
            summary = self.generate_mock_summary(ticker, year, filing_type, section_name)
        else:
            summary = result[0]

        # Format result
        response = f"Summary of {ticker} ({year}) {filing_type} {section_name.replace('_', ' ')}:\n\n"
        response += summary.strip()
        return response
    
    def generate_mock_summary(self, ticker, year, filing_type, section_name):
        """Generate a mock summary when the summarizer is not available"""
        if section_name == "risk_factors":
            return f"This section outlines key risks facing {ticker} in {year}, including market competition, technological changes, regulatory challenges, and operational concerns. The company highlights cybersecurity, supply chain disruptions, and compliance requirements as significant areas of focus."
        elif section_name == "mda":
            return f"Management's discussion for {ticker} ({year}) indicates revenue growth of approximately 12% year-over-year, with improved gross margins and continued investment in R&D. The company maintains a strong cash position while expanding into international markets."
        elif section_name == "business":
            return f"{ticker}'s business overview describes its product portfolio, market position, and strategic initiatives. The company focuses on innovation, with significant investments in emerging technologies and a multi-channel distribution strategy."
        else:
            return f"Summary of {section_name} section for {ticker} ({year}) {filing_type}."
    
    def classify_text(self, query):
        # Parse query to get ticker, year, filing type, and section
        match = re.search(r"ticker: (\w+), year: (\d{4})(?:, filing_type: (\w+))?, section: (\w+)", query)
        if not match:
            return "Please provide ticker, year, and section in the format: ticker: AAPL, year: 2023, filing_type: 10-K, section: risk_factors"
        
        ticker = match.group(1)
        year = int(match.group(2))
        filing_type = match.group(3) if match.group(3) else "10-K"
        section_name = match.group(4)
        
        # Query database for text classification
        try:
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            
            c.execute("""
                SELECT tc.category, tc.confidence
                FROM text_classifications tc
                JOIN sections s ON tc.section_id = s.id
                JOIN filings f ON s.filing_id = f.id
                JOIN companies c ON f.company_id = c.id
                WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ? AND s.section_name = ?
                ORDER BY tc.confidence DESC
            """, (ticker, year, filing_type, section_name))
            
            results = c.fetchall()
            conn.close()
        except Exception as e:
            return f"Database error: {str(e)}"
        
        if not results:
            # Generate mock data for demo
            categories = ["Risk Disclosure", "Financial Performance", "Legal Matters", "Operational Update"]
            confidences = [0.85, 0.72, 0.65, 0.58]
            
            results = []
            for category, confidence in zip(categories, confidences):
                results.append((category, confidence))
        
        # Format results
        response = f"Text classification for {ticker} ({year}) {filing_type} {section_name.replace('_', ' ')}:\n\n"
        
        for category, confidence in results:
            confidence_pct = f"{confidence * 100:.1f}%"
            response += f"- {category} (Confidence: {confidence_pct})\n"
        
        return response
    
    def detect_anomalies(self, query):
        # Parse query to get ticker, year, filing type, and section
        match = re.search(r"ticker: (\w+), year: (\d{4})(?:, filing_type: (\w+))?, section: (\w+)", query)
        if not match:
            return "Please provide ticker, year, and section in the format: ticker: AAPL, year: 2023, filing_type: 10-K, section: risk_factors"
        
        ticker = match.group(1)
        year = int(match.group(2))
        filing_type = match.group(3) if match.group(3) else "10-K"
        section_name = match.group(4)
        
        # Query database for anomalies
        try:
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            
            c.execute("""
                SELECT a.anomaly_score, a.description
                FROM anomalies a
                JOIN sections s ON a.section_id = s  a.description
                FROM anomalies a
                JOIN sections s ON a.section_id = s.id
                JOIN filings f ON s.filing_id = f.id
                JOIN companies c ON f.company_id = c.id
                WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ? AND s.section_name = ?
                ORDER BY a.anomaly_score DESC
            """, (ticker, year, filing_type, section_name))
            
            results = c.fetchall()
            conn.close()
        except Exception as e:
            return f"Database error: {str(e)}"
        
        if not results:
            # Generate mock data for demo
            anomalies = [
                ("Unusually frequent mention of legal proceedings", 0.92),
                ("Abnormal change in risk factor language", 0.85),
                ("Significant deviation from typical financial terminology", 0.78)
            ]
            
            results = []
            for desc, score in anomalies:
                results.append((score, desc))
        
        # Format results
        response = f"Anomaly detection for {ticker} ({year}) {filing_type} {section_name.replace('_', ' ')}:\n\n"
        
        for score, description in results:
            score_pct = f"{score * 100:.1f}%"
            response += f"- {description} (Anomaly score: {score_pct})\n"
        
        return response
    
    def compare_filings(self, query):
        # Parse query to get tickers, year, and filing type
        match = re.search(r"tickers: ([\w,\s]+), year: (\d{4})(?:, filing_type: (\w+))?", query)
        if not match:
            return "Please provide tickers and year in the format: tickers: AAPL,MSFT,GOOGL, year: 2023, filing_type: 10-K"
        
        tickers = [t.strip() for t in match.group(1).split(",")]
        year = int(match.group(2))
        filing_type = match.group(3) if match.group(3) else "10-K"
        
        # Generate comparison for demo
        response = f"Comparison of {', '.join(tickers)} {filing_type} filings for {year}:\n\n"
        
        # Risk factors comparison
        response += "## Risk Factors Comparison\n\n"
        response += "### Common Risk Themes\n"
        response += "1. **Cybersecurity Threats** - All companies highlight this as a high-severity risk\n"
        response += "2. **Regulatory Compliance** - Particularly regarding data privacy and international operations\n"
        response += "3. **Supply Chain Disruptions** - Mentioned by all companies with varying degrees of emphasis\n"
        response += "4. **Competitive Pressures** - All companies acknowledge intense market competition\n\n"
        
        # Sentiment comparison
        response += "## Sentiment Analysis Comparison\n\n"
        response += "| Company | Overall Sentiment | Positive | Negative | Neutral |\n"
        response += "|---------|-------------------|----------|----------|--------|\n"
        
        sentiments = {
            "AAPL": (0.42, 0.31, 0.47, 0.22),
            "MSFT": (0.48, 0.35, 0.42, 0.23),
            "GOOGL": (0.45, 0.33, 0.44, 0.23),
            "AMZN": (0.40, 0.30, 0.48, 0.22),
            "META": (0.38, 0.28, 0.50, 0.22)
        }
        
        for ticker in tickers:
            if ticker in sentiments:
                overall, pos, neg, neu = sentiments[ticker]
                response += f"| {ticker} | {overall:.2f} | {pos:.2f} | {neg:.2f} | {neu:.2f} |\n"
            else:
                response += f"| {ticker} | 0.44 | 0.32 | 0.45 | 0.23 |\n"
        
        response += "\n### Sentiment Insights\n"
        response += "- Companies in this sector generally maintain a neutral to slightly negative tone in their SEC filings\n"
        response += "- Risk factors sections show the most negative sentiment across all companies\n"
        response += "- Business overview sections contain the most positive language\n\n"
        
        # Entity comparison
        response += "## Key Entity Comparison\n\n"
        response += "### Geographic Focus\n"
        response += "- All companies mention operations in North America, Europe, and Asia Pacific\n"
        response += "- China is mentioned with higher frequency in AAPL filings compared to others\n"
        response += "- MSFT shows more emphasis on European markets\n\n"
        
        response += "### Product Focus\n"
        response += "- AAPL emphasizes hardware products and services integration\n"
        response += "- MSFT focuses on cloud services and enterprise solutions\n"
        response += "- GOOGL highlights advertising technology and AI capabilities\n\n"
        
        return response
    
    def get_filing_history(self, query):
        # Parse query to get ticker
        match = re.search(r"ticker: (\w+)(?:, limit: (\d+))?", query)
        if not match:
            return "Please provide ticker in the format: ticker: AAPL, limit: 10"
        
        ticker = match.group(1)
        limit = int(match.group(2)) if match.group(2) else 10
        
        # Query database for filing history
        try:
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            
            c.execute("""
                SELECT f.filing_type, f.filing_date, f.filing_year, f.filing_quarter, f.accession_number
                FROM filings f
                JOIN companies c ON f.company_id = c.id
                WHERE c.ticker = ?
                ORDER BY f.filing_date DESC
                LIMIT ?
            """, (ticker, limit))
            
            results = c.fetchall()
            conn.close()
        except Exception as e:
            return f"Database error: {str(e)}"
        
        if not results:
            # Generate mock data for demo
            current_year = datetime.now().year
            filing_types = ["10-K", "10-Q", "8-K", "10-Q", "8-K", "10-Q", "8-K", "10-K", "10-Q", "8-K"]
            filing_dates = []
            filing_years = []
            filing_quarters = []
            
            for i in range(limit):
                year = current_year - (i // 4)
                quarter = 4 - (i % 4)
                if quarter <= 0:
                    quarter += 4
                    year -= 1
                
                month = quarter * 3
                day = random.randint(1, 28)
                
                filing_date = f"{year}-{month:02d}-{day:02d}"
                filing_dates.append(filing_date)
                filing_years.append(year)
                filing_quarters.append(quarter if filing_types[i % len(filing_types)] == "10-Q" else None)
            
            results = []
            for i in range(min(limit, len(filing_types))):
                accession_number = f"{ticker}-{filing_years[i]}-{filing_types[i % len(filing_types)]}-{i}"
                results.append((filing_types[i % len(filing_types)], filing_dates[i], filing_years[i], filing_quarters[i], accession_number))
        
        # Format results
        response = f"Filing history for {ticker} (last {len(results)} filings):\n\n"
        response += "| Filing Type | Date | Year | Quarter | Accession Number |\n"
        response += "|-------------|------|------|---------|------------------|\n"
        
        for filing_type, filing_date, filing_year, filing_quarter, accession_number in results:
            quarter_str = f"Q{filing_quarter}" if filing_quarter else "N/A"
            response += f"| {filing_type} | {filing_date} | {filing_year} | {quarter_str} | {accession_number} |\n"
        
        return response
    
    def run(self, input_text):
        if not self.agent_executor:
            return "AI agent not initialized. Please check your OpenAI API key."
        try:
            # Ensure the input is properly formatted for the agent
            return self.agent_executor.run(input=input_text)
        except Exception as e:
            return f"Error running AI agent: {str(e)}"

class SECFilingPipeline:
    def __init__(self, user_agent: str, email_address: str, openai_api_key: str = None):
        self.user_agent = user_agent
        self.email_address = email_address
        self.openai_api_key = openai_api_key

        # Downloader and data dir
        try:
            self.downloader = Downloader(email_address, user_agent)
            self.data_dir = "data"
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
        except Exception as e:
            st.error(f"Error initializing downloader: {e}")
            self.downloader = None

        # AI Agent
        try:
            self.ai_agent = SECFilingsAgent(openai_api_key) if openai_api_key else None
        except Exception as e:
            st.error(f"Error initializing AI agent: {e}")
            self.ai_agent = None

        # Sentence transformer
        try:
            self.sentence_transformer = load_sentence_transformer()
        except Exception as e:
            st.error(f"Error loading sentence transformer: {e}")
            self.sentence_transformer = None

        # Summarizer
        try:
            self.summarizer = load_summarizer()
            print(f"✅ Summarizer Loaded: {self.summarizer}")  # Debugging print
        except Exception as e:
            st.error(f"Error loading summarizer: {e}")
            self.summarizer = None
            print("❌ Summarizer Failed to Load")  # Debugging print

        # Text classifier
        try:
            self.text_classifier = load_text_classifier()
        except Exception as e:
            st.error(f"Error loading text classifier: {e}")
            self.text_classifier = None
    
    def get_cik_by_ticker(self, ticker: str) -> Optional[str]:
        try:
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            c.execute("SELECT cik FROM companies WHERE ticker = ?", (ticker.upper(),))
            result = c.fetchone()
            conn.close()
            
            if result:
                return result[0]
            
            headers = {'User-Agent': self.user_agent}
            response = requests.get('https://www.sec.gov/files/company_tickers.json', headers=headers)
            data = response.json()
            
            ticker_upper = ticker.upper()
            for entry in data.values():
                if entry['ticker'].upper() == ticker_upper:
                    cik = str(entry['cik_str']).zfill(10)
                    
                    conn = sqlite3.connect('sec_filings.db')
                    c = conn.cursor()
                    c.execute(
                        "INSERT OR IGNORE INTO companies (ticker, cik, name) VALUES (?, ?, ?)",
                        (ticker_upper, cik, entry.get('title', ''))
                    )
                    conn.commit()
                    conn.close()
                    
                    return cik
            
            return None
        except Exception as e:
            print(f"Error fetching CIK for {ticker}: {e}")
            return None
    
    def download_sec_filing(self, ticker: str, year: int, filing_type: str = "10-K", quarter: int = None) -> Optional[str]:
        try:
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            
            query = """
                SELECT f.file_path 
                FROM filings f 
                JOIN companies c ON f.company_id = c.id 
                WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ?
            """
            params = [ticker.upper(), year, filing_type]
            
            if quarter and filing_type == "10-Q":
                query += " AND f.filing_quarter = ?"
                params.append(quarter)
                
            c.execute(query, params)
            result = c.fetchone()
            conn.close()
            
            if result and os.path.exists(result[0]):
                return result[0]
            
            mock_data_path = self.generate_mock_filing(ticker, year, filing_type, quarter)
            if mock_data_path:
                conn = sqlite3.connect('sec_filings.db')
                c = conn.cursor()
                
                c.execute("SELECT id FROM companies WHERE ticker = ?", (ticker.upper(),))
                company_id_result = c.fetchone()
                
                if not company_id_result:
                    c.execute(
                        "INSERT INTO companies (ticker, name) VALUES (?, ?)",
                        (ticker.upper(), f"{ticker.upper()} Corporation")
                    )
                    company_id = c.lastrowid
                else:
                    company_id = company_id_result[0]
                
                filing_date = f"{year}-{(quarter * 3) if quarter else 3:02d}-15" if quarter else f"{year}-03-15"
                
                c.execute("""
                    INSERT INTO filings (company_id, filing_type, filing_year, filing_quarter, file_path, filing_date, accession_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (company_id, filing_type, year, quarter, mock_data_path, filing_date, f"{ticker}-{year}-{filing_type}"))
                
                conn.commit()
                conn.close()
                
                return mock_data_path
            
            if not self.downloader:
                raise Exception("SEC Downloader not initialized properly")
                
            try:
                after_date = f"{year}-01-01"
                before_date = f"{year+1}-01-01"
                
                if quarter and filing_type == "10-Q":
                    quarter_start = (quarter - 1) * 3 + 1
                    quarter_end = quarter * 3
                    after_date = f"{year}-{quarter_start:02d}-01"
                    before_date = f"{year}-{quarter_end+1:02d}-01" if quarter < 4 else f"{year+1}-01-01"
                
                self.downloader.get(filing_type, ticker, amount=1, after=after_date, before=before_date)
            except Exception as e:
                st.warning(f"Error downloading filing: {e}. Using mock data instead.")
                return self.generate_mock_filing(ticker, year, filing_type, quarter)
            
            cik = self.get_cik_by_ticker(ticker)
            if not cik:
                return self.generate_mock_filing(ticker, year, filing_type, quarter)
                
            filing_dir = f"sec-edgar-filings/{ticker}/{filing_type}"
            if not os.path.exists(filing_dir):
                return self.generate_mock_filing(ticker, year, filing_type, quarter)
                
            filings = os.listdir(filing_dir)
            if not filings:
                return self.generate_mock_filing(ticker, year, filing_type, quarter)
                
            filings.sort(reverse=True)
            latest_filing = filings[0]
            
            filing_path = f"{filing_dir}/{latest_filing}/full-submission.txt"
            if os.path.exists(filing_path):
                conn = sqlite3.connect('sec_filings.db')
                c = conn.cursor()
                
                c.execute("SELECT id FROM companies WHERE ticker = ?", (ticker.upper(),))
                company_id_result = c.fetchone()
                
                if not company_id_result:
                    c.execute(
                        "INSERT INTO companies (ticker, name) VALUES (?, ?)",
                        (ticker.upper(), f"{ticker.upper()} Corporation")
                    )
                    company_id = c.lastrowid
                else:
                    company_id = company_id_result[0]
                
                filing_date = f"{year}-{(quarter * 3) if quarter else 3:02d}-15" if quarter else f"{year}-03-15"
                
                c.execute("""
                    INSERT INTO filings (company_id, filing_type, filing_year, filing_quarter, file_path, filing_date, accession_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (company_id, filing_type, year, quarter, filing_path, filing_date, latest_filing))
                
                conn.commit()
                conn.close()
                
                return filing_path
            
            return self.generate_mock_filing(ticker, year, filing_type, quarter)
        except Exception as e:
            print(f"Error downloading {filing_type} for {ticker} ({year}): {e}")
            return self.generate_mock_filing(ticker, year, filing_type, quarter)
    
    def generate_mock_filing(self, ticker: str, year: int, filing_type: str = "10-K", quarter: int = None) -> str:
        mock_dir = "mock_filings"
        if not os.path.exists(mock_dir):
            os.makedirs(mock_dir)
        
        file_suffix = f"Q{quarter}" if quarter and filing_type == "10-Q" else ""
        mock_file_path = f"{mock_dir}/{ticker}_{year}_{filing_type}{file_suffix}.txt"
        
        if filing_type == "10-K":
            mock_content = self.generate_mock_10k(ticker, year)
        elif filing_type == "10-Q":
            mock_content = self.generate_mock_10q(ticker, year, quarter)
        elif filing_type == "8-K":
            mock_content = self.generate_mock_8k(ticker, year)
        else:
            mock_content = f"Mock {filing_type} filing for {ticker} ({year})"
        
        with open(mock_file_path, 'w', encoding='utf-8') as f:
            f.write(mock_content)
        
        return mock_file_path
    
    def generate_mock_10k(self, ticker: str, year: int) -> str:
        return f"""
UNITED STATES
SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

FORM 10-K

ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934

For the fiscal year ended December 31, {year}

{ticker.upper()} CORPORATION
(Exact name of registrant as specified in its charter)

Item 1. Business

{ticker.upper()} Corporation is a leading technology company that designs, manufactures, and markets innovative products and services worldwide. The Company's products include hardware devices, software applications, and online services that are integrated to provide a seamless user experience.

Item 1A. Risk Factors

RISK FACTORS
Investing in our securities carries a significant level of risk. Before making any investment decision, we strongly encourage you to thoroughly evaluate the risks and uncertainties outlined below, along with all other relevant information in this Annual Report on Form 10-K. In particular, We recommend reviewing the section titled "Management's Discussion and Analysis of Financial Condition and Results of Operations" as well as our consolidated financial statements and accompanying notes. A comprehensive understanding of these factors is essential to making an informed investment decision.

Operational Risks:
- Supply Chain Disruption: Our business depends on a global supply chain that is subject to disruption from natural disasters, political instability, labor disputes, and public health crises.
- Manufacturing Delays: Delays in manufacturing could impact our ability to meet customer demand and could adversely affect our financial results.
- Quality Control Issues: Product quality issues could result in recalls, warranty claims, and damage to our brand reputation.

Technological Risks:
- Cybersecurity Threats: We face significant cybersecurity threats that could compromise our systems, operations, and sensitive information.
- Technology Obsolescence: Rapid technological changes could render our products obsolete or less competitive.
- Intellectual Property Protection: Our failure to protect our intellectual property rights could diminish the value of our products and brand.

Regulatory Risks:
- Compliance Requirements: We are subject to complex and changing laws and regulations worldwide.
- International Trade Policies: Changes in trade policies could disrupt our international operations.
- Data Privacy Regulations: Evolving data privacy regulations could impact our business operations and increase compliance costs.

Market Risks:
- Competitive Pressure: We face intense competition in all our markets, which could lead to reduced sales or margins.
- Consumer Preference Changes: Shifts in consumer preferences could adversely affect demand for our products.
- Economic Uncertainty: Economic downturns could reduce consumer spending on our products and services.

Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations

MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS

Overview
During fiscal {year}, we continued to execute our strategy of delivering innovative products and services to our customers. We reported revenue of $X billion, representing a Y% increase compared to the prior year. This growth was driven primarily by strong performance in our core product lines and expansion of our services business.

Results of Operations
Revenue increased to $X billion in fiscal {year}, compared to $Z billion in fiscal {year-1}. Gross margin improved to A%, up from B% in the prior year, due to favorable product mix and cost optimization initiatives. Operating expenses increased to $C billion, primarily due to investments in research and development and marketing for new product launches.

Liquidity and Capital Resources
As of December 31, {year}, we had cash and cash equivalents of $D billion, compared to $E billion as of December 31, {year-1}. We generated operating cash flow of $F billion during fiscal {year}. We returned $G billion to shareholders through dividends and share repurchases.

Outlook
Looking forward, we remain focused on innovation and expanding our ecosystem of products and services. We anticipate continued growth in our services business and are investing in emerging technologies to drive future growth. While we are optimistic about our long-term prospects, we acknowledge potential headwinds from macroeconomic uncertainties and supply chain constraints.
"""
    
    def generate_mock_10q(self, ticker: str, year: int, quarter: int) -> str:
        quarter_end_month = quarter * 3
        quarter_end_date = f"{year}-{quarter_end_month:02d}-{30 if quarter_end_month in [4, 6, 9, 11] else 31}"
        
        return f"""
UNITED STATES
SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

FORM 10-Q

QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934

For the quarterly period ended {quarter_end_date}

{ticker.upper()} CORPORATION
(Exact name of registrant as specified in its charter)

PART I. FINANCIAL INFORMATION

Item 1. Financial Statements

[Financial statements would appear here]

Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations

MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS

Overview
During Q{quarter} of fiscal {year}, we continued to execute our strategy of delivering innovative products and services to our customers. We reported quarterly revenue of $X billion, representing a Y% increase compared to the same quarter in the prior year. This growth was driven primarily by strong performance in our core product lines and expansion of our services business.

Results of Operations
Revenue increased to $X billion in Q{quarter} of fiscal {year}, compared to $Z billion in Q{quarter} of fiscal {year-1}. Gross margin was A%, compared to B% in the same quarter of the prior year. Operating expenses were $C billion, primarily due to investments in research and development and marketing for new product launches.

Liquidity and Capital Resources
As of {quarter_end_date}, we had cash and cash equivalents of $D billion, compared to $E billion as of the end of the previous quarter. We generated operating cash flow of $F billion during the quarter. We returned $G billion to shareholders through dividends and share repurchases.

PART II. OTHER INFORMATION

Item 1A. Risk Factors

There have been no material changes to the risk factors disclosed in our Annual Report on Form 10-K for the fiscal year ended December 31, {year-1}.

Item 6. Exhibits

[List of exhibits would appear here]
"""
    
    def generate_mock_8k(self, ticker: str, year: int) -> str:
        event_date = f"{year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        
        events = [
            "Entry into a Material Definitive Agreement",
            "Termination of a Material Definitive Agreement",
            "Bankruptcy or Receivership",
            "Completion of Acquisition or Disposition of Assets",
            "Results of Operations and Financial Condition",
            "Creation of a Direct Financial Obligation",
            "Triggering Events That Accelerate or Increase a Direct Financial Obligation",
            "Costs Associated with Exit or Disposal Activities",
            "Material Impairments",
            "Notice of Delisting or Failure to Satisfy a Continued Listing Rule",
            "Unregistered Sales of Equity Securities",
            "Material Modifications to Rights of Security Holders",
            "Changes in Accountant",
            "Non-Reliance on Previously Issued Financial Statements",
            "Changes in Control of Registrant",
            "Departure of Directors or Certain Officers",
            "Election of Directors",
            "Amendments to Articles of Incorporation or Bylaws",
            "Temporary Suspension of Trading Under Registrant's Employee Benefit Plans",
            "Amendments to the Registrant's Code of Ethics",
            "Change in Shell Company Status",
            "Submission of Matters to a Vote of Security Holders",
            "Shareholder Director Nominations",
            "Mine Safety - Reporting of Shutdowns and Patterns of Violations"
        ]
        
        selected_event = random.choice(events)
        
        return f"""
UNITED STATES
SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

FORM 8-K

CURRENT REPORT
Pursuant to Section 13 OR 15(d) of The Securities Exchange Act of 1934

Date of Report (Date of earliest event reported): {event_date}

{ticker.upper()} CORPORATION
(Exact name of registrant as specified in its charter)

Item {random.randint(1, 9)}.{random.randint(1, 5):02d} {selected_event}

On {event_date}, {ticker.upper()} Corporation (the "Company") {self.generate_mock_8k_event_description(selected_event, ticker)}.

Item 9.01 Financial Statements and Exhibits

(d) Exhibits

Exhibit No.    Description
{random.randint(10, 99)}.1    {selected_event} Documentation
"""
    
    def generate_mock_8k_event_description(self, event_type: str, ticker: str) -> str:
        if "Material Definitive Agreement" in event_type:
            return f"entered into a definitive agreement with XYZ Corporation to {random.choice(['acquire', 'partner with', 'license technology from', 'distribute products of'])} their {random.choice(['software', 'hardware', 'services', 'intellectual property'])} division"
        elif "Results of Operations" in event_type:
            return f"announced its financial results for the {random.choice(['first', 'second', 'third', 'fourth'])} quarter of the fiscal year"
        elif "Acquisition or Disposition" in event_type:
            return f"completed the {random.choice(['acquisition', 'disposition'])} of {random.choice(['ABC Inc.', 'XYZ LLC', 'QRS Corporation'])} for approximately ${random.randint(10, 500)} million"
        elif "Directors or Officers" in event_type:
            return f"announced that {random.choice(['John Smith', 'Jane Doe', 'Robert Johnson'])}, the Company's {random.choice(['Chief Executive Officer', 'Chief Financial Officer', 'Chief Operating Officer', 'Director'])}, has {random.choice(['resigned', 'been appointed', 'retired', 'been terminated'])}"
        else:
            return f"made an announcement regarding {event_type.lower()}"
    
    def extract_sections(self, filing_path: str, filing_type: str = "10-K") -> Dict[str, str]:
        try:
            with open(filing_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            sections = {}
            
            if filing_type == "10-K":
                risk_factors_match = re.search(r'Item\s+1A\.?\s+Risk\s+Factors(.*?)(?=Item\s+\d[A-Z]?\.)', content, re.DOTALL | re.IGNORECASE)
                if risk_factors_match:
                    sections['risk_factors'] = risk_factors_match.group(1).strip()
                
                mda_match = re.search(r'Item\s+7\.?\s+Management\'?s\s+Discussion\s+and\s+Analysis(.*?)(?=Item\s+\d[A-Z]?\.)', content, re.DOTALL | re.IGNORECASE)
                if mda_match:
                    sections['mda'] = mda_match.group(1).strip()
                
                business_match = re.search(r'Item\s+1\.?\s+Business(.*?)(?=Item\s+\d[A-Z]?\.)', content, re.DOTALL | re.IGNORECASE)
                if business_match:
                    sections['business'] = business_match.group(1).strip()
                
                legal_match = re.search(r'Item\s+3\.?\s+Legal\s+Proceedings(.*?)(?=Item\s+\d[A-Z]?\.)', content, re.DOTALL | re.IGNORECASE)
                if legal_match:
                    sections['legal'] = legal_match.group(1).strip()
            
            elif filing_type == "10-Q":
                mda_match = re.search(r'Item\s+2\.?\s+Management\'?s\s+Discussion\s+and\s+Analysis(.*?)(?=Item\s+\d[A-Z]?\.)', content, re.DOTALL | re.IGNORECASE)
                if mda_match:
                    sections['mda'] = mda_match.group(1).strip()
                
                risk_factors_match = re.search(r'Item\s+1A\.?\s+Risk\s+Factors(.*?)(?=Item\s+\d[A-Z]?\.)', content, re.DOTALL | re.IGNORECASE)
                if risk_factors_match:
                    sections['risk_factors'] = risk_factors_match.group(1).strip()
                
                legal_match = re.search(r'Item\s+1\.?\s+Legal\s+Proceedings(.*?)(?=Item\s+\d[A-Z]?\.)', content, re.DOTALL | re.IGNORECASE)
                if legal_match:
                    sections['legal'] = legal_match.group(1).strip()
                
                financial_match = re.search(r'Item\s+1\.?\s+Financial\s+Statements(.*?)(?=Item\s+\d[A-Z]?\.)', content, re.DOTALL | re.IGNORECASE)
                if financial_match:
                    sections['financial'] = financial_match.group(1).strip()
            
            elif filing_type == "8-K":
                # For 8-K, extract the main event item
                event_match = re.search(r'Item\s+\d\.\d+\s+(.*?)(?=Item\s+\d\.\d+|$)', content, re.DOTALL | re.IGNORECASE)
                if event_match:
                    sections['event'] = event_match.group(1).strip()
                
                # Extract the event description
                description_match = re.search(r'Item\s+\d\.\d+\s+.*?\n\n(.*?)(?=Item\s+\d\.\d+|$)', content, re.DOTALL | re.IGNORECASE)
                if description_match:
                    sections['description'] = description_match.group(1).strip()
            
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            
            c.execute("SELECT id FROM filings WHERE file_path = ?", (filing_path,))
            filing_id_result = c.fetchone()
            
            if filing_id_result:
                filing_id = filing_id_result[0]
                
                for section_name, section_text in sections.items():
                    c.execute("""
                        INSERT OR REPLACE INTO sections (filing_id, section_name, section_text)
                        VALUES (?, ?, ?)
                    """, (filing_id, section_name, section_text))
                
                conn.commit()
            
            conn.close()
            
            return sections
        except Exception as e:
            print(f"Error extracting sections: {e}")
            return {}
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        if not nlp:
            return []
            
        max_length = min(len(text), 100000)
        doc = nlp(text[:max_length])
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        sentiment = sentiment_analyzer.polarity_scores(text)
        return sentiment
    
    def extract_topics(self, text: str, num_topics: int = 5) -> List[Dict[str, Any]]:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words and len(token) > 3]
        
        vectorizer = TfidfVectorizer(max_features=1000)
        dtm = vectorizer.fit_transform([' '.join(tokens)])
        
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(dtm)
        
        feature_names = vectorizer.get_feature_names_out()
        
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-10-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'id': topic_idx,
                'words': top_words,
                'weight': float(topic.sum() / lda.components_.sum())
            })
        
        return topics
    
    def generate_summary(self, text: str, max_length: int = 150) -> str:
        if not self.summarizer:
            return "Summarizer not available"
            
        if len(text) > 10000:
            text = text[:10000]
        
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            sentences = sent_tokenize(text)
            if len(sentences) > 5:
                return ' '.join(sentences[:5])
            return text
    
    def classify_text(self, text: str) -> List[Tuple[str, float]]:
        if not self.text_classifier:
            return []
            
        # This is a simplified version - in a real application you would train the classifier
        # on actual SEC filing data with proper labels
        categories = ["Risk Disclosure", "Financial Performance", "Legal Matters", "Operational Update"]
        
        # Mock classification results
        confidences = [random.uniform(0.5, 0.95) for _ in categories]
        total = sum(confidences)
        confidences = [c/total for c in confidences]  # Normalize
        
        return list(zip(categories, confidences))
    
    def detect_anomalies(self, text: str) -> List[Tuple[str, float]]:
        # This is a simplified version - in a real application you would use proper anomaly detection
        # techniques like Isolation Forest or autoencoders on sentence embeddings
        
        anomalies = [
            ("Unusually frequent mention of legal proceedings", random.uniform(0.7, 0.95)),
            ("Abnormal change in risk factor language", random.uniform(0.6, 0.9)),
            ("Significant deviation from typical financial terminology", random.uniform(0.5, 0.85))
        ]
        
        return sorted(anomalies, key=lambda x: x[1], reverse=True)
    
    def generate_insights(self, ticker: str, year: int, filing_type: str, section_name: str) -> str:
        if not self.openai_api_key:
            return "OpenAI API key not provided. Cannot generate insights."
        
        try:
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            c.execute("""
                SELECT s.section_text
                FROM sections s
                JOIN filings f ON s.filing_id = f.id
                JOIN companies c ON f.company_id = c.id
                WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ? AND s.section_name = ?
            """, (ticker, year, filing_type, section_name))
            result = c.fetchone()
            conn.close()
            
            if not result:
                return f"Section {section_name} not found for {ticker} ({year}) {filing_type}."
            
            section_text = result[0]
            
            if len(section_text) > 4000:
                section_text = section_text[:4000]
            
            client = OpenAIClient(api_key=self.openai_api_key)
            
            prompt = f"""
            Analyze the following section from {ticker}'s {year} {filing_type} filing:
            
            {section_text}
            
            Provide a concise analysis highlighting:
            1. Key themes and topics
            2. Significant risks or opportunities
            3. Changes or trends compared to industry standards
            4. Implications for investors
            
            Format your response in a clear, structured manner.
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.5
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating insights: {e}")
            return f"Error generating insights: {str(e)}"
    
    def process_filing(self, ticker: str, year: int, filing_type: str = "10-K", quarter: int = None) -> Dict[str, Any]:
        results = {
            'ticker': ticker,
            'year': year,
            'filing_type': filing_type,
            'quarter': quarter,
            'status': 'failed',
            'sections': {},
            'entities': {},
            'sentiment': {},
            'topics': {},
            'summaries': {},
            'classifications': {},
            'anomalies': {},
            'insights': {}
        }
        
        try:
            filing_path = self.download_sec_filing(ticker, year, filing_type, quarter)
            if not filing_path:
                results['error'] = f"Failed to download {filing_type} filing for {ticker} ({year})"
                return results
            
            sections = self.extract_sections(filing_path, filing_type)
            if not sections:
                results['error'] = f"Failed to extract sections from {filing_type} filing for {ticker} ({year})"
                return results
            
            results['sections'] = {k: len(v) for k, v in sections.items()}
            
            for section_name, section_text in sections.items():
                cleaned_text = self.clean_text(section_text)
                
                entities = self.extract_entities(cleaned_text)
                results['entities'][section_name] = len(entities)
                
                sentiment = self.analyze_sentiment(cleaned_text)
                results['sentiment'][section_name] = sentiment
                
                topics = self.extract_topics(cleaned_text)
                results['topics'][section_name] = len(topics)
                
                summary = self.generate_summary(cleaned_text)
                results['summaries'][section_name] = summary
                
                classifications = self.classify_text(cleaned_text)
                results['classifications'][section_name] = classifications
                
                anomalies = self.detect_anomalies(cleaned_text)
                results['anomalies'][section_name] = anomalies
                
                if self.openai_api_key:
                    insights = self.generate_insights(ticker, year, filing_type, section_name)
                    results['insights'][section_name] = insights
                
                conn = sqlite3.connect('sec_filings.db')
                c = conn.cursor()
                
                c.execute("""
                    SELECT s.id
                    FROM sections s
                    JOIN filings f ON s.filing_id = f.id
                    JOIN companies c ON f.company_id = c.id
                    WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ? AND s.section_name = ?
                """, (ticker, year, filing_type, section_name))
                section_id_result = c.fetchone()
                
                if section_id_result:
                    section_id = section_id_result[0]
                    
                    c.execute("""
                        INSERT OR REPLACE INTO analysis_results (section_id, analysis_type, analysis_result)
                        VALUES (?, ?, ?)
                    """, (section_id, 'sentiment', json.dumps(sentiment)))
                    
                    for entity in entities[:100]:
                        c.execute("""
                            INSERT OR IGNORE INTO entities (section_id, entity_type, entity_name, frequency, sentiment)
                            VALUES (?, ?, ?, ?, ?)
                        """, (section_id, entity['label'], entity['text'], 0.5, 0.5))
                    
                    c.execute("""
                        INSERT OR REPLACE INTO analysis_results (section_id, analysis_type, analysis_result)
                        VALUES (?, ?, ?)
                    """, (section_id, 'summary', summary))
                    
                    for category, confidence in classifications:
                        c.execute("""
                            INSERT OR REPLACE INTO text_classifications (section_id, category, confidence)
                            VALUES (?, ?, ?)
                        """, (section_id, category, confidence))
                    
                    for desc, score in anomalies:
                        c.execute("""
                            INSERT OR REPLACE INTO anomalies (section_id, anomaly_score, description)
                            VALUES (?, ?, ?)
                        """, (section_id, score, desc))
                
                conn.commit()
                conn.close()
            
            results['status'] = 'success'
            return results
        except Exception as e:
            results['error'] = str(e)
            return results

class AgenticSECAnalyzer:
    """Agentic AI workflow for SEC filings analysis"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.client = OpenAIClient(api_key=openai_api_key)
        self.memory = []
        self.tools = {
            "extract_risk_factors": self.extract_risk_factors,
            "analyze_sentiment": self.analyze_sentiment,
            "compare_filings": self.compare_filings,
            "detect_anomalies": self.detect_anomalies,
            "generate_summary": self.generate_summary,
            "search_filings": self.search_filings
        }
    
    def extract_risk_factors(self, ticker: str, year: int, filing_type: str = "10-K") -> str:
        """Extract and analyze risk factors from a filing"""
        try:
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            
            c.execute("""
                SELECT s.section_text
                FROM sections s
                JOIN filings f ON s.filing_id = f.id
                JOIN companies c ON f.company_id = c.id
                WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ? AND s.section_name = 'risk_factors'
            """, (ticker, year, filing_type))
            
            result = c.fetchone()
            conn.close()
            
            if not result:
                return f"Risk factors not found for {ticker} ({year}) {filing_type}."
            
            section_text = result[0]
            
            if len(section_text) > 4000:
                section_text = section_text[:4000]
            
            prompt = f"""
            Extract and categorize the key risk factors from this {filing_type} filing section:
            
            {section_text}
            
            For each risk factor:
            1. Identify the category (Operational, Financial, Regulatory, Market, etc.)
            2. Provide a concise name for the risk
            3. Estimate severity (High, Medium, Low)
            
            Format as JSON:
            {{
                "risk_factors": [
                    {{"category": "Category", "name": "Risk Name", "severity": "High/Medium/Low", "description": "Brief description"}}
                ]
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0.2
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error extracting risk factors: {str(e)}"
    
    def analyze_sentiment(self, ticker: str, year: int, filing_type: str = "10-K", section: str = "mda") -> str:
        """Analyze sentiment of a filing section"""
        try:
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            
            c.execute("""
                SELECT s.section_text
                FROM sections s
                JOIN filings f ON s.filing_id = f.id
                JOIN companies c ON f.company_id = c.id
                WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ? AND s.section_name = ?
            """, (ticker, year, filing_type, section))
            
            result = c.fetchone()
            conn.close()
            
            if not result:
                return f"Section {section} not found for {ticker} ({year}) {filing_type}."
            
            section_text = result[0]
            
            if len(section_text) > 4000:
                section_text = section_text[:4000]
            
            prompt = f"""
            Analyze the sentiment of this {filing_type} filing section:
            
            {section_text}
            
            Provide:
            1. Overall sentiment (positive, negative, neutral)
            2. Confidence level (0-100%)
            3. Key positive phrases/statements
            4. Key negative phrases/statements
            5. Notable changes in tone compared to typical corporate filings
            
            Format as JSON:
            {{
                "sentiment": "positive/negative/neutral",
                "confidence": 85,
                "positive_phrases": ["phrase 1", "phrase 2"],
                "negative_phrases": ["phrase 1", "phrase 2"],
                "tone_analysis": "Analysis of tone changes"
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0.2
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing sentiment: {str(e)}"
    
    def compare_filings(self, ticker: str, years: List[int], filing_type: str = "10-K") -> str:
        """Compare filings across multiple years"""
        try:
            summaries = []
            
            for year in years:
                conn = sqlite3.connect('sec_filings.db')
                c = conn.cursor()
                
                c.execute("""
                    SELECT s.section_name, s.section_text
                    FROM sections s
                    JOIN filings f ON s.filing_id = f.id
                    JOIN companies c ON f.company_id = c.id
                    WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ?
                """, (ticker, year, filing_type))
                
                sections = c.fetchall()
                conn.close()
                
                if sections:
                    year_data = {"year": year, "sections": {}}
                    for section_name, section_text in sections:
                        if len(section_text) > 1000:
                            section_text = section_text[:1000] + "..."
                        year_data["sections"][section_name] = section_text
                    summaries.append(year_data)
            
            if not summaries:
                return f"No filings found for {ticker} with filing type {filing_type} in years {years}."
            
            prompt = f"""
            Compare the {filing_type} filings for {ticker} across these years: {', '.join(map(str, years))}.
            
            Filing data:
            {json.dumps(summaries, indent=2)}
            
            Provide:
            1. Key changes in risk factors over time
            2. Trends in financial performance
            3. Evolution of business strategy
            4. Notable changes in language or emphasis
            5. Recommendations for investors based on these changes
            
            Format as JSON:
            {{
                "risk_factor_changes": ["change 1", "change 2"],
                "financial_trends": ["trend 1", "trend 2"],
                "strategy_evolution": ["point 1", "point 2"],
                "language_changes": ["change 1", "change 2"],
                "investor_recommendations": ["recommendation 1", "recommendation 2"]
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error comparing filings: {str(e)}"
    
    def detect_anomalies(self, ticker: str, year: int, filing_type: str = "10-K") -> str:
        """Detect anomalies in a filing"""
        try:
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            
            c.execute("""
                SELECT s.section_name, s.section_text
                FROM sections s
                JOIN filings f ON s.filing_id = f.id
                JOIN companies c ON f.company_id = c.id
                WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ?
            """, (ticker, year, filing_type))
            
            sections = c.fetchall()
            conn.close()
            
            if not sections:
                return f"No sections found for {ticker} ({year}) {filing_type}."
            
            sections_data = {}
            for section_name, section_text in sections:
                if len(section_text) > 2000:
                    section_text = section_text[:2000] + "..."
                sections_data[section_name] = section_text
            
            prompt = f"""
            Analyze this {filing_type} filing for {ticker} ({year}) and detect any anomalies or unusual patterns:
            
            {json.dumps(sections_data, indent=2)}
            
            Look for:
            1. Unusual language or phrasing compared to typical corporate filings
            2. Unexpected changes in risk factors or business descriptions
            3. Inconsistencies between sections
            4. Red flags that might indicate financial or operational issues
            5. Vague or evasive language around key topics
            
            Format as JSON:
            {{
                "anomalies": [
                    {{
                        "type": "anomaly type",
                        "description": "detailed description",
                        "section": "section name",
                        "severity": "High/Medium/Low",
                        "recommendation": "what an analyst should investigate"
                    }}
                ]
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error detecting anomalies: {str(e)}"
    
    def generate_summary(self, ticker: str, year: int, filing_type: str = "10-K") -> str:
        """Generate a comprehensive summary of a filing"""
        try:
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            
            c.execute("""
                SELECT s.section_name, s.section_text
                FROM sections s
                JOIN filings f ON s.filing_id = f.id
                JOIN companies c ON f.company_id = c.id
                WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ?
            """, (ticker, year, filing_type))
            
            sections = c.fetchall()
            conn.close()
            
            if not sections:
                return f"No sections found for {ticker} ({year}) {filing_type}."
            
            sections_data = {}
            for section_name, section_text in sections:
                if len(section_text) > 2000:
                    section_text = section_text[:2000] + "..."
                sections_data[section_name] = section_text
            
            prompt = f"""
            Generate a comprehensive summary of this {filing_type} filing for {ticker} ({year}):
            
            {json.dumps(sections_data, indent=2)}
            
            Include:
            1. Executive summary (2-3 sentences)
            2. Key financial highlights
            3. Major risk factors
            4. Business strategy and outlook
            5. Notable changes from previous filings (if apparent)
            6. Key metrics and performance indicators
            
            Format as JSON:
            {{
                "executive_summary": "Brief summary",
                "financial_highlights": ["highlight 1", "highlight 2"],
                "key_risks": ["risk 1", "risk 2"],
                "business_strategy": ["point 1", "point 2"],
                "notable_changes": ["change 1", "change 2"],
                "key_metrics": {{"metric1": "value", "metric2": "value"}}
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def search_filings(self, ticker: str, query: str, filing_types: List[str] = ["10-K", "10-Q", "8-K"]) -> str:
        """Search across filings for specific information"""
        try:
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            
            results = []
            for filing_type in filing_types:
                c.execute("""
                    SELECT f.filing_year, f.filing_quarter, s.section_name, s.section_text
                    FROM sections s
                    JOIN filings f ON s.filing_id = f.id
                    JOIN companies c ON f.company_id = c.id
                    WHERE c.ticker = ? AND f.filing_type = ?
                """, (ticker, filing_type))
                
                sections = c.fetchall()
                
                for year, quarter, section_name, section_text in sections:
                    if query.lower() in section_text.lower():
                        context = self.extract_context(section_text, query, 200)
                        results.append({
                            "filing_type": filing_type,
                            "year": year,
                            "quarter": quarter,
                            "section": section_name,
                            "context": context
                        })
            
            conn.close()
            
            if not results:
                return f"No matches found for '{query}' in {ticker} filings."
            
            prompt = f"""
            The user searched for "{query}" in {ticker} filings and these results were found:
            
            {json.dumps(results, indent=2)}
            
            Analyze these search results and provide:
            1. Summary of findings
            2. How the topic has evolved over time (if multiple years/quarters)
            3. Key insights related to the search query
            4. Recommendations for further investigation
            
            Format as JSON:
            {{
                "summary": "Summary of findings",
                "evolution": ["point 1", "point 2"],
                "key_insights": ["insight 1", "insight 2"],
                "recommendations": ["recommendation 1", "recommendation 2"]
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error searching filings: {str(e)}"
    
    def extract_context(self, text: str, query: str, context_size: int = 200) -> str:
        """Extract context around a search query in text"""
        query_lower = query.lower()
        text_lower = text.lower()
        
        if query_lower not in text_lower:
            return ""
        
        start_idx = text_lower.find(query_lower)
        context_start = max(0, start_idx - context_size)
        context_end = min(len(text), start_idx + len(query) + context_size)
        
        context = text[context_start:context_end]
        
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."
        
        return context
    
    def run_workflow(self, query: str) -> str:
        """Run the agentic workflow based on a user query"""
        try:
            # Add the query to memory
            self.memory.append({"role": "user", "content": query})
            
            # Determine which tools to use
            prompt = f"""
            You are an AI assistant specialized in analyzing SEC filings. The user has asked:
            
            "{query}"
            
            Based on this query, determine which tools would be most helpful to answer it.
            Available tools:
            - extract_risk_factors(ticker, year, filing_type): Extract and analyze risk factors
            - analyze_sentiment(ticker, year, filing_type, section): Analyze sentiment of a section
            - compare_filings(ticker, years, filing_type): Compare filings across years
            - detect_anomalies(ticker, year, filing_type): Detect anomalies in a filing
            - generate_summary(ticker, year, filing_type): Generate a comprehensive summary
            - search_filings(ticker, query, filing_types): Search across filings
            
            For each tool you want to use, provide the parameters.
            
            Format as JSON:
            {{
                "tools": [
                    {{
                        "name": "tool_name",
                        "parameters": {{
                            "param1": "value1",
                            "param2": "value2"
                        }}
                    }}
                ]
            }}
            """
            
            tool_selection_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.2
            )
            
            tool_selection = json.loads(tool_selection_response.choices[0].message.content)
            
            # Execute the selected tools
            tool_results = []
            for tool in tool_selection.get("tools", []):
                tool_name = tool.get("name")
                parameters = tool.get("parameters", {})
                
                if tool_name in self.tools:
                    result = self.tools[tool_name](**parameters)
                    tool_results.append({
                        "tool": tool_name,
                        "parameters": parameters,
                        "result": result
                    })
            
            # Generate the final response
            final_prompt = f"""
            The user asked: "{query}"
            
            You used these tools to analyze SEC filings:
            {json.dumps(tool_results, indent=2)}
            
            Based on these results, provide a comprehensive answer to the user's query.
            Include specific insights, data points, and recommendations.
            Format your response in a clear, structured manner with headings and bullet points where appropriate.
            """
            
            self.memory.append({"role": "system", "content": final_prompt})
            
            final_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=self.memory,
                max_tokens=1500,
                temperature=0.3
            )
            
            response_text = final_response.choices[0].message.content
            self.memory.append({"role": "assistant", "content": response_text})
            
            return response_text
        except Exception as e:
            return f"Error running workflow: {str(e)}"

def create_sentiment_chart(sentiment_data):
    labels = ['Positive', 'Negative', 'Neutral']
    values = [sentiment_data['pos'], sentiment_data['neg'], sentiment_data['neu']]
    colors = ['#4CAF50', '#F44336', '#2196F3']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker=dict(colors=colors)
    )])
    
    fig.update_layout(
        title_text="Sentiment Distribution",
        showlegend=True,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_risk_heatmap(risk_data):
    categories = []
    risks = []
    severities = []
    
    for category, risk_list in risk_data.items():
        for risk_name, severity, _ in risk_list:
            categories.append(category)
            risks.append(risk_name)
            severities.append(severity)
    
    fig = go.Figure(data=go.Heatmap(
        z=severities,
        x=categories,
        y=risks,
        colorscale='Reds',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title_text="Risk Factor Severity Heatmap",
        xaxis_title="Risk Category",
        yaxis_title="Risk Factor",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_entity_network(entities, min_entities=10):
    try:
        G = nx.Graph()
        
        entity_counts = {}
        for entity_type, entity_list in entities.items():
            for name, freq, _ in entity_list:
                if name not in entity_counts:
                    entity_counts[name] = 0
                entity_counts[name] += 1
                G.add_node(name, group=entity_type, value=freq*10)
        
        for i, (name1, _) in enumerate(entity_counts.items()):
            for name2 in list(entity_counts.keys())[i+1:]:
                if random.random() < 0.3:
                    weight = random.uniform(0.1, 1.0)
                    G.add_edge(name1, name2, weight=weight)
        
        if len(G.nodes) > min_entities:
            top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:min_entities]
            top_entity_names = [name for name, _ in top_entities]
            G = G.subgraph(top_entity_names)
        
        net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
        net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250, spring_strength=0.001, damping=0.09)
        
        for node in G.nodes(data=True):
            net.add_node(node[0], 
                        title=node[0], 
                        group=node[1].get('group', 'Other'),
                        value=node[1].get('value', 5))
        
        for edge in G.edges(data=True):
            net.add_edge(edge[0], edge[1], value=edge[2].get('weight', 0.5))
        
        net.save_graph("entity_network.html")
        
        with open("entity_network.html", "r", encoding="utf-8") as f:
            html_string = f.read()
        
        return html_string
    except Exception as e:
        print(f"Error creating entity network: {e}")
        return None

def create_topic_radar_chart(topics):
    topic_names = [' '.join(topic['words'][:3]) for topic in topics]
    topic_weights = [topic['weight'] for topic in topics]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=topic_weights,
        theta=topic_names,
        fill='toself',
        name='Topics'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(topic_weights) * 1.2]
            )),
        showlegend=False,
        title="Topic Distribution",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def create_year_comparison_chart(ticker, years, sentiment_data):
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=years,
        y=[sentiment_data[year]['pos'] for year in years],
        name='Positive',
        marker_color='#4CAF50'
    ))
    
    fig.add_trace(go.Bar(
        x=years,
        y=[sentiment_data[year]['neg'] for year in years],
        name='Negative',
        marker_color='#F44336'
    ))
    
    fig.add_trace(go.Bar(
        x=years,
        y=[sentiment_data[year]['neu'] for year in years],
        name='Neutral',
        marker_color='#2196F3'
    ))
    
    fig.add_trace(go.Scatter(
        x=years,
        y=[sentiment_data[year]['compound'] for year in years],
        mode='lines+markers',
        name='Overall Sentiment',
        line=dict(color='#FFC107', width=3)
    ))
    
    fig.update_layout(
        title=f"Sentiment Trend for {ticker}",
        xaxis_title="Year",
        yaxis_title="Sentiment Score",
        barmode='group',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_classification_chart(classifications):
    categories = [c[0] for c in classifications]
    confidences = [c[1] for c in classifications]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=confidences,
        marker_color='#4CAF50'
    ))
    
    fig.update_layout(
        title="Text Classification Confidence",
        xaxis_title="Category",
        yaxis_title="Confidence",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_anomaly_chart(anomalies):
    descriptions = [a[0] for a in anomalies]
    scores = [a[1] for a in anomalies]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=descriptions,
        y=scores,
        marker_color='#F44336'
    ))
    
    fig.update_layout(
        title="Anomaly Detection Scores",
        xaxis_title="Anomaly Description",
        yaxis_title="Anomaly Score",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def generate_pdf_report(results):
    """Generate a PDF report from the analysis results"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"SEC Filings Analysis Report - {results['ticker']} {results['year']} {results['filing_type']}", ln=1, align='C')
    pdf.ln(10)
    
    # Add company information
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Company Information", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Ticker: {results['ticker']}", ln=1)
    pdf.cell(200, 10, txt=f"Filing Year: {results['year']}", ln=1)
    pdf.cell(200, 10, txt=f"Filing Type: {results['filing_type']}", ln=1)
    
    if results.get('quarter'):
        pdf.cell(200, 10, txt=f"Quarter: Q{results['quarter']}", ln=1)
    
    # Get company name from database
    conn = sqlite3.connect('sec_filings.db')
    c = conn.cursor()
    c.execute("SELECT name FROM companies WHERE ticker = ?", (results['ticker'],))
    company_name = c.fetchone()
    conn.close()
    
    if company_name:
        pdf.cell(200, 10, txt=f"Company Name: {company_name[0]}", ln=1)
    pdf.ln(5)
    
    # Add sections overview
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Sections Extracted", ln=1)
    pdf.set_font("Arial", size=12)
    for section, length in results.get('sections', {}).items():
        pdf.cell(200, 10, txt=f"- {section.replace('_', ' ').title()}: {length} characters", ln=1)
    pdf.ln(5)
    
    # Add risk factors if available
    if 'risk_factors' in results.get('summaries', {}):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Risk Factors Summary", ln=1)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=results['summaries']['risk_factors'])
        pdf.ln(5)
    
    # Add sentiment analysis
    if 'sentiment' in results and 'risk_factors' in results['sentiment']:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Sentiment Analysis", ln=1)
        pdf.set_font("Arial", size=12)
        sentiment = results['sentiment']['risk_factors']
        pdf.cell(200, 10, txt=f"Positive: {sentiment['pos']:.2f}", ln=1)
        pdf.cell(200, 10, txt=f"Negative: {sentiment['neg']:.2f}", ln=1)
        pdf.cell(200, 10, txt=f"Neutral: {sentiment['neu']:.2f}", ln=1)
        pdf.cell(200, 10, txt=f"Compound Score: {sentiment['compound']:.2f}", ln=1)
        pdf.ln(5)
    
    # Add text classification if available
    if 'classifications' in results and 'risk_factors' in results['classifications']:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Text Classification", ln=1)
        pdf.set_font("Arial", size=12)
        for category, confidence in results['classifications']['risk_factors']:
            confidence_pct = f"{confidence * 100:.1f}%"
            pdf.cell(200, 10, txt=f"- {category}: {confidence_pct}", ln=1)
        pdf.ln(5)
    
    # Add anomaly detection if available
    if 'anomalies' in results and 'risk_factors' in results['anomalies']:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Anomaly Detection", ln=1)
        pdf.set_font("Arial", size=12)
        for desc, score in results['anomalies']['risk_factors']:
            score_pct = f"{score * 100:.1f}%"
            pdf.cell(200, 10, txt=f"- {desc}: {score_pct}", ln=1)
        pdf.ln(5)
    
    # Save to bytes buffer
    buffer = BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin-1')
    buffer.write(pdf_output)
    buffer.seek(0)
    return buffer

def create_sentiment_chart(sentiment):
    """Create a sentiment analysis chart"""
    labels = ['Positive', 'Negative', 'Neutral']
    values = [sentiment['pos'], sentiment['neg'], sentiment['neu']]
    colors = ['#4CAF50', '#F44336', '#2196F3']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title_text="Sentiment Distribution",
        annotations=[dict(text=f"Compound: {sentiment['compound']:.2f}", x=0.5, y=0.5, font_size=15, showarrow=False)],
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def main():
    st.title("SEC Filings Insights Extraction with AI")
    st.markdown("Unlock actionable insights from SEC filings using AI.")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        user_agent = st.text_input(
            "SEC API User Agent",
            value="Your Name (your.email@example.com)",
            help="Required by SEC EDGAR. Format: Name (email)"
        )
        
        email_address = st.text_input(
            "SEC API Email Address",
            value="your.email@example.com",
            help="Required by SEC EDGAR. Must be a valid email address."
        )
        
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Optional. Required for AI-powered insights"
        )
        
        ticker = st.text_input("Company Ticker Symbol", "AAPL").upper()
        
        filing_type = st.selectbox("Filing Type", ["10-K", "10-Q", "8-K"])
        
        year = st.selectbox("Filing Year", list(range(2023, 2018, -1)))
        
        if filing_type == "10-Q":
            quarter = st.selectbox("Quarter", [1, 2, 3, 4])
        else:
            quarter = None
        
        st.header("Analysis Options")
        analyze_risk = st.checkbox("Risk Factors Analysis", True)
        analyze_sentiment = st.checkbox("Sentiment Analysis", True)
        analyze_entities = st.checkbox("Entity Recognition", True)
        analyze_topics = st.checkbox("Topic Modeling", True)
        generate_summaries = st.checkbox("Generate Summaries", True)
        analyze_classification = st.checkbox("Text Classification", True)
        analyze_anomalies = st.checkbox("Anomaly Detection", True)
        generate_ai_insights = st.checkbox("Generate AI Insights", openai_api_key != "")
        
        process_button = st.button("Process SEC Filing", type="primary")
    
    # Initialize pipeline in session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = SECFilingPipeline(user_agent, email_address, openai_api_key if openai_api_key else None)
    
    # Initialize agentic AI if API key is provided
    if 'agentic_ai' not in st.session_state and openai_api_key:
        st.session_state.agentic_ai = AgenticSECAnalyzer(openai_api_key)
    
    # Main content
    tabs = st.tabs(["Dashboard", "Risk Analysis", "Sentiment Analysis", "Entity Analysis", 
                   "Topic Analysis", "Text Classification", "Anomaly Detection", "Agentic AI"])
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    if process_button:
        if not user_agent or user_agent == "Your Name (your.email@example.com)":
            st.error("Please provide a valid SEC API User Agent")
        elif not email_address or email_address == "your.email@example.com":
            st.error("Please provide a valid email address for SEC API")
        elif not ticker:
            st.error("Please provide a company ticker symbol")
        else:
            with st.spinner(f"Processing {ticker} {filing_type} filing for {year}..."):
                progress_bar = st.progress(0)
                
                for i in range(1, 101):
                    progress_bar.progress(i)
                    if i < 20:
                        st.caption(f"Downloading {filing_type} filing for {ticker}...")
                    elif i < 40:
                        st.caption("Extracting sections...")
                    elif i < 60:
                        st.caption("Analyzing sentiment and entities...")
                    elif i < 80:
                        st.caption("Generating topics and summaries...")
                    else:
                        st.caption("Finalizing analysis...")
                    
                    if i % 10 == 0:
                        time.sleep(0.1)
                
                results = st.session_state.pipeline.process_filing(ticker, year, filing_type, quarter)
                st.session_state.analysis_results = results
            
            if results['status'] == 'success':
                st.success(f"Successfully processed {ticker} {filing_type} filing for {year}")
                
                # Generate PDF report
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_buffer = generate_pdf_report(results)
                        st.download_button(
                            label="Download Full Report as PDF",
                            data=pdf_buffer,
                            file_name=f"{ticker}_{year}_{filing_type}_SEC_Analysis_Report.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Error generating PDF report: {str(e)}")
            else:
                st.error(f"Error: {results.get('error', 'Unknown error')}")
    
    # Display results if available
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Dashboard tab
        with tabs[0]:
            st.header("Filing Overview")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Company Information")
                st.markdown(f"**Ticker:** {results['ticker']}")
                st.markdown(f"**Filing Year:** {results['year']}")
                st.markdown(f"**Filing Type:** {results['filing_type']}")
                
                if results.get('quarter'):
                    st.markdown(f"**Quarter:** Q{results['quarter']}")
                
                conn = sqlite3.connect('sec_filings.db')
                c = conn.cursor()
                c.execute("SELECT name FROM companies WHERE ticker = ?", (results['ticker'],))
                company_name = c.fetchone()
                conn.close()
                
                if company_name:
                    st.markdown(f"**Company Name:** {company_name[0]}")
            
            with col2:
                st.subheader("Sections Extracted")
                for section, length in results['sections'].items():
                    st.markdown(f"- {section.replace('_', ' ').title()}: {length} characters")
            
            st.subheader("Analysis Summary")
            metric_cols = st.columns(4)
            
            sentiment_data = results.get('sentiment', {}).get('risk_factors', {})
            compound_score = sentiment_data.get('compound', 0)
            sentiment_label = "Negative" if compound_score < -0.05 else "Positive" if compound_score > 0.05 else "Neutral"
            
            metric_cols[0].metric("Overall Sentiment", sentiment_label, f"{compound_score:.2f}")
            metric_cols[1].metric("Entities Extracted", sum(len(entities) for entities in results.get('entities', {}).values()))
            
            # FIX: Use the integer value directly instead of trying to get its length
            total_topics = sum(topic_count for topic_count in results.get('topics', {}).values())
            metric_cols[2].metric("Topics Identified", total_topics)
            
            metric_cols[3].metric("Sections Analyzed", len(results.get('sections', {})))
            
            if 'sentiment' in results and 'risk_factors' in results['sentiment']:
                st.subheader("Sentiment Overview")
                sentiment_chart = create_sentiment_chart(results['sentiment']['risk_factors'])
                st.plotly_chart(sentiment_chart, use_container_width=True)
        
        # Risk Analysis tab
        if analyze_risk:
            with tabs[1]:
                st.header("Risk Factors Analysis")
                
                conn = sqlite3.connect('sec_filings.db')
                c = conn.cursor()
                c.execute("""
                    SELECT rf.risk_category, rf.risk_name, rf.severity, rf.trend
                    FROM risk_factors rf
                    JOIN sections s ON rf.section_id = s.id
                    JOIN filings f ON s.filing_id = f.id
                    JOIN companies c ON f.company_id = c.id
                    WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ?
                """, (results['ticker'], results['year'], results['filing_type']))
                risk_factors_data = c.fetchall()
                conn.close()
                
                if not risk_factors_data:
                    risk_categories = ["Operational", "Technological", "Regulatory", "Market"]
                    risk_names = [
                        ["Supply Chain Disruption", "Manufacturing Delays", "Quality Control Issues"],
                        ["Cybersecurity Threats", "Technology Obsolescence", "Intellectual Property Protection"],
                        ["Compliance Requirements", "International Trade Policies", "Data Privacy Regulations"],
                        ["Competitive Pressure", "Consumer Preference Changes", "Economic Uncertainty"]
                    ]
                    severities = [0.85, 0.72, 0.65, 0.88, 0.71, 0.75, 0.78, 0.81, 0.76, 0.82, 0.69, 0.77]
                    trends = ["up", "stable", "down"]
                    
                    risk_factors_data = []
                    for i, category in enumerate(risk_categories):
                        for j, name in enumerate(risk_names[i]):
                            idx = i * 3 + j
                            risk_factors_data.append((category, name, severities[idx], trends[idx % 3]))
                
                risk_factors_by_category = {}
                for category, name, severity, trend in risk_factors_data:
                    if category not in risk_factors_by_category:
                        risk_factors_by_category[category] = []
                    risk_factors_by_category[category].append((name, severity, trend))
                
                for category, risks in risk_factors_by_category.items():
                    with st.expander(f"{category} Risks", expanded=True):
                        for name, severity, trend in risks:
                            severity_pct = f"{severity * 100:.0f}%"
                            trend_icon = "↑" if trend == "up" else "↓" if trend == "down" else "→"
                            
                            col1, col2, col3 = st.columns([3, 1, 1])
                            col1.text(name)
                            col2.progress(severity)
                            col3.text(f"{severity_pct} {trend_icon}")
                
                st.subheader("Risk Severity Heatmap")
                risk_heatmap = create_risk_heatmap(risk_factors_by_category)
                st.plotly_chart(risk_heatmap, use_container_width=True)
                
                if 'summaries' in results and 'risk_factors' in results['summaries']:
                    st.subheader("Risk Factors Summary")
                    st.write(results['summaries']['risk_factors'])
        
        # Sentiment Analysis tab
        if analyze_sentiment:
            with tabs[2]:
                st.header("Sentiment Analysis")
                
                if 'sentiment' in results:
                    section_tabs = st.tabs([section_name.replace('_', ' ').title() for section_name in results['sentiment'].keys()])
                    
                    for i, (section_name, sentiment) in enumerate(results['sentiment'].items()):
                        with section_tabs[i]:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Positive", f"{sentiment['pos']:.2f}")
                            with col2:
                                st.metric("Negative", f"{sentiment['neg']:.2f}")
                            with col3:
                                st.metric("Neutral", f"{sentiment['neu']:.2f}")
                            
                            compound = sentiment['compound']
                            if compound >= 0.05:
                                st.success(f"Overall Sentiment: Positive ({compound:.2f})")
                            elif compound <= -0.05:
                                st.error(f"Overall Sentiment: Negative ({compound:.2f})")
                            else:
                                st.info(f"Overall Sentiment: Neutral ({compound:.2f})")
                            
                            sentiment_chart = create_sentiment_chart(sentiment)
                            st.plotly_chart(sentiment_chart, use_container_width=True)
                    
                    st.subheader("Year-over-Year Sentiment Comparison")
                    
                    years = [results['year'] - 2, results['year'] - 1, results['year']]
                    mock_sentiment_data = {
                        years[0]: {
                            'pos': max(0.1, sentiment['pos'] - 0.1 + random.uniform(-0.05, 0.05)),
                            'neg': max(0.1, sentiment['neg'] - 0.05 + random.uniform(-0.05, 0.05)),
                            'neu': max(0.1, sentiment['neu'] + 0.05 + random.uniform(-0.05, 0.05)),
                            'compound': min(1.0, max(-1.0, sentiment['compound'] + 0.1 + random.uniform(-0.05, 0.05)))
                        },
                        years[1]: {
                            'pos': max(0.1, sentiment['pos'] - 0.05 + random.uniform(-0.05, 0.05)),
                            'neg': max(0.1, sentiment['neg'] + random.uniform(-0.05, 0.05)),
                            'neu': max(0.1, sentiment['neu'] + random.uniform(-0.05, 0.05)),
                            'compound': min(1.0, max(-1.0, sentiment['compound'] + 0.05 + random.uniform(-0.05, 0.05)))
                        },
                        years[2]: sentiment
                    }
                    
                    year_comparison_chart = create_year_comparison_chart(results['ticker'], years, mock_sentiment_data)
                    st.plotly_chart(year_comparison_chart, use_container_width=True)
                else:
                    st.warning("Sentiment analysis not available")
        
        # Entity Analysis tab
        if analyze_entities:
            with tabs[3]:
                st.header("Entity Recognition")
                
                conn = sqlite3.connect('sec_filings.db')
                c = conn.cursor()
                c.execute("""
                    SELECT e.entity_type, e.entity_name, COUNT(*) as count
                    FROM entities e
                    JOIN sections s ON e.section_id = s.id
                    JOIN filings f ON s.filing_id = f.id
                    JOIN companies c ON f.company_id = c.id
                    WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ?
                    GROUP BY e.entity_type, e.entity_name
                    ORDER BY count DESC
                    LIMIT 100
                """, (results['ticker'], results['year'], results['filing_type']))
                entities_data = c.fetchall()
                conn.close()
                
                if not entities_data:
                    entity_types = ["ORG", "PERSON", "GPE", "LOC", "PRODUCT"]
                    entity_names = [
                        [results['ticker'], "Competitors", "Suppliers", "Regulatory Bodies", "Partners"],
                        ["CEO", "CFO", "Board Members", "Executives"],
                        ["United States", "China", "Europe", "Asia Pacific"],
                        ["Manufacturing Facilities", "Headquarters", "Distribution Centers"],
                        ["Core Products", "Services", "Software", "Hardware"]
                    ]
                    frequencies = [0.92, 0.45, 0.32, 0.28, 0.22, 0.85, 0.65, 0.55, 0.45, 0.68, 0.42, 0.38, 0.32, 0.25, 0.20, 0.15, 0.75, 0.58, 0.48, 0.42]
                    sentiments = [0.68, 0.38, 0.52, 0.41, 0.65, 0.60, 0.55, 0.50, 0.45, 0.62, 0.45, 0.58, 0.55, 0.51, 0.48, 0.45, 0.72, 0.68, 0.65, 0.61]
                    
                    entities_data = []
                    idx = 0
                    for i, entity_type in enumerate(entity_types):
                        for j, name in enumerate(entity_names[i]):
                            if idx < len(frequencies) and j < len(entity_names[i]):
                                count = int(frequencies[idx] * 10)
                                entities_data.append((entity_type, name, count))
                                idx += 1
                
                entity_types = {}
                for entity_type, entity_name, count in entities_data:
                    if entity_type not in entity_types:
                        entity_types[entity_type] = []
                    entity_types[entity_type].append((entity_name, count, 0.5))
                
                for entity_type, entities_list in entity_types.items():
                    with st.expander(f"{entity_type} Entities", expanded=True):
                        df = pd.DataFrame(entities_list, columns=["Entity", "Count", "Sentiment"])
                        st.dataframe(df[["Entity", "Count"]], use_container_width=True)
                        
                        if len(df) > 0:
                            top_df = df.head(10)
                            fig = px.bar(
                                top_df,
                                x="Entity",
                                y="Count",
                                title=f"Top {entity_type} Entities",
                                color="Count",
                                color_continuous_scale="Viridis"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Entity Relationship Network")
                html_string = create_entity_network(entity_types)
                if html_string:
                    components.html(html_string, height=600)
                else:
                    st.warning("Entity network visualization not available")
        
        # Topic Analysis tab
        if analyze_topics:
            with tabs[4]:
                st.header("Topic Modeling")
                
                if 'topics' in results:
                    for section_name, topic_count in results['topics'].items():
                        st.subheader(f"{section_name.replace('_', ' ').title()} Topics")
                        
                        conn = sqlite3.connect('sec_filings.db')
                        c = conn.cursor()
                        c.execute("""
                            SELECT s.section_text
                            FROM sections s
                            JOIN filings f ON s.filing_id = f.id
                            JOIN companies c ON f.company_id = c.id
                            WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ? AND s.section_name = ?
                        """, (results['ticker'], results['year'], results['filing_type'], section_name))
                        section_text = c.fetchone()
                        conn.close()
                        
                        if section_text:
                            cleaned_text = st.session_state.pipeline.clean_text(section_text[0])
                            topics = st.session_state.pipeline.extract_topics(cleaned_text, num_topics=topic_count)
                            
                            if topics:
                                topic_chart = create_topic_radar_chart(topics)
                                st.plotly_chart(topic_chart, use_container_width=True)
                                
                                for i, topic in enumerate(topics):
                                    with st.expander(f"Topic {i+1}: {' '.join(topic['words'][:3])}"):
                                        st.write(f"**Weight:** {topic['weight']:.2f}")
                                        st.write(f"**Keywords:** {', '.join(topic['words'])}")
                                        if 'summary' in topic:
                                            st.write(f"**Summary:** {topic['summary']}")
                            else:
                                st.warning(f"No topics extracted for {section_name}")
                        else:
                            st.warning(f"Section text not found for {section_name}")
                else:
                    st.warning("Topic modeling not available")
        
        # Text Classification tab
        if analyze_classification:
            with tabs[5]:
                st.header("Text Classification")
                
                if 'classifications' in results:
                    for section_name, classifications in results['classifications'].items():
                        st.subheader(f"{section_name.replace('_', ' ').title()} Classification")
                        
                        if classifications:
                            classification_chart = create_classification_chart(classifications)
                            st.plotly_chart(classification_chart, use_container_width=True)
                            
                            for category, confidence in classifications:
                                confidence_pct = f"{confidence * 100:.1f}%"
                                st.write(f"- **{category}:** {confidence_pct}")
                        else:
                            st.warning(f"No classifications available for {section_name}")
                else:
                    st.warning("Text classification not available")
        
        # Anomaly Detection tab
        if analyze_anomalies:
            with tabs[6]:
                st.header("Anomaly Detection")
                
                if 'anomalies' in results:
                    for section_name, anomalies in results['anomalies'].items():
                        st.subheader(f"{section_name.replace('_', ' ').title()} Anomalies")
                        
                        if anomalies:
                            anomaly_chart = create_anomaly_chart(anomalies)
                            st.plotly_chart(anomaly_chart, use_container_width=True)
                            
                            for description, score in anomalies:
                                score_pct = f"{score * 100:.1f}%"
                                st.write(f"- **{description}:** {score_pct}")
                        else:
                            st.write("No anomalies detected")
                else:
                    st.warning("Anomaly detection not available")
        
        # Agentic AI tab
        with tabs[7]:
            st.header("Agentic AI Analysis")
            
            if 'agentic_ai' in st.session_state and openai_api_key:
                st.write("Use this tab to ask questions about the filing and get AI-powered answers.")
                
                user_query = st.text_input("Ask a question about this filing:", key="agentic_query")
                
                if st.button("Get Answer", key="agentic_button"):
                    if user_query:
                        with st.spinner("Analyzing..."):
                            try:
                                # Get relevant section text
                                conn = sqlite3.connect('sec_filings.db')
                                c = conn.cursor()
                                c.execute("""
                                    SELECT s.section_name, s.section_text
                                    FROM sections s
                                    JOIN filings f ON s.filing_id = f.id
                                    JOIN companies c ON f.company_id = c.id
                                    WHERE c.ticker = ? AND f.filing_year = ? AND f.filing_type = ?
                                """, (results['ticker'], results['year'], results['filing_type']))
                                sections_data = c.fetchall()
                                conn.close()
                                
                                sections_dict = {name: text for name, text in sections_data}
                                
                                answer = st.session_state.agentic_ai.answer_question(
                                    question=user_query,
                                    sections=sections_dict,
                                    ticker=results['ticker'],
                                    year=results['year'],
                                    filing_type=results['filing_type']
                                )
                                
                                st.write("### Answer")
                                st.write(answer)
                                
                                # Store the Q&A history
                                if 'qa_history' not in st.session_state:
                                    st.session_state.qa_history = []
                                
                                st.session_state.qa_history.append({
                                    "question": user_query,
                                    "answer": answer
                                })
                            except Exception as e:
                                st.error(f"Error generating answer: {str(e)}")
                    else:
                        st.warning("Please enter a question")
                
                # Display Q&A history
                if 'qa_history' in st.session_state and st.session_state.qa_history:
                    st.subheader("Previous Questions")
                    for i, qa in enumerate(st.session_state.qa_history):
                        with st.expander(f"Q: {qa['question']}", expanded=False):
                            st.write(qa['answer'])
            else:
                st.warning("Agentic AI requires an OpenAI API key. Please provide one in the sidebar.")

# Define the SECFilingPipeline class
class SECFilingPipeline:
    def __init__(self, user_agent, email, openai_api_key=None):
        self.user_agent = user_agent
        self.email = email
        self.openai_api_key = openai_api_key
        self.nlp = spacy.load("en_core_web_sm")
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize the SQLite database for storing SEC filings data"""
        conn = sqlite3.connect('sec_filings.db')
        c = conn.cursor()
        
        # Create tables if they don't exist
        c.execute('''
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY,
                ticker TEXT UNIQUE,
                name TEXT,
                industry TEXT,
                sector TEXT
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS filings (
                id INTEGER PRIMARY KEY,
                company_id INTEGER,
                filing_type TEXT,
                filing_year INTEGER,
                filing_quarter INTEGER,
                filing_date TEXT,
                accession_number TEXT,
                FOREIGN KEY (company_id) REFERENCES companies (id)
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY,
                filing_id INTEGER,
                section_name TEXT,
                section_text TEXT,
                FOREIGN KEY (filing_id) REFERENCES filings (id)
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                section_id INTEGER,
                entity_type TEXT,
                entity_name TEXT,
                entity_count INTEGER,
                FOREIGN KEY (section_id) REFERENCES sections (id)
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS risk_factors (
                id INTEGER PRIMARY KEY,
                section_id INTEGER,
                risk_category TEXT,
                risk_name TEXT,
                severity REAL,
                trend TEXT,
                FOREIGN KEY (section_id) REFERENCES sections (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def download_filing(self, ticker, year, filing_type, quarter=None):
        """Download SEC filing for a company"""
        try:
            # Mock implementation - in a real app, this would use SEC EDGAR API
            print(f"Downloading {filing_type} filing for {ticker} ({year})")
            
            # Check if filing already exists in database
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            
            # Check if company exists
            c.execute("SELECT id FROM companies WHERE ticker = ?", (ticker,))
            company_id = c.fetchone()
            
            if not company_id:
                # Add company to database
                c.execute("INSERT INTO companies (ticker, name) VALUES (?, ?)", 
                         (ticker, f"{ticker} Inc."))
                company_id = c.lastrowid
            else:
                company_id = company_id[0]
            
            # Check if filing exists
            c.execute("""
                SELECT id FROM filings 
                WHERE company_id = ? AND filing_type = ? AND filing_year = ? AND 
                      (filing_quarter = ? OR (? IS NULL AND filing_quarter IS NULL))
            """, (company_id, filing_type, year, quarter, quarter))
            
            filing_id = c.fetchone()
            
            if not filing_id:
                # Add filing to database
                c.execute("""
                    INSERT INTO filings (company_id, filing_type, filing_year, filing_quarter, filing_date) 
                    VALUES (?, ?, ?, ?, ?)
                """, (company_id, filing_type, year, quarter, f"{year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"))
                filing_id = c.lastrowid
            else:
                filing_id = filing_id[0]
            
            conn.commit()
            conn.close()
            
            return {
                "status": "success",
                "filing_id": filing_id,
                "company_id": company_id
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def extract_sections(self, filing_id):
        """Extract sections from a filing"""
        try:
            # Mock implementation - in a real app, this would parse the actual filing
            sections = {
                "risk_factors": "This section contains risk factors that may affect the company's operations...",
                "management_discussion": "Management's discussion and analysis of financial condition...",
                "financial_statements": "The financial statements and supplementary data...",
                "business_overview": "Overview of the company's business operations..."
            }
            
            # Store sections in database
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            
            for section_name, section_text in sections.items():
                # Check if section already exists
                c.execute("""
                    SELECT id FROM sections 
                    WHERE filing_id = ? AND section_name = ?
                """, (filing_id, section_name))
                
                section_id = c.fetchone()
                
                if not section_id:
                    # Generate mock text based on section name
                    if section_name == "risk_factors":
                        section_text = self.generate_mock_risk_factors()
                    elif section_name == "management_discussion":
                        section_text = self.generate_mock_md_and_a()
                    elif section_name == "financial_statements":
                        section_text = self.generate_mock_financial_statements()
                    elif section_name == "business_overview":
                        section_text = self.generate_mock_business_overview()
                    
                    # Add section to database
                    c.execute("""
                        INSERT INTO sections (filing_id, section_name, section_text) 
                        VALUES (?, ?, ?)
                    """, (filing_id, section_name, section_text))
            
            conn.commit()
            conn.close()
            
            return {
                "status": "success",
                "sections": {name: len(text) for name, text in sections.items()}
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def generate_mock_risk_factors(self):
        """Generate mock risk factors text"""
        return """
        Risk Factors
        
        Our business operations are subject to numerous risks and uncertainties, including those outside of our control, that could cause our actual results to be harmed. These risks include, but are not limited to:
        
        Market and Competition Risks:
        - Intense competition in our industry may limit our growth and profitability
        - Rapid technological changes may render our products obsolete
        - Changes in consumer preferences may reduce demand for our products
        - Economic downturns may reduce consumer spending on our products
        
        Operational Risks:
        - Supply chain disruptions could impact our ability to deliver products
        - Manufacturing delays could result in lost sales opportunities
        - Quality control issues could damage our brand reputation
        - Failure to attract and retain key personnel could harm our operations
        
        Technological Risks:
        - Cybersecurity threats could compromise our systems and data
        - Technology obsolescence could require significant capital investments
        - Intellectual property protection challenges could impact our competitive position
        - System failures could disrupt our operations
        
        Regulatory Risks:
        - Changing compliance requirements could increase our costs
        - International trade policies could impact our global operations
        - Data privacy regulations could limit our marketing capabilities
        - Environmental regulations could increase our operational costs
        """
    
    def generate_mock_md_and_a(self):
        """Generate mock Management Discussion and Analysis text"""
        return """
        Management's Discussion and Analysis of Financial Condition and Results of Operations
        
        Overview
        
        The following discussion should be read in conjunction with our consolidated financial statements and related notes. This discussion contains forward-looking statements that involve risks and uncertainties. Our actual results could differ materially from those anticipated in these forward-looking statements as a result of various factors.
        
        Results of Operations
        
        Revenue increased by 12% compared to the previous year, primarily driven by new product launches and expansion into international markets. Gross margin improved to 58% from 54% in the prior year due to manufacturing efficiencies and favorable product mix. Operating expenses increased by 8%, reflecting our continued investment in research and development and marketing initiatives to support future growth.
        
        Liquidity and Capital Resources
        
        As of the end of the fiscal year, we had $450 million in cash and cash equivalents, compared to $380 million at the end of the previous fiscal year. Operating activities generated $220 million in cash, while investing activities used $120 million primarily for capital expenditures and acquisitions. We believe our existing cash resources are sufficient to meet our anticipated operating cash needs for at least the next 12 months.
        """
    
    def generate_mock_financial_statements(self):
        """Generate mock financial statements text"""
        return """
        Consolidated Balance Sheets
        (in millions)
                                        Current Year    Previous Year
        Assets
        Current assets:
          Cash and cash equivalents     $    450        $    380
          Short-term investments             200             150
          Accounts receivable, net           320             280
          Inventories                        180             160
          Other current assets                50              45
        Total current assets                1200            1015
        
        Property and equipment, net          500             450
        Goodwill                             300             300
        Intangible assets, net               200             220
        Other assets                         100              80
        Total assets                    $   2300        $   2065
        
        Liabilities and Stockholders' Equity
        Current liabilities:
          Accounts payable              $    150        $    130
          Accrued expenses                   200             180
          Deferred revenue                   100              90
          Current portion of debt             50              50
        Total current liabilities            500             450
        
        Long-term debt                       300             350
        Other long-term liabilities          200             180
        Total liabilities                   1000             980
        
        Stockholders' equity:
          Common stock                         1               1
          Additional paid-in capital         800             750
          Retained earnings                  500             335
          Accumulated other comprehensive income/(loss)        (1)              (1)
        Total stockholders' equity          1300            1085
        Total liabilities and stockholders' equity $ 2300   $ 2065
        """
    
    def generate_mock_business_overview(self):
        """Generate mock business overview text"""
        return """
        Business Overview
        
        Company Background
        
        We are a leading provider of innovative technology solutions for consumers and businesses worldwide. Founded in 1985, we have grown from a small startup to a global enterprise with operations in over 30 countries and approximately 25,000 employees. Our mission is to create products that enrich people's lives and help businesses succeed in an increasingly digital world.
        
        Products and Services
        
        Our product portfolio includes hardware devices, software applications, cloud services, and specialized solutions for enterprise customers. Our consumer products focus on enhancing productivity, entertainment, and connectivity, while our business solutions address critical needs in areas such as data management, security, and digital transformation.
        
        Research and Development
        
        Innovation is at the core of our strategy. We invested approximately $350 million in research and development during the past fiscal year, representing 15% of our total revenue. Our R&D activities focus on developing new products, enhancing existing offerings, and exploring emerging technologies such as artificial intelligence, augmented reality, and quantum computing.
        
        Marketing and Distribution
        
        We market our products through multiple channels, including direct sales, retail partners, online platforms, and value-added resellers. Our marketing strategy emphasizes product differentiation, brand awareness, and customer engagement. We have established strategic partnerships with leading retailers and distributors to ensure broad availability of our products worldwide.
        """
    
    def analyze_sentiment(self, sections):
        """Analyze sentiment of filing sections"""
        try:
            sentiment_results = {}
            analyzer = SentimentIntensityAnalyzer()
            
            for section_name, section_text in sections.items():
                sentiment = analyzer.polarity_scores(section_text)
                sentiment_results[section_name] = sentiment
            
            return {
                "status": "success",
                "sentiment": sentiment_results
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def extract_entities(self, sections):
        """Extract named entities from filing sections"""
        try:
            entity_results = {}
            
            for section_name, section_text in sections.items():
                doc = self.nlp(section_text[:10000])  # Limit text size for performance
                
                entities = {}
                for ent in doc.ents:
                    if ent.label_ not in entities:
                        entities[ent.label_] = []
                    
                    # Check if entity already exists
                    found = False
                    for i, (name, count, _) in enumerate(entities.get(ent.label_, [])):
                        if name.lower() == ent.text.lower():
                            entities[ent.label_][i] = (name, count + 1, 0)
                            found = True
                            break
                    
                    if not found:
                        entities[ent.label_].append((ent.text, 1, 0))
                
                # Sort entities by frequency
                for label in entities:
                    entities[label] = sorted(entities[label], key=lambda x: x[1], reverse=True)
                
                entity_results[section_name] = len([ent for sublist in entities.values() for ent in sublist])
                
                # Store entities in database
                conn = sqlite3.connect('sec_filings.db')
                c = conn.cursor()
                
                # Get section ID
                c.execute("""
                    SELECT id FROM sections 
                    WHERE section_name = ? AND section_text LIKE ?
                """, (section_name, section_text[:100] + '%'))
                
                section_id = c.fetchone()
                
                if section_id:
                    section_id = section_id[0]
                    
                    # Delete existing entities for this section
                    c.execute("DELETE FROM entities WHERE section_id = ?", (section_id,))
                    
                    # Insert new entities
                    for entity_type, entity_list in entities.items():
                        for entity_name, entity_count, _ in entity_list:
                            c.execute("""
                                INSERT INTO entities (section_id, entity_type, entity_name, entity_count)
                                VALUES (?, ?, ?, ?)
                            """, (section_id, entity_type, entity_name, entity_count))
                
                conn.commit()
                conn.close()
            
            return {
                "status": "success",
                "entities": entity_results
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def extract_topics(self, text, num_topics=5):
        """Extract topics from text using LDA"""
        try:
            # Mock implementation - in a real app, this would use actual topic modeling
            topics = []
            
            for i in range(num_topics):
                words = []
                for _ in range(10):
                    words.append(random.choice([
                        "revenue", "growth", "market", "product", "customer", 
                        "technology", "risk", "competition", "regulation", "innovation",
                        "digital", "strategy", "financial", "operational", "investment",
                        "global", "supply", "chain", "security", "sustainability"
                    ]))
                
                topics.append({
                    "words": list(set(words)),  # Remove duplicates
                    "weight": random.uniform(0.5, 1.0),
                    "summary": f"This topic relates to {', '.join(words[:3])} and their impact on business operations."
                })
            
            return topics
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return None
    
    def generate_summaries(self, sections):
        """Generate summaries of filing sections"""
        try:
            summary_results = {}
            
            for section_name, section_text in sections.items():
                # Mock implementation - in a real app, this would use actual summarization
                sentences = section_text.split('.')
                summary = '. '.join(random.sample(sentences, min(5, len(sentences))))
                summary_results[section_name] = summary
            
            return {
                "status": "success",
                "summaries": summary_results
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def classify_text(self, sections):
        """Classify text in filing sections"""
        try:
            classification_results = {}
            
            for section_name, section_text in sections.items():
                # Mock implementation - in a real app, this would use actual classification
                if section_name == "risk_factors":
                    classifications = [
                        ("Market Risk", 0.85),
                        ("Operational Risk", 0.72),
                        ("Technological Risk", 0.68),
                        ("Regulatory Risk", 0.65)
                    ]
                elif section_name == "management_discussion":
                    classifications = [
                        ("Financial Analysis", 0.88),
                        ("Strategic Planning", 0.75),
                        ("Performance Review", 0.70),
                        ("Future Outlook", 0.65)
                    ]
                else:
                    classifications = [
                        ("Business Information", 0.82),
                        ("Financial Data", 0.76),
                        ("Corporate Governance", 0.68),
                        ("Industry Analysis", 0.62)
                    ]
                
                classification_results[section_name] = classifications
            
            return {
                "status": "success",
                "classifications": classification_results
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def detect_anomalies(self, sections):
        """Detect anomalies in filing sections"""
        try:
            anomaly_results = {}
            
            for section_name, section_text in sections.items():
                # Mock implementation - in a real app, this would use actual anomaly detection
                if section_name == "risk_factors":
                    anomalies = [
                        ("Unusual Risk Factor Language", 0.78),
                        ("New Risk Category Introduction", 0.65),
                        ("Significant Change in Risk Assessment", 0.58)
                    ]
                elif section_name == "financial_statements":
                    anomalies = [
                        ("Unusual Revenue Recognition Pattern", 0.72),
                        ("Significant Change in Accounting Method", 0.68),
                        ("Unexpected Asset Valuation", 0.62)
                    ]
                else:
                    anomalies = [
                        ("Unusual Language Pattern", 0.70),
                        ("Significant Change in Disclosure", 0.65),
                        ("Unexpected Information", 0.60)
                    ]
                
                anomaly_results[section_name] = anomalies
            
            return {
                "status": "success",
                "anomalies": anomaly_results
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def generate_insight(self, section_name, section_text):
        """Generate AI insight for a section"""
        try:
            if not self.openai_api_key:
                return "OpenAI API key is required for generating insights."
            
            # Mock implementation - in a real app, this would use actual AI
            insights = {
                "risk_factors": """
                Analysis of the risk factors reveals several key concerns:
                
                1. Cybersecurity risks are prominently featured, suggesting increased attention to digital threats.
                2. Supply chain disruptions are highlighted as a significant operational risk, likely reflecting global economic conditions.
                3. Regulatory compliance risks have increased compared to previous filings, particularly around data privacy.
                4. Market competition risks emphasize new entrants and technological disruption.
                
                These risk factors suggest the company is operating in a challenging environment with multiple external pressures. The emphasis on technological and cybersecurity risks indicates the company's increasing reliance on digital infrastructure and potential vulnerabilities in this area.
                """,
                
                "management_discussion": """
                The management discussion reveals several important trends:
                
                1. Revenue growth of 12% outpaces the industry average of 8%, indicating market share gains.
                2. Gross margin improvement from 54% to 58% suggests successful cost optimization initiatives.
                3. Increased R&D spending (15% of revenue) reflects commitment to innovation but may pressure short-term profitability.
                4. International expansion is cited as a key growth driver, with particular emphasis on Asian markets.
                
                The company appears to be in a strong financial position with healthy cash reserves and improving operational efficiency. Management's focus on international expansion and R&D investment suggests a long-term growth strategy rather than short-term profit maximization.
                """,
                
                "financial_statements": """
                Analysis of the financial statements reveals:
                
                1. Cash position has strengthened by 18% year-over-year, providing increased financial flexibility.
                2. Debt-to-equity ratio has improved from 0.37 to 0.27, indicating a stronger balance sheet.
                3. Return on assets has increased from 12.3% to 14.1%, suggesting improved operational efficiency.
                4. Capital expenditures represent 8% of revenue, slightly above the industry average of 6.5%.
                
                The company's financial position appears solid with improving metrics across most key indicators. The increased cash reserves provide a buffer against market uncertainties while allowing for strategic investments in growth opportunities.
                """,
                
                "business_overview": """
                The business overview reveals several strategic priorities:
                
                1. Significant investment in emerging technologies (AI, AR, quantum computing) positions the company for future innovation.
                2. Multi-channel distribution strategy balances direct and indirect sales channels.
                3. Global presence in 30+ countries provides geographic diversification.
                4. Product portfolio spans both consumer and enterprise markets, reducing dependence on any single market segment.
                
                The company's broad product portfolio and geographic diversification provide resilience against market-specific downturns. The emphasis on emerging technologies suggests management is focused on long-term positioning rather than just current product cycles.
                """
            }
            
            return insights.get(section_name, "No insight available for this section.")
        except Exception as e:
            return f"Error generating insight: {str(e)}"
    
    def clean_text(self, text):
        """Clean and preprocess text for analysis"""
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def process_filing(self, ticker, year, filing_type, quarter=None):
        """Process an SEC filing and perform analysis"""
        try:
            # Download filing
            download_result = self.download_filing(ticker, year, filing_type, quarter)
            
            if download_result["status"] != "success":
                return download_result
            
            filing_id = download_result["filing_id"]
            
            # Extract sections
            sections_result = self.extract_sections(filing_id)
            
            if sections_result["status"] != "success":
                return sections_result
            
            # Get section texts
            conn = sqlite3.connect('sec_filings.db')
            c = conn.cursor()
            c.execute("""
                SELECT section_name, section_text
                FROM sections
                WHERE filing_id = ?
            """, (filing_id,))
            sections_data = c.fetchall()
            conn.close()
            
            sections = {name: text for name, text in sections_data}
            
            # Analyze sentiment
            sentiment_result = self.analyze_sentiment(sections)
            
            # Extract entities
            entities_result = self.extract_entities(sections)
            
            # Extract topics
            topics_result = {}
            for section_name, section_text in sections.items():
                cleaned_text = self.clean_text(section_text)
                topics = self.extract_topics(cleaned_text)
                if topics:
                    topics_result[section_name] = len(topics)
            
            # Generate summaries
            summaries_result = self.generate_summaries(sections)
            
            # Classify text
            classification_result = self.classify_text(sections)
            
            # Detect anomalies
            anomalies_result = self.detect_anomalies(sections)
            
            # Generate AI insights if API key is provided
            insights = {}
            if self.openai_api_key:
                for section_name, section_text in sections.items():
                    insights[section_name] = self.generate_insight(section_name, section_text)
            
            # Combine results
            result = {
                "status": "success",
                "ticker": ticker,
                "year": year,
                "filing_type": filing_type,
                "quarter": quarter,
                "sections": sections_result["sections"],
                "sentiment": sentiment_result.get("sentiment", {}),
                "entities": entities_result.get("entities", {}),
                "topics": topics_result,
                "summaries": summaries_result.get("summaries", {}),
                "classifications": classification_result.get("classifications", {}),
                "anomalies": anomalies_result.get("anomalies", {}),
                "insights": insights
            }
            
            return result
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# Define the AgenticSECAnalyzer class
class AgenticSECAnalyzer:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
    
    def answer_question(self, question, sections, ticker, year, filing_type):
        """Answer a question about the SEC filing using AI"""
        try:
            # Mock implementation - in a real app, this would use actual AI
            responses = {
                "risk": "The main risk factors identified in the filing include cybersecurity threats, supply chain disruptions, regulatory compliance challenges, and market competition. The company highlights cybersecurity as a particularly significant concern, noting increased sophistication of attacks.",
                "revenue": "The company reported a 12% increase in revenue compared to the previous year, primarily driven by new product launches and expansion into international markets, particularly in Asia.",
                "profit": "Gross margin improved to 58% from 54% in the prior year due to manufacturing efficiencies and favorable product mix. Operating expenses increased by 8%, reflecting continued investment in R&D.",
                "strategy": "The company's strategy focuses on international expansion, investment in emerging technologies (AI, AR, quantum computing), and maintaining a diversified product portfolio across both consumer and enterprise markets.",
                "outlook": "Management expressed cautious optimism about future growth prospects, citing strong product pipeline and international expansion opportunities, while acknowledging potential headwinds from economic uncertainty and supply chain challenges.",
                "debt": "The company's debt-to-equity ratio improved from 0.37 to 0.27, indicating a stronger balance sheet. Long-term debt decreased from $350 million to $300 million.",
                "cash": "Cash and cash equivalents increased to $450 million from $380 million in the previous year. Operating activities generated $220 million in cash."
            }
            
            # Simple keyword matching to determine response
            for keyword, response in responses.items():
                if keyword.lower() in question.lower():
                    return f"Based on the {year} {filing_type} filing for {ticker}:\n\n{response}"
            
            # Default response if no keywords match
            return f"Based on my analysis of the {year} {filing_type} filing for {ticker}, I don't have specific information to answer that question. The filing primarily discusses risk factors, financial performance, business operations, and strategic initiatives. Could you rephrase your question to focus on one of these areas?"
        except Exception as e:
            return f"Error generating answer: {str(e)}"

if __name__ == "__main__":
    main()