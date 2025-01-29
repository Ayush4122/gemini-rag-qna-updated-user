# app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import PyPDF2
import docx
from io import StringIO
import sqlite3
from datetime import datetime
import socket
import uuid
import pandas as pd
import plotly.express as px

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Database setup for user tracking
def init_database():
    conn = sqlite3.connect('user_tracking.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            ip_address TEXT,
            hostname TEXT,
            timestamp DATETIME,
            name TEXT,
            designation TEXT,
            purpose TEXT,
            input_method TEXT,
            questions_asked INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            question TEXT,
            timestamp DATETIME,
            FOREIGN KEY (session_id) REFERENCES user_visits(session_id)
        )
    ''')
    conn.commit()
    conn.close()

# Log user visit
def log_user_visit(name=None, designation=None, purpose=None, input_method=None):
    conn = sqlite3.connect('user_tracking.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Generate unique session ID if not already in session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Get IP and hostname
    try:
        ip_address = socket.gethostbyname(socket.gethostname())
        hostname = socket.gethostname()
    except Exception:
        ip_address = "Unknown"
        hostname = "Unknown"
    
    # Current timestamp
    timestamp = datetime.now()
    
    # Insert visit record
    cursor.execute('''
        INSERT INTO user_visits 
        (session_id, ip_address, hostname, timestamp, name, designation, purpose, input_method) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (st.session_state.session_id, ip_address, hostname, timestamp, name, designation, purpose, input_method))
    
    conn.commit()
    conn.close()
    
    return st.session_state.session_id

# Log user question
def log_question(question):
    conn = sqlite3.connect('user_tracking.db', check_same_thread=False)
    cursor = conn.cursor()
    
    timestamp = datetime.now()
    
    # Log the question
    cursor.execute('''
        INSERT INTO user_questions (session_id, question, timestamp)
        VALUES (?, ?, ?)
    ''', (st.session_state.session_id, question, timestamp))
    
    # Update questions asked count
    cursor.execute('''
        UPDATE user_visits 
        SET questions_asked = questions_asked + 1 
        WHERE session_id = ?
    ''', (st.session_state.session_id,))
    
    conn.commit()
    conn.close()

# Retrieve user visit statistics
def get_user_statistics():
    conn = sqlite3.connect('user_tracking.db', check_same_thread=False)
    
    # Total visits
    total_visits_df = pd.read_sql_query(
        'SELECT COUNT(*) as total_visits FROM user_visits', conn)
    
    # Unique visits
    unique_visits_df = pd.read_sql_query(
        'SELECT COUNT(DISTINCT session_id) as unique_visits FROM user_visits', conn)
    
    # Purpose distribution
    purpose_df = pd.read_sql_query('''
        SELECT purpose, COUNT(*) as count 
        FROM user_visits 
        WHERE purpose IS NOT NULL 
        GROUP BY purpose 
        ORDER BY count DESC
    ''', conn)
    
    # Designation distribution
    designation_df = pd.read_sql_query('''
        SELECT designation, COUNT(*) as count 
        FROM user_visits 
        WHERE designation IS NOT NULL 
        GROUP BY designation 
        ORDER BY count DESC
    ''', conn)
    
    # Input method distribution
    input_method_df = pd.read_sql_query('''
        SELECT input_method, COUNT(*) as count 
        FROM user_visits 
        WHERE input_method IS NOT NULL 
        GROUP BY input_method
    ''', conn)
    
    # Questions per session
    questions_df = pd.read_sql_query('''
        SELECT questions_asked, COUNT(*) as count 
        FROM user_visits 
        GROUP BY questions_asked
    ''', conn)
    
    # Usage over time
    usage_trend_df = pd.read_sql_query('''
        SELECT date(timestamp) as date, COUNT(*) as visits 
        FROM user_visits 
        GROUP BY date(timestamp) 
        ORDER BY date
    ''', conn)
    
    conn.close()
    
    return {
        'total_visits': total_visits_df.iloc[0]['total_visits'],
        'unique_visits': unique_visits_df.iloc[0]['unique_visits'],
        'purpose_df': purpose_df,
        'designation_df': designation_df,
        'input_method_df': input_method_df,
        'questions_df': questions_df,
        'usage_trend_df': usage_trend_df
    }

def fetch_website_content(url, max_pages=5):
    def get_links(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith(url)]
    
    visited = set()
    to_visit = [url]
    content = ""
    
    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url not in visited:
            try:
                response = requests.get(current_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                content += soup.get_text() + "\n\n"
                visited.add(current_url)
                to_visit.extend([link for link in get_links(current_url) if link not in visited])
            except Exception as e:
                st.error(f"Error fetching {current_url}: {str(e)}")
    
    return content

def read_pdf(file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
    return text

def read_docx(file):
    text = ""
    try:
        doc = docx.Document(file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
    return text

def read_txt(file):
    text = ""
    try:
        stringio = StringIO(file.getvalue().decode("utf-8"))
        text = stringio.read()
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
    return text

def create_vector_store(content):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_text(content)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    
    vector_store = FAISS.from_texts(splits, embeddings)
    return vector_store

def answer_question(vector_store, question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    
    return qa.run(question)

# Initialize database
init_database()

# Streamlit UI
st.title("Advanced Document Q&A Chatbot (Gemini)")

# User Information Collection (Sidebar)
st.sidebar.header("User Information")
user_name = st.sidebar.text_input("Your Name (Optional)")
user_designation = st.sidebar.text_input("Your Designation (Optional)")
user_purpose = st.sidebar.selectbox(
    "Purpose of Use", 
    ["", "Research", "Education", "Professional", "Personal", "Other"]
)

# Input method selection
input_method = st.radio(
    "Choose input method:",
    ("Website URL", "Upload Document")
)

# Log user visit when information is provided
if user_name or user_designation or user_purpose:
    session_id = log_user_visit(
        name=user_name, 
        designation=user_designation, 
        purpose=user_purpose,
        input_method=input_method
    )

content = ""

if input_method == "Website URL":
    url = st.text_input("Enter website URL:")
    if url:
        with st.spinner("Fetching website content..."):
            content = fetch_website_content(url)

else:
    uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'docx', 'txt'])
    if uploaded_file:
        with st.spinner("Processing document..."):
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                content = read_pdf(uploaded_file)
            elif file_extension == 'docx':
                content = read_docx(uploaded_file)
            elif file_extension == 'txt':
                content = read_txt(uploaded_file)

question = st.text_input("Ask a question about the content:")

if content and question:
    # Log the question
    log_question(question)
    
    with st.spinner("Creating vector store..."):
        vector_store = create_vector_store(content)
    
    with st.spinner("Analyzing and answering..."):
        answer = answer_question(vector_store, question)
    
    st.write("Answer:", answer)

    # Chat History
    st.markdown("---")
    st.write("Chat History:")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append((question, answer))
    
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.write(f"Q{i}: {q}")
        st.write(f"A{i}: {a}")

# User Statistics
st.sidebar.markdown("---")
st.sidebar.header("Application Usage Statistics")
if st.sidebar.button("Show Statistics"):
    stats = get_user_statistics()
    
    # Basic Stats
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Total Visits", stats['total_visits'])
    col2.metric("Unique Users", stats['unique_visits'])
    
    # Purpose Distribution
    if not stats['purpose_df'].empty:
        st.sidebar.subheader("Purpose Distribution")
        fig_purpose = px.pie(stats['purpose_df'], values='count', names='purpose', 
                           title='Usage by Purpose')
        st.sidebar.plotly_chart(fig_purpose, use_container_width=True)
    
    # Designation Distribution
    if not stats['designation_df'].empty:
        st.sidebar.subheader("User Designations")
        fig_designation = px.bar(stats['designation_df'], x='designation', y='count',
                               title='Users by Designation')
        st.sidebar.plotly_chart(fig_designation, use_container_width=True)
    
    # Input Method Distribution
    if not stats['input_method_df'].empty:
        st.sidebar.subheader("Input Methods Used")
        fig_input = px.pie(stats['input_method_df'], values='count', names='input_method',
                          title='Input Method Distribution')
        st.sidebar.plotly_chart(fig_input, use_container_width=True)
    
    # Questions per Session
    if not stats['questions_df'].empty:
        st.sidebar.subheader("Questions per Session")
        fig_questions = px.bar(stats['questions_df'], x='questions_asked', y='count',
                             title='Questions Asked per Session')
        st.sidebar.plotly_chart(fig_questions, use_container_width=True)
    
    # Usage Trend
    if not stats['usage_trend_df'].empty:
        st.sidebar.subheader("Usage Trend")
        fig_trend = px.line(stats['usage_trend_df'], x='date', y='visits',
                           title='Daily Usage Trend')
        st.sidebar.plotly_chart(fig_trend, use_container_width=True)
