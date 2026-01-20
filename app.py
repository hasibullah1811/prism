import streamlit as st
import json
import tiktoken
import textwrap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Prism", 
    page_icon="💎", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* GLOBAL FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
    }

    /* --- 1. HIDE HEADER & FOOTER --- */
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stToolbar"] { display: none; }
    footer { display: none; }
    header { background-color: transparent !important; }

    /* --- 2. LOCK SIDEBAR OPEN --- */
    
    /* Hide the button that collapses the sidebar */
    button[data-testid="baseButton-header"] {
        display: none !important;
    }
    
    /* Hide the "Open Sidebar" arrow (so it looks cleaner) */
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }
    
    /* Force sidebar width and visibility */
    section[data-testid="stSidebar"] {
        display: block !important;
        width: 336px !important;
    }
    
    /* --- 3. UI STYLING --- */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Clean Cards */
    .chunk-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
    }
    
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 12px;
        border-bottom: 1px solid #f1f5f9;
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.05em;
    }
    
    .chunk-text {
        font-size: 15px;
        line-height: 1.6;
        color: #334155;
    }
    
    .highlight-overlap {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 2px 4px;
        border-radius: 4px;
        border-bottom: 2px solid #fde047;
    }
    
    .match-card {
        border-left: 4px solid #10b981;
        background-color: #f0fdf4;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 8px;
        border-radius: 99px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 6px;
    }
    
    .badge-gray { background: #f1f5f9; color: #64748b; }
    .badge-green { background: #dcfce7; color: #15803d; }
    .badge-red { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def get_overlap_text(prev_chunk, current_chunk):
    if not prev_chunk: return ""
    check_len = min(len(prev_chunk), len(current_chunk))
    for i in range(check_len, 0, -1):
        suffix = prev_chunk[-i:]
        prefix = current_chunk[:i]
        if suffix == prefix: return suffix
    return ""

def calculate_similarity(query, chunks):
    if not query or not chunks: return [0.0] * len(chunks)
    vectorizer = TfidfVectorizer(stop_words='english').fit_transform([query] + chunks)
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors[0:1], vectors[1:]).flatten()

def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("## 💎 Prism")
    st.markdown("Visual debugger for RAG pipelines.")
    st.markdown("---")
    
    st.markdown("### ⚙️ Configuration")
    chunk_size = st.slider("Chunk Size", 50, 2000, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 50, 10)
    
    if chunk_overlap >= chunk_size:
        st.warning("⚠️ Overlap should be smaller than size.")
        
    st.markdown("---")
    st.markdown("### 📊 Live Stats")
    stats_placeholder = st.empty() # Placeholder for later updates

# --- 5. MAIN AREA ---
# Input Section
st.markdown("### Source Document")
text_input = st.text_area(
    "Input Text", 
    height=150, 
    placeholder="Paste your raw text or article here...", 
    label_visibility="collapsed"
)

# Default Demo Text
if not text_input:
    text_input = """RAG (Retrieval-Augmented Generation) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.

Building a RAG pipeline usually involves two main steps: retrieval and generation. In the retrieval step, the system searches for relevant documents in a knowledge base. In the generation step, the LLM uses the retrieved context to answer the user's question.

However, splitting the text correctly is crucial. If you split the text in the middle of a sentence, the model might lose the semantic meaning. This is why tools like Prism are necessary to visualize the process."""

# Search Section
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### Simulation")
    query = st.text_input("Search Query", placeholder="Enter a question to test retrieval...", label_visibility="collapsed")
with col2:
    st.markdown("### Export")
    # Button will be rendered later with data

# --- 6. LOGIC & RENDER ---
if text_input:
    # Processing
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    chunks = splitter.split_text(text_input)
    scores = calculate_similarity(query, chunks)
    total_tokens = count_tokens(text_input)

    # Sidebar Stats Update
    stats_placeholder.markdown(f"""
    - **Chunks:** {len(chunks)}
    - **Tokens:** {total_tokens}
    - **Cost:** `${total_tokens * 0.0000001:.7f}`
    """)
    
    # Export Button Logic
    chunk_data = [{"id": i, "content": c, "tokens": count_tokens(c)} for i, c in enumerate(chunks)]
    with col2:
        st.download_button("📥 JSON", data=json.dumps(chunk_data), file_name="prism-export.json", mime="application/json")

    # Visualization
    st.markdown("---")
    
    previous_chunk = None
    
    for i, chunk in enumerate(chunks):
        # 1. Calculate Overlap
        overlap_text = ""
        non_overlap_text = chunk
        if previous_chunk:
            overlap = get_overlap_text(previous_chunk, chunk)
            if overlap:
                overlap_text = overlap
                non_overlap_text = chunk[len(overlap):]
        
        # 2. Logic & Scores
        score = scores[i]
        is_match = score > 0.2
        tokens = count_tokens(chunk)
        
        # 3. Badge Assembly
        badges = []
        # Token Badge
        badges.append(f'<span class="badge badge-gray">{tokens} tok</span>')
        # Match Badge
        if is_match:
            badges.append(f'<span class="badge badge-green">Match {int(score*100)}%</span>')
        # Warning Badge (Ends with punctuation?)
        if not chunk.strip().endswith(('.', '!', '?', '"', '”')):
            badges.append('<span class="badge badge-red">⚠️ Bad Cut</span>')
            
        badges_html = "".join(badges)
        
        # 4. CSS Classes
        card_class = "chunk-card match-card" if is_match else "chunk-card"
        
        # 5. Render HTML
        display_text = non_overlap_text.replace("\n", "<br>")
        
        html_code = textwrap.dedent(f"""
            <div class="{card_class}">
                <div class="card-header">
                    <span>CHUNK {i+1:02d}</span>
                    <div>{badges_html}</div>
                </div>
                <div class="chunk-text">
                    <span class="highlight-overlap" title="Overlap Region">{overlap_text}</span>{display_text}
                </div>
            </div>
        """)
        
        st.markdown(html_code, unsafe_allow_html=True)
        previous_chunk = chunk