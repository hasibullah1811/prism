import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Prism | RAG Visualizer", page_icon="💎", layout="wide")

# Custom CSS for the "chunk cards"
st.markdown("""
<style>
    .chunk-card {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border: 1px solid #eee;
        color: #1f1f1f; /* Force dark text */
        transition: all 0.3s ease;
    }
    .chunk-header {
        font-size: 0.8em;
        color: #555555;
        margin-bottom: 5px;
        font-weight: bold;
    }
    .highlight-overlap {
        background-color: #ffd700;
        color: black;
        padding: 0 2px;
        border-radius: 3px;
        font-weight: bold;
    }
    /* New Style for Search Matches */
    .match-card {
        border: 2px solid #00c853 !important; /* Green Border */
        box-shadow: 0px 4px 15px rgba(0, 200, 83, 0.2);
    }
    .match-badge {
        background-color: #00c853;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        float: right;
    .warning-badge {
        background-color: #ffcc00; /* Yellow-Orange */
        color: black;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        float: right;
        margin-left: 10px;
        font-weight: bold;
    }
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def get_overlap_text(prev_chunk, current_chunk):
    if not prev_chunk: return ""
    check_len = min(len(prev_chunk), len(current_chunk))
    for i in range(check_len, 0, -1):
        suffix = prev_chunk[-i:]
        prefix = current_chunk[:i]
        if suffix == prefix: return suffix
    return ""

def calculate_similarity(query, chunks):
    """
    Calculates the cosine similarity between the query and every chunk.
    Returns a list of scores (0.0 to 1.0).
    """
    if not query or not chunks:
        return [0.0] * len(chunks)
    
    # Add query to the list of documents so we can vectorize them all together
    documents = [query] + chunks
    
    # Convert text to numbers (TF-IDF vectors)
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    # Calculate similarity between Query (index 0) and all Chunks (index 1 to end)
    query_vector = vectors[0:1]
    chunk_vectors = vectors[1:]
    
    # cosine_similarity returns a matrix, we flatten it to a simple list
    scores = cosine_similarity(query_vector, chunk_vectors).flatten()
    return scores

# --- HEADER ---
col1, col2 = st.columns([1, 6])
with col1: st.title("💎")
with col2: 
    st.title("Prism")
    st.caption("The RAG Chunk Visualizer. See how your data gets split & retrieved.")

st.divider()

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.header("🔪 Settings")
    chunk_size = st.slider("Chunk Size", 50, 2000, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 50, 10)
    if chunk_overlap >= chunk_size:
        st.error("⚠️ Overlap must be smaller than Chunk Size!")
        chunk_overlap = chunk_size - 1
    # st.divider()
    # st.header("💾 Export")

# --- MAIN: INPUT AREA ---
default_text = """RAG (Retrieval-Augmented Generation) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.

Building a RAG pipeline usually involves two main steps: retrieval and generation. In the retrieval step, the system searches for relevant documents in a knowledge base. In the generation step, the LLM uses the retrieved context to answer the user's question.

However, splitting the text correctly is crucial. If you split the text in the middle of a sentence, the model might lose the semantic meaning. This is why tools like Prism are necessary to visualize the process."""

text_input = st.text_area("📄 Paste your document text here:", value=default_text, height=200)

# --- SEARCH BAR ---
query = st.text_input("🔍 Simulate Retrieval (Type a query to see which chunks match):", placeholder="e.g., 'What is the retrieval step?'")

# --- LOGIC: SPLITTING & SEARCH ---
if text_input:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    chunks = splitter.split_text(text_input)
    print("✅ DEBUG: Splitting finished.")
    
# Run the Search logic
    print("✅ DEBUG: Starting Similarity Calculation...") 
    scores = calculate_similarity(query, chunks)
    print("✅ DEBUG: Similarity Calculation Done.") 

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Chunks", len(chunks))
    m2.metric("Avg Chunk Size", f"{int(sum(len(c) for c in chunks) / len(chunks))} chars")
    
    # Show Top Match Score
    top_score = max(scores) if any(scores) else 0
    m3.metric("Top Match Score", f"{int(top_score * 100)}%")

    # --- EXPORT BUTTON ---
    # Convert chunks to a structured JSON format
    chunk_data = [{"chunk_id": i+1, "content": c, "length": len(c)} for i, c in enumerate(chunks)]
    json_string = json.dumps(chunk_data, indent=2)
    
    st.download_button(
        label="📥 Download Chunks as JSON",
        data=json_string,
        file_name="prism_chunks.json",
        mime="application/json",
        help="Download these chunks to use in your Vector Database."
    )
    
    st.divider()
    st.subheader("Results")

    colors = ["#e3f2fd", "#e8f5e9", "#f3e5f5", "#fff3e0"]
    previous_chunk = None

for i, chunk in enumerate(chunks):
        color = colors[i % len(colors)]
        
        # --- 1. Overlap Logic ---
        overlap_text = ""
        non_overlap_text = chunk
        if previous_chunk:
            overlap = get_overlap_text(previous_chunk, chunk)
            if overlap:
                overlap_text = overlap
                non_overlap_text = chunk[len(overlap):]
        
        # --- 2. Search Match Logic ---
        score = scores[i]
        is_match = score > 0.2
        card_class = "chunk-card match-card" if is_match else "chunk-card"
        
        # Create Badges HTML
        badges_html = ""
        if is_match:
            badges_html += f'<span class="match-badge">Match: {int(score*100)}%</span>'

        # --- 3. "Bad Cut" Warning Logic ⚠️ ---
        # Check if the chunk ends with valid punctuation.
        # We strip whitespace to ignore trailing newlines.
        stripped_chunk = chunk.strip()
        if not stripped_chunk.endswith(('.', '!', '?', '"', '”')):
            badges_html += '<span class="warning-badge">⚠️ Bad Cut</span>'
        
        # --- 4. Render HTML ---
        display_text = non_overlap_text.replace("\n", "<br>")
        
        html_code = f'<div class="{card_class}" style="background-color: {color}"><div class="chunk-header">CHUNK {i + 1} • {len(chunk)} CHARS {badges_html}</div><span class="highlight-overlap" title="Overlap">{overlap_text}</span>{display_text}</div>'

        st.markdown(html_code, unsafe_allow_html=True)
        previous_chunk = chunk

else:
    st.info("👈 Waiting for text input...")