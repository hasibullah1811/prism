import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_overlap_text(prev_chunk, current_chunk):
    """
    Finds the overlapping text between the end of the previous chunk 
    and the start of the current chunk.
    """
    if not prev_chunk:
        return ""
        
    # We look for the longest suffix of prev_chunk that matches a prefix of current_chunk
    # Start checking from the smaller length of the two chunks
    check_len = min(len(prev_chunk), len(current_chunk))
    
    for i in range(check_len, 0, -1):
        suffix = prev_chunk[-i:]
        prefix = current_chunk[:i]
        if suffix == prefix:
            return suffix
            
    return ""

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
        color: #1f1f1f; /* <--- FORCE DARK TEXT COLOR */
    }
    .chunk-header {
        font-size: 0.8em;
        color: #555555; /* <--- FORCE DARK GREY FOR HEADERS */
        margin-bottom: 5px;
        font-weight: bold;
    }
    .highlight-overlap {
        background-color: #ffd700;
        color: black; /* <--- Ensure highlight text is also black */
        padding: 0 2px;
        border-radius: 3px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col1, col2 = st.columns([1, 6])
with col1:
    st.title("💎")
with col2:
    st.title("Prism")
    st.caption("The RAG Chunk Visualizer. See how your data gets split before you embed it.")

st.divider()

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.header("🔪 Splitter Settings")
    
    chunk_size = st.slider(
        "Chunk Size", 
        min_value=50, 
        max_value=2000, 
        value=500, 
        step=50,
        help="The maximum number of characters in a single chunk."
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap", 
        min_value=0, 
        max_value=500, 
        value=50, 
        step=10,
        help="How many characters interfere with the previous chunk to maintain context."
    )
    
    # Validation to prevent errors
    if chunk_overlap >= chunk_size:
        st.error("⚠️ Overlap must be smaller than Chunk Size!")
        chunk_overlap = chunk_size - 1

    separators = st.text_input(
        "Separators (Preview)", 
        value=["\n\n", "\n", " ", ""], 
        disabled=True,
        help="Recursive splitter tries these separators in order."
    )

# --- MAIN: INPUT AREA ---
default_text = """RAG (Retrieval-Augmented Generation) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.

Building a RAG pipeline usually involves two main steps: retrieval and generation. In the retrieval step, the system searches for relevant documents in a knowledge base. In the generation step, the LLM uses the retrieved context to answer the user's question.

However, splitting the text correctly is crucial. If you split the text in the middle of a sentence, the model might lose the semantic meaning. This is why tools like Prism are necessary to visualize the process."""

text_input = st.text_area("📄 Paste your document text here:", value=default_text, height=300)

# --- LOGIC: SPLITTING ---
if text_input:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = splitter.split_text(text_input)

    # --- METRICS ROW ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Chunks", len(chunks))
    m2.metric("Avg Chunk Size", f"{int(sum(len(c) for c in chunks) / len(chunks))} chars")
    # Simple cost estimation (approx $0.0001 per 1k tokens for embeddings)
    est_tokens = len(text_input) / 4
    m3.metric("Est. Tokens", int(est_tokens), help="Rough estimate (1 token ≈ 4 chars)")

    st.subheader("🔍 Visualization")
    
    colors = ["#e3f2fd", "#e8f5e9", "#f3e5f5", "#fff3e0"]

    previous_chunk = None

    for i, chunk in enumerate(chunks):
        color = colors[i % len(colors)]
        
        # Calculate overlap with the previous chunk
        overlap_text = ""
        non_overlap_text = chunk
        
        if previous_chunk:
            overlap = get_overlap_text(previous_chunk, chunk)
            if overlap:
                overlap_text = overlap
                # Remove the overlapping part from the display text so we don't print it twice
                # (We will print it with a highlighter style)
                non_overlap_text = chunk[len(overlap):]

        # HTML Injection: We highlight the overlap in Yellow
        html_code = f"""
        <div class="chunk-card" style="background-color: {color}">
            <div class="chunk-header">CHUNK {i + 1} • {len(chunk)} CHARS</div>
            <span class="highlight-overlap" title="This text is repeated from the previous chunk">{overlap_text}</span>{non_overlap_text}
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)
        
        previous_chunk = chunk

else:
    st.info("👈 Waiting for text input...")