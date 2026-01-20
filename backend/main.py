from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from sklearn.decomposition import PCA

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Added 'query' field (optional)
class SplitRequest(BaseModel):
    text: str
    query: str = ""
    chunk_size: int
    overlap: int

def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def get_overlap_text(prev_chunk, current_chunk):
    if not prev_chunk: return ""
    check_len = min(len(prev_chunk), len(current_chunk))
    for i in range(check_len, 0, -1):
        suffix = prev_chunk[-i:]
        prefix = current_chunk[:i]
        if suffix == prefix: return suffix
    return ""

# --- NEW FUNCTION: TF-IDF Search ---
def calculate_similarity(query, chunks):
    if not query or not chunks: return [0.0] * len(chunks)
    # Create vectors for Query + All Chunks
    documents = [query] + chunks
    tfidf = TfidfVectorizer(stop_words='english').fit_transform(documents)
    # Calculate cosine similarity between Query (0) and Chunks (1..N)
    cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    return cosine_similarities

# --- UPDATED MATH FUNCTION ---
def calculate_vectors(query, chunks):
    if not chunks: return [], []
    
    # 1. Vectorize (TF-IDF)
    documents = [query] + chunks if query else chunks
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # 2. Calculate Similarity (Score)
    if query:
        # Query is at index 0, chunks are 1..N
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    else:
        scores = [0.0] * len(chunks)

    # 3. Calculate 2D Map (PCA)
    # We need at least 3 points to make a meaningful 2D PCA (Query + 2 Chunks)
    coords = []
    if len(documents) >= 3:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(tfidf_matrix.toarray())
        coords = reduced.tolist() # Convert to standard Python list [ [x,y], [x,y] ]
    else:
        # Fallback for too few chunks
        coords = [[0,0]] * len(documents)
        
    return scores, coords

@app.post("/process-text")
def process_text(request: SplitRequest):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=request.chunk_size,
        chunk_overlap=request.overlap,
        length_function=len
    )
    chunks = splitter.split_text(request.text)
    
    # Get Scores AND Coordinates
    scores, coords = calculate_vectors(request.query, chunks)
    
    results = []
    previous_chunk = None
    
    # If we have a query, the first coord is the query, the rest are chunks
    chunk_coords = coords[1:] if request.query else coords
    
    for i, chunk in enumerate(chunks):
        overlap_text = get_overlap_text(previous_chunk, chunk)
        remaining_text = chunk[len(overlap_text):]
        
        # Get x, y for this chunk (default to 0 if PCA failed)
        x, y = chunk_coords[i] if i < len(chunk_coords) else (0, 0)
        
        results.append({
            "id": i + 1,
            "overlap": overlap_text,
            "remaining": remaining_text,
            "tokens": count_tokens(chunk),
            "bad_cut": not chunk.strip().endswith(('.', '!', '?', '"', '”')),
            "score": float(scores[i]) if i < len(scores) else 0.0,
            "x": float(x), # Send X to frontend
            "y": float(y)  # Send Y to frontend
        })
        previous_chunk = chunk
        
    # Return Query Coordinates too (so we can plot the Red Dot)
    query_x, query_y = (coords[0][0], coords[0][1]) if request.query and coords else (0, 0)
        
    return {
        "chunks": results,
        "query_coords": {"x": float(query_x), "y": float(query_y)}
    }