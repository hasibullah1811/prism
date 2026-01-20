from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

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

@app.post("/process-text")
def process_text(request: SplitRequest):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=request.chunk_size,
        chunk_overlap=request.overlap,
        length_function=len
    )
    chunks = splitter.split_text(request.text)
    
    # Calculate Search Scores
    scores = calculate_similarity(request.query, chunks)
    
    results = []
    previous_chunk = None
    
    for i, chunk in enumerate(chunks):
        overlap_text = get_overlap_text(previous_chunk, chunk)
        remaining_text = chunk[len(overlap_text):]
        
        results.append({
            "id": i + 1,
            "overlap": overlap_text,
            "remaining": remaining_text,
            "tokens": count_tokens(chunk),
            "bad_cut": not chunk.strip().endswith(('.', '!', '?', '"', '”')),
            "score": float(scores[i]) # Send score to frontend
        })
        previous_chunk = chunk
        
    return {"chunks": results}