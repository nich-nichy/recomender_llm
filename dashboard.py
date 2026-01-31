"""
Emotion-Aware Book Recommender
"""

import os
import numpy as np
import pandas as pd
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# Setup
# -------------------------------------------------
print("Starting OFFLINE Book Recommender")

# -------------------------------------------------
# Load data
# -------------------------------------------------
print("Loading book data...")
books = pd.read_csv("books_with_emotions.csv")
print(f"Loaded {len(books)} books")

# Create high-res thumbnails
books["large_thumbnail"] = books["thumbnail"].fillna("cover_not_found.jpg")
books["large_thumbnail"] = books["large_thumbnail"].apply(
    lambda x: x + "@fife=w800" if x != "cover_not_found.jpg" else x
)

# -------------------------------------------------
# Prepare descriptions
# -------------------------------------------------
print("Preparing descriptions...")
books_clean = books.dropna(subset=["isbn13", "description"]).copy()
books_clean["isbn13"] = pd.to_numeric(books_clean["isbn13"], errors="coerce")
books_clean = books_clean.dropna(subset=["isbn13"])

descriptions = []
isbn_list = []

for _, row in books_clean.iterrows():
    desc = str(row["description"]).replace("\n", " ").strip()
    if desc:
        descriptions.append(desc)
        isbn_list.append(int(row["isbn13"]))

print(f"Prepared {len(descriptions)} descriptions")

# -------------------------------------------------
# Load local embedding model (FREE)
# -------------------------------------------------
print("Loading local embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

# -------------------------------------------------
# Generate / load embeddings
# -------------------------------------------------
CACHE_FILE = "embeddings_local.npy"

if os.path.exists(CACHE_FILE):
    print("Loading cached embeddings...")
    embeddings_matrix = np.load(CACHE_FILE)
else:
    print("Generating embeddings (one-time)...")
    embeddings = model.encode(
        descriptions,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    embeddings_matrix = np.array(embeddings).astype("float32")
    np.save(CACHE_FILE, embeddings_matrix)
    print(f"Saved embeddings → {CACHE_FILE}")

print(f"Embeddings ready: {embeddings_matrix.shape}")

# -------------------------------------------------
# Build FAISS index
# -------------------------------------------------
print("Building FAISS index...")
index = faiss.IndexFlatIP(EMBEDDING_DIM)  # cosine similarity
index.add(embeddings_matrix)
print(f"FAISS index size: {index.ntotal}")

# -------------------------------------------------
# Recommendation logic
# -------------------------------------------------
def recommend_books(query: str, category: str = "All", tone: str = "All"):
    try:
        if not query or not query.strip():
            return [("cover_not_found.jpg", "Please enter a description.")]

        query_emb = model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        k = min(50, index.ntotal)
        distances, indices = index.search(query_emb, k)

        matched_isbns = [
            isbn_list[i] for i in indices[0] if i < len(isbn_list)
        ]

        recs = books[books["isbn13"].isin(matched_isbns)].copy()

        if recs.empty:
            return [("cover_not_found.jpg", "No matching books found.")]

        if category != "All":
            recs = recs[recs["simple_categories"] == category]

        if tone == "Happy":
            recs = recs.sort_values("joy", ascending=False)
        elif tone == "Surprising":
            recs = recs.sort_values("surprise", ascending=False)
        elif tone == "Angry":
            recs = recs.sort_values("anger", ascending=False)
        elif tone == "Suspenseful":
            recs = recs.sort_values("fear", ascending=False)
        elif tone == "Sad":
            recs = recs.sort_values("sadness", ascending=False)

        output = []
        for _, row in recs.head(16).iterrows():
            authors = str(row.get("authors", "Unknown")).split(";")[0]
            caption = f"**{row['title']}**\nby {authors}"
            output.append((row["large_thumbnail"], caption))

        return output

    except Exception as e:
        print("❌ Backend error:", e)
        return [("cover_not_found.jpg", "Internal error occurred.")]

# -------------------------------------------------
# UI
# -------------------------------------------------
categories = ["All"] + sorted(
    books["simple_categories"].dropna().astype(str).unique().tolist()
)
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 Sentiment analysis Book Recommender")

    query = gr.Textbox(
        label="Describe a book or mood",
        placeholder="A dark mystery thriller with twists...",
        lines=2
    )

    with gr.Row():
        category = gr.Dropdown(categories, value="All", label="Category")
        tone = gr.Dropdown(tones, value="All", label="Emotion")

    btn = gr.Button("🔍 Find Books")

    gallery = gr.Gallery(columns=4, rows=2, height="auto")

    btn.click(recommend_books, [query, category, tone], gallery)
    query.submit(recommend_books, [query, category, tone], gallery)

# -------------------------------------------------
# Launch
# -------------------------------------------------
if __name__ == "__main__":
    print("🚀 Running at http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860)
