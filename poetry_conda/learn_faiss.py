import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd

df = pd.read_csv('sample_text.csv')
print(df)

encoder = SentenceTransformer('all-mpnet-base-v2')
vectors = encoder.encode(df.text)
print(vectors.shape)

dim = vectors.shape[1]
index = faiss.IndexFlatL2(dim)
print(index)

index.add(vectors)

search_query = "I want to buy a Tommy Hillfiger Jeans"
vec = encoder.encode(search_query).reshape(1,-1)

distances, Idx = index.search(vec, k=2)
print(df.loc[Idx[0]])

search_query = "Apple is good to eat."
vec = encoder.encode(search_query).reshape(1,-1)

distances, Idx = index.search(vec, k=2)
print(df.loc[Idx[0]])

search_query = "Looking for a place to visit holidays."
vec = encoder.encode(search_query).reshape(1,-1)

distances, Idx = index.search(vec, k=2)
print(df.loc[Idx[0]])

