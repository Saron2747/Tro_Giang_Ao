from live_rag_smoke import Client, Response, module
import json

query = "Hàm băm là gì?"
vector = Response("http://127.0.0.1:11434/api/embed", {"model": "nomic-embed-text", "input": "search_query: " + query}).json()["embeddings"][0]
collection = Client().get_collection("kb_attt_101_public")
result = collection.query(query_embeddings=[vector], n_results=24, include=["documents", "metadatas", "distances"])
print(json.dumps({"count": collection.count(), "hits": [{"distance": dist, "meta": meta, "text": doc}
    for dist, meta, doc in zip(result["distances"][0], result["metadatas"][0], result["documents"][0])]}, ensure_ascii=True))
