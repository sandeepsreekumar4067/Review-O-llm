examples = {
#     "contextual": ["good", "bad", "wait", "long", "wouldnt reccomend"],
#     "casual": ["hi", "hello", "hey", "oh"],
# }
# threshold = 1.4
# legal_embeddings = embedding_model.embed_documents(examples["contextual"])
# casual_embeddings = embedding_model.embed_documents(examples["casual"])


# def classify_query(query):
#     # Embed the query
#     query_embedding = embedding_model.embed_documents([query])

#     # Calculate similarities

#     legal_similarity = cosine_similarity(query_embedding, legal_embeddings)
#     casual_similarity = cosine_similarity(query_embedding, casual_embeddings)

#     # Get the maximum similarity score for legal and casual categories
#     max_legal_similarity = max(legal_similarity[0])
#     max_casual_similarity = max(casual_similarity[0])

#     # Set a threshold for classifying the query
#     threshold = 0.4

#     # Compare similarities to classify as legal or casual
#     if (
#         max_legal_similarity > threshold
#         and max_legal_similarity > max_casual_similarity
#     ):
#         return {"status": "legal", "score": max_legal_similarity}
#     else:
#         return {"status": "casual", "score": max_casual_similarity}

