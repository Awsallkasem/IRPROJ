# -*- coding: utf-8 -*-

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse, csr_matrix


tfidf_file_path = r"C:\Users\user\Documents\IRSystem1\tfidf_matrix_new.pkl"
inverted_index_file_path = r"C:\Users\user\Documents\IRSystem1\inverted_index_new.pkl"
docs_file_path = r"C:\Users\user\Documents\IRSystem1\docs12.pkl"

def load_tfidf_matrix(file_path):
    with open(file_path, 'rb') as tfidf_file:
        tfidf_matrix, vectorizer = pickle.load(tfidf_file)
    # Ensure the tfidf_matrix is sparse
    if not issparse(tfidf_matrix):
        tfidf_matrix = csr_matrix(tfidf_matrix)

    return tfidf_matrix, vectorizer

def load_inverted_index(file_path):
    with open(file_path, 'rb') as file:
        inverted_index = pickle.load(file)
    return inverted_index

def load_docs(file_path):
    with open(file_path, 'rb') as file:
        docs = pickle.load(file)
    return docs

tfidf_matrix, vectorizer = load_tfidf_matrix(tfidf_file_path)
inverted_index = load_inverted_index(inverted_index_file_path)
docs = load_docs(docs_file_path)

# def calculate_similarity(query_vector, tfidf_matrix):
#     similarity_scores = cosine_similarity(query_vector.reshape(1, -1), tfidf_matrix)
#     return similarity_scores

# def calculate_similarity(query_vector, tfidf_matrix):
#     # Ensure the query_vector is sparse
#     if not issparse(query_vector):
#         query_vector = csr_matrix(query_vector)
#     similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
#     return similarity_scores

def calculate_similarity_in_batches(query_vector, tfidf_matrix, batch_size=1000):
    if not issparse(query_vector):
        query_vector = csr_matrix(query_vector)

    n_docs = tfidf_matrix.shape[0]
    similarity_scores = np.zeros(n_docs)

    for start in range(0, n_docs, batch_size):
        end = min(start + batch_size, n_docs)
        batch = tfidf_matrix[start:end]
        batch_similarity = cosine_similarity(query_vector, batch).flatten()
        similarity_scores[start:end] = batch_similarity

    return similarity_scores

#للتجربة فقط
# def process_query(query, vectorizer, tfidf_matrix, docs):
#     query_vector = vectorizer.transform([query])
#     similarity_scores = calculate_similarity(query_vector, tfidf_matrix)

#     # Convert similarity scores to a list of (doc_id, score) tuples
#     results = [(doc_id, score) for doc_id, score in zip(docs.keys(), similarity_scores.flatten())]
#     # Sort results by score in descending order
#     sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

#     return sorted_results

# query = "example search"
# results = process_query(query, vectorizer, tfidf_matrix, docs)
# print("Search results:", results)

#Display First Part of Results
# N = 5
# half = N // 2
# for doc_id, score in results[:half]:
#     print(f"Document ID: {doc_id}, Similarity Score: {score}")
#     print(docs[doc_id][:500])  # Display the first 500 characters of the document
#     print("...")
#     print("-----------------------------")

#Display Second Part of Results
# for doc_id, score in results[half:N]:
#     print(f"Document ID: {doc_id}, Similarity Score: {score}")
#     print(docs[doc_id][:500])  # Display the first 500 characters of the document
#     print("...")
#     print("-----------------------------")