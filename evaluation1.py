import numpy as np
from collections import defaultdict

from similarity import calculate_similarity_in_batches, vectorizer, tfidf_matrix, docs
from processquery import process


def read_qrels(qrels_file_path):
    qrels = defaultdict(dict)
    query_counts = defaultdict(int)

    with open(qrels_file_path, 'r') as qrels_file:
        for line in qrels_file:
            line_parts = line.strip().split(' ')
            if len(line_parts) != 4:
                print("Invalid line format:", line)
                continue

            query_id, _, doc_id, relevance = line_parts
            query_id = int(query_id)
            doc_idd = doc_id.split('_')[0]

            qrels[query_id][doc_idd] = int(relevance)
            query_counts[query_id] += 1

    return qrels, query_counts


def read_queries(queries_file_path):
    queries = {}
    with open(queries_file_path, 'r') as queries_file:
        for line in queries_file:
            line_parts = line.strip().split(None, 1)
            if len(line_parts) < 2:
                continue
            query_id = int(line_parts[0])
            query = line_parts[1]
            query = process(query)
            queries[query_id] = query

    return queries


def save_relevant_num(relevant_num_file, query_id, query, num_relevant, num_returned, precision_list, relevant_documents, query_counts):
    relevant_num_file.write(f"Query ID: {query_id}\n")
    relevant_num_file.write(f"Query: {query}\n")
    relevant_num_file.write(f"Query Count: {query_counts[query_id]}\n")
    relevant_num_file.write(f"Relevant Documents: {num_relevant}\n")
    relevant_num_file.write(f"Total Returned Documents: {num_returned}\n")
    relevant_num_file.write("Precision List: " + ', '.join(map(str, precision_list)) + "\n")
    relevant_num_file.write("\n".join(relevant_documents))
    relevant_num_file.write("\n\n")


def retrieve_similar_documents(query_id, query, qrels, similarity_scores):
    # sorted_indices = np.argsort(similarity_scores, axis=1)[0, ::-1]
    sorted_indices = np.argsort(similarity_scores)[::-1]  # Sort in descending order
    threshold = 0.00000001
    num_relevant = 0
    num_returned = 0
    relevant_documents = []
    precision_list = []
    Ap = []

    for idx in sorted_indices:
        similarity_score = similarity_scores[idx]
        doc_id = list(docs.keys())[idx]
        doc_idd = doc_id.split('_')[0]

        if similarity_score > 0.1:
            if similarity_score >= threshold and doc_idd in qrels.get(query_id, {}):
                num_relevant += 1
                relevant_documents.append(f"Similarity Score: {similarity_score}, Document ID: {doc_id}, Document: {docs[doc_id]}")
                AP = num_relevant / num_returned if num_returned > 0 else 0.0
                Ap.append(AP)

            num_returned += 1
            if num_returned >= 10:
                break
            precision = num_relevant / num_returned if num_returned > 0 else 0.0
            precision_list.append(precision)
            print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
            print(precision_list)

    Total = len(sorted_indices)

    return num_relevant, num_returned, precision_list, Total, relevant_documents, Ap


def AP(Ap):
    print(len(Ap))
    return sum(Ap) / len(Ap) if len(Ap) > 0 else 0.0


def calculate_average_precision(precision_list):
    return sum(precision_list) / len(precision_list) if len(precision_list) > 0 else 0.0


def calculate_recall(num_relevant, query_count):
    return num_relevant / query_count


def process_queries(qrels_file_path, queries_file_path, relevant_num_file_path):
    qrels, query_counts = read_qrels(qrels_file_path)
    queries = read_queries(queries_file_path)
    average_precisions = []
    MAP=[]

    with open(relevant_num_file_path, 'w') as relevant_num_file:
        for query_id, query in queries.items():
            print("Query ID:", query_id)
            print("Query:", query)

            if query_id not in qrels:
                print("No relevance judgments found for this query.")
                continue

            query_vector = vectorizer.transform([query])
            similarity_scores = calculate_similarity_in_batches(query_vector, tfidf_matrix)

            num_relevant, num_returned, precision_list, Total, relevant_documents, Ap = retrieve_similar_documents(query_id, query, qrels, similarity_scores)

            save_relevant_num(relevant_num_file, query_id, query, num_relevant, num_returned, precision_list, relevant_documents, query_counts)

            average_precision = calculate_average_precision(precision_list)
            recall = calculate_recall(num_relevant, query_counts[query_id])

            average_precisions.append(average_precision)
            print("Average Precision:", average_precision)
            print("Recall:", recall)

            ap = AP(Ap)  # Calculate AP for the current query
            MAP.append(ap)


            print("AP:", ap)

            print("--------------------------------------")
        mean_ap = np.mean(MAP)

    return average_precisions, mean_ap


# Set file paths
qrels_file_path = r"C:\Users\user\Documents\IRSystem1\qrel1.txt"
queries_file_path = r"C:\Users\user\Documents\IRSystem1\queries1.txt"
relevant_num_file_path = r"C:\Users\user\Documents\IRSystem1\relevanttt_num_queries"

# Process queries
average_precisions, mean_ap = process_queries(qrels_file_path, queries_file_path, relevant_num_file_path)

# print("Mean Average Precision:", np.mean(average_precisions))
print("Mean AP:", mean_ap)
