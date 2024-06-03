import traceback

from flask import Flask, request, jsonify, render_template
from processquery import process  
from similarity import calculate_similarity_in_batches, vectorizer, tfidf_matrix, docs
from suggestion import on_text_changed

app = Flask(__name__)

# with open(r"C:\Users\user\Documents\IRSystem1\antique-test-queries.txt", 'r') as file:
#     query = file.read()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    try:
        query = request.json['query']  # Assuming the query is sent as a JSON object with the key 'query'
        number = request.json['number']
        page = request.json.get('page', 1)
        per_page = request.json.get('per_page', 10)
        query = process(query)


        output_file = r"C:\Users\user\Documents\IRSystem1\num.txt"
        with open(output_file, 'w') as file:
            file.write(number)
            file.close()

        output_file = r"C:\Users\user\Documents\IRSystem1\copy.txt"
        with open(output_file, 'w') as file:
            file.write(query)
            file.close()


        query_vector = vectorizer.transform([query])

        # Directly use cosine_similarity on sparse matrices
        similarity_scores = calculate_similarity_in_batches(query_vector, tfidf_matrix)

        # Sort the results by similarity score
        results = []

        for doc_id, score in sorted(zip(docs.keys(), similarity_scores), key=lambda x: x[1], reverse=True):
            doc_content = docs[doc_id]
            result = {
                'doc_id': doc_id,
                'score': score,
                'content': doc_content
            }
            results.append(result)

        start = (page - 1) * per_page
        end = start + per_page
        paginated_results = results[start:end]

        return jsonify({
            "results": paginated_results,
            "total_results": len(results),
            "page": page,
            "per_page": per_page
        })

        # return jsonify(results)
    except Exception as e:
            print(traceback.format_exc())
            return jsonify({"error": str(e)}), 500


@app.route('/suggestion', methods=['POST'])
def suggestion():
    try:
        query = request.json['query']
        result = on_text_changed(query)
        return jsonify(result)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=8000)
