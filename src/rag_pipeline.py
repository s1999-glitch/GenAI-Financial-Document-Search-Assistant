import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_documents(file_path):
    """
    Load financial documents from a text file.
    Each document starts with DOCUMENT 1, DOCUMENT 2, etc.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Split documents whenever DOCUMENT 1, DOCUMENT 2, etc. appears
    documents = re.split(r"(?=DOCUMENT \d+:)", text)

    # Remove empty spaces
    documents = [doc.strip() for doc in documents if doc.strip()]

    return documents


def retrieve_relevant_documents(query, documents, top_k=3):
    """
    Retrieve the most relevant financial documents using TF-IDF and cosine similarity.
    """

    if len(documents) == 0:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")

    document_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])

    similarity_scores = cosine_similarity(query_vector, document_vectors).flatten()
    ranked_indices = similarity_scores.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        results.append(
            {
                "document": documents[index],
                "score": round(float(similarity_scores[index]), 3),
            }
        )

    return results


def generate_business_answer(query, retrieved_documents):
    """
    Generate a business-style answer from the retrieved document.
    """

    if not retrieved_documents:
        return "No relevant information was found in the available documents."

    best_document = retrieved_documents[0]["document"]
    score = retrieved_documents[0]["score"]

    answer = f"""
### Business Answer

Based on the retrieved financial document, the most relevant information for your question is:

{best_document}

### Business Interpretation

This information can support financial analysis, M&A due diligence, credit review, risk assessment, and management decision-making. It helps users quickly identify important insights from financial documents instead of manually reviewing long reports.

### Retrieval Confidence

Similarity score: {score}

A higher score means the retrieved document is more closely related to the question.
"""

    return answer
