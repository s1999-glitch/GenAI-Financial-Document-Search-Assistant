import streamlit as st
from src.rag_pipeline import (
    load_documents,
    retrieve_relevant_documents,
    generate_business_answer,
)


st.set_page_config(
    page_title="GenAI Financial Document Search Assistant",
    page_icon="📄",
    layout="wide",
)


st.title("GenAI Financial Document Search Assistant")

st.write(
    "A finance-focused RAG prototype that helps users search financial documents, "
    "retrieve relevant information, and generate business-ready insights."
)


st.sidebar.title("Project Information")

st.sidebar.write("**Use Case:**")
st.sidebar.write("Financial document search, M&A due diligence, credit review, risk analysis")

st.sidebar.write("**Techniques:**")
st.sidebar.write("NLP, TF-IDF retrieval, cosine similarity, RAG concept")

st.sidebar.write("**Built With:**")
st.sidebar.write("Python, Streamlit, Scikit-learn")


DATA_PATH = "data/sample_financial_documents.txt"


try:
    documents = load_documents(DATA_PATH)

    st.success(f"{len(documents)} financial documents loaded successfully.")

    st.subheader("Ask a Question")

    user_question = st.text_input(
        "Enter your question:",
        placeholder="Example: What are the key risks in the M&A due diligence review?",
    )

    top_k = st.slider(
        "Number of relevant documents to retrieve:",
        min_value=1,
        max_value=5,
        value=3,
    )

    if st.button("Search Documents"):
        if user_question.strip() == "":
            st.warning("Please enter a question.")
        else:
            results = retrieve_relevant_documents(
                query=user_question,
                documents=documents,
                top_k=top_k,
            )

            answer = generate_business_answer(user_question, results)

            st.subheader("Generated Business Answer")
            st.markdown(answer)

            st.subheader("Retrieved Document Sections")

            for i, result in enumerate(results, start=1):
                with st.expander(
                    f"Retrieved Document {i} | Similarity Score: {result['score']}"
                ):
                    st.write(result["document"])

    st.subheader("Example Questions")

    example_questions = [
        "What are the key risks in the M&A due diligence review?",
        "How can machine learning support credit risk assessment?",
        "What is IFRS 9 Expected Credit Loss?",
        "How can anomaly detection help identify suspicious transactions?",
        "What operational issues were identified from branch feedback?",
        "Why is technology integration important in M&A?",
        "How can NLP improve client servicing analytics?",
    ]

    for question in example_questions:
        st.write(f"- {question}")


except FileNotFoundError:
    st.error(
        "Data file not found. Please check whether data/sample_financial_documents.txt exists."
    )
