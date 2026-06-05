import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

job_desc = st.text_area("Paste Job Description")

if uploaded_file and job_desc:

    pdf = PdfReader(uploaded_file)

    resume_text = ""

    for page in pdf.pages:
        resume_text += page.extract_text()

    cv = CountVectorizer()

    matrix = cv.fit_transform([resume_text, job_desc])

    similarity = cosine_similarity(matrix)[0][1]

    score = round(similarity * 100, 2)

    st.subheader("Results")

    st.write(f"Match Score: {score}%")

    if score > 70:
        st.success("Strong Match")
    elif score > 40:
        st.warning("Moderate Match")
    else:
        st.error("Weak Match")
