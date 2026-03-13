
import streamlit as st
from rag_engine import load_data, build_index, retrieve

st.title("MUJ-Copilot")

st.write("Ask anything about academics, student wellbeing, or campus life.")

texts = load_data()
index, texts = build_index(texts)

query = st.text_input("Ask a question")

if query:

    results = retrieve(query,index,texts)

    st.subheader("Relevant Data")
    for r in results:
        st.write(r)

    st.subheader("AI Response")

    response = f"""
Based on MUJ student datasets, here are insights:

{results}

Possible recommendation:
Students with higher study hours and attendance show better final grades.
Managing stress and maintaining sleep improves academic performance.
"""

    st.write(response)
