import streamlit as st
from rag_pipeline import load_documents, prepare_chunks, create_vector_store, search, generate_answer
#-----#
st.markdown("""
<style>

body {
    background-color: #FFE4F1;
}

.stApp {
    background-color: #FFE4F1;
}

h1 {
    color: #8B1E54;
    text-align: center;
    font-weight: 700;
}

textarea {
    border-radius: 15px !important;
}

.stButton>button {
    background-color: #FF4FA3;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #E63E90;
    color: white;
}

[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1px solid #FFB6D5;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

[data-testid="stSidebar"] {
    background-color: #FFE4F0;
}

.sidebar-card {
    background-color: #FFF4FA;
    padding: 15px;
    border-radius: 12px;
    border: 2px solid #FF6FAE;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


st.set_page_config(
    page_title="Compliance AI Assistant",
    page_icon="💗⚖️",
    layout="centered"
)


st.markdown(
"""
<h1>⚖️Compliance AI Assistant⚖️</h1>
<p style="text-align:center;color:#8B1E54;font-size:18px;">
AI-Powered Regulatory Intelligence for Legal Compliance
</p>
""",
unsafe_allow_html=True
)

@st.cache_resource
def initialize_system():

    docs = load_documents()

    chunks = prepare_chunks(docs)

    index, model, chunk_metadata = create_vector_store(chunks)

    return index, model, chunk_metadata

st.sidebar.markdown(
"""
<div class="postit">

### 💡 Example Questions

These work best with the uploaded legal documents:

• What penalty can be imposed if an employer pays less than the minimum wage under the Minimum Wages Act?

• What punishment applies if safety provisions are violated under the Factories Act?

• What responsibilities do employers have under the Maharashtra Shops and Establishments Act?

• What compliance obligations do employers have under the four labour codes?

Tip: Ask specific questions about penalties, employer duties, or violations.

</div>
""",
unsafe_allow_html=True
)

with st.spinner("Initializing AI system..."):

    index, model, chunk_metadata = initialize_system()



query = st.text_area(
    "💼 Ask your legal compliance question",
    height=120,
    placeholder="Example: What is the punishment for violating the Labour Laws?"
)

if st.button("Generate Answer"):

    if query.strip() == "":
        st.warning("Please enter a question")

    else:

        with st.spinner("Searching documents..."):

            retrieved_chunks = search(query, index, chunk_metadata)

            answer = generate_answer(query, retrieved_chunks, model)

        st.divider()

        st.subheader("⚖️ Legal Summary")
        st.markdown(
f"""
<div style="
background-color:#FFF7FB;
padding:20px;
border-radius:15px;
border:1px solid #FFB6D5;
font-size:16px;
line-height:1.6;
">
{answer}
</div>
""",
unsafe_allow_html=True
)

        st.subheader("📚 Retrieved Sources")

        with st.expander("Click to view legal references"):

            for i, chunk in enumerate(retrieved_chunks):

                st.markdown(
f"""
<div style="color:#8B1E54;font-weight:600">
📄 Source {i+1}: {chunk['filename']}
</div>
""",
unsafe_allow_html=True
)

                st.markdown(chunk["text"][:400] + "...")

st.markdown(
"""
<hr>
<p style="text-align:center;color:#8B1E54">
✨ Built with RAG + FAISS + LLM ✨
</p>
""",
unsafe_allow_html=True
)