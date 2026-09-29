import streamlit as st
import os
from src.graph import build_rag_graph

# Page configuration
st.set_page_config(
    page_title="Agentic AI RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic AI RAG Assistant")
st.caption("Ask questions strictly based on the provided Agentic AI eBook.")

# Sidebar for metadata, score, and retrieved context chunks
with st.sidebar:
    st.header("🔍 Retrieval Inspector")
    st.info("Here you can inspect the retrieved context chunks and the confidence score for the latest response.")
    
    score_placeholder = st.empty()
    context_placeholder = st.empty()
    
    st.markdown("---")
    st.markdown("### ⚙️ System Settings")
    st.write("**Model:** gpt-4o-mini")
    st.write("**Embedding:** text-embedding-3-small")
    st.write("**Vector DB:** Pinecone (agentic-ai-index)")
    st.write("**Orchestrator:** LangGraph StateGraph")

# Initialize graph cached
@st.cache_resource
def get_graph():
    index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
    return build_rag_graph(index_name=index_name)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User query input
if prompt := st.chat_input("Ask a question about the Agentic AI eBook..."):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query through LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Retrieving from Pinecone & Generating grounded response..."):
            try:
                graph = get_graph()
                result = graph.invoke({
                    "question": prompt,
                    "context": [],
                    "answer": "",
                    "score": 0.0
                })
                
                answer = result["answer"]
                context_chunks = result["context"]
                score = result["score"]

                st.markdown(answer)

                # Store assistant response in history
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Update sidebar with confidence score and context chunks
                score_color = "green" if score > 0.5 else "red"
                score_placeholder.markdown(f"**Confidence Score:** :{score_color}[{score:.2f}]")

                with context_placeholder.container():
                    st.subheader(f"Retrieved Chunks ({len(context_chunks)})")
                    if context_chunks:
                        for idx, chunk in enumerate(context_chunks, 1):
                            with st.expander(f"📄 Chunk {idx}", expanded=(idx == 1)):
                                st.markdown(chunk)
                    else:
                        st.warning("No context chunks retrieved.")
            except Exception as e:
                st.error(f"Error executing RAG pipeline: {str(e)}")
