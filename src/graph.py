from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_pinecone import PineconeVectorStore
from src.config import (
    PINECONE_INDEX_NAME,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    TOP_K,
)


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


def get_llm_and_embeddings():
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GEMINI_API_KEY
        )
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            google_api_key=GEMINI_API_KEY
        )
    else:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
    return llm, embeddings


def build_rag_graph(index_name: str = None):
    target_index = index_name or PINECONE_INDEX_NAME
    llm, embeddings = get_llm_and_embeddings()

    vectorstore = PineconeVectorStore(index_name=target_index, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # 1. Retrieve Node
    def retrieve_node(state: AgentState):
        docs = retriever.invoke(state["question"])
        context_texts = [d.page_content for d in docs]
        return {"context": context_texts}

    # 2. Generate Node with Strict Grounding
    def generate_node(state: AgentState):
        context_str = "\n\n".join(state["context"])
        prompt = f"""You are a strict assistant. Answer the question relying ONLY on the context below.
If the context does not contain enough info, state 'I cannot answer based on the provided document.'

Context:
{context_str}

Question: {state['question']}"""

        response = llm.invoke(prompt)
        answer_text = response.content

        # Score evaluation based on context availability & refusal detection
        if not state["context"] or "cannot answer based on the provided document" in answer_text.lower():
            confidence = 0.0
        else:
            confidence = 0.95

        return {"answer": answer_text, "score": confidence}

    # 3. Assemble StateGraph
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
