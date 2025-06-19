# pipeline.py

import os
os.environ['CQLENG_ALLOW_SCHEMA_MANAGEMENT'] = '1'
os.environ['EVENTLET_MONKEY_PATCH'] = '0'  # Disable automatic monkey patching

import eventlet
eventlet.monkey_patch(thread=False)  # Don't monkey patch threading

from typing import List
from typing_extensions import TypedDict
from langchain.schema import document
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.vectorstores import Cassandra  # Changed from langchain.vectorstores.cassandra
from langchain_community.embeddings import HuggingFaceEmbeddings  # Changed from langchain_huggingface
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langgraph.graph import END, StateGraph, START
from langchain_groq import ChatGroq
import cassio
import streamlit as st

# Use environment variables in place of Google Colab's userdata
# Set these in your environment or use a .env file with dotenv.load_dotenv()
groq_api_key = st.secrets["GROQ_API_KEY"]
hf_api_key = st.secrets["HF_API_KEY"]

# Load documents from web
def load_documents():
    # Predefined content to use when web scraping fails
    default_content = [
        {
            "content": """
            AI agents are autonomous systems that can perceive and act in their environment.
            Agents can be categorized into:
            1. Goal-driven agents that pursue specific objectives
            2. Reactive agents that respond to environmental stimuli
            3. Learning agents that improve through experience
            """,
            "source": "default_agents",
            "title": "AI Agents Overview"
        },
        {
            "content": """
            Prompt engineering is the practice of designing effective prompts for language models.
            Key principles include:
            1. Clear and specific instructions
            2. Context and examples when needed
            3. Breaking complex tasks into steps
            4. Proper formatting and structure
            """,
            "source": "default_prompt_engineering",
            "title": "Prompt Engineering Guide"
        },
        {
            "content": """
            Adversarial attacks on language models include:
            1. Input manipulation to produce incorrect outputs
            2. Prompt injection attacks
            3. Model extraction attempts
            4. Jailbreak techniques
            Common defenses involve input validation and robust model training.
            """,
            "source": "default_adversarial",
            "title": "LLM Security"
        }
    ]

    # Create documents from default content
    from langchain_core.documents import Document
    docs = [
        Document(
            page_content=item["content"],
            metadata={"source": item["source"], "title": item["title"]}
        )
        for item in default_content
    ]
    
    return docs

# Update the existing document loading code
doc_list = load_documents()

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

docs_split = text_splitter.split_documents(doc_list) if doc_list else []

# Vector store setup
try:
    # Initialize embeddings with error handling
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Initialize vector store with default documents
    docs = load_documents()
    
    astra_vector_store = Cassandra(
        embedding=embeddings,
        session=None,
        keyspace=None,
        table_name="qa_mini_demo"
    )
    
    # Add documents to vector store
    astra_vector_store.add_documents(docs)
    
    # Create index wrapper and retriever
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    retriever = astra_vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    
except Exception as e:
    st.error(f"Error setting up knowledge base: {str(e)}")
    st.stop()

# LLM Routing
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(...)

llm = ChatGroq(groq_api_key=groq_api_key, model_name="meta-llama/llama-4-scout-17b-16e-instruct")
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """you are an expert at routing a user question to a vectorstore or wikipedia.
the vectorstore contains documents related to agents , prompts engineering , and adversial attacks.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""
route_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
question_router = route_prompt | structured_llm_router

# Wikipedia Setup
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# LangGraph pipeline
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def retrieve(state):
    question = state["question"]
    document = retriever.invoke(question)
    return {"document": document, "question": question}

def wiki_search(state):
    question = state["question"]
    docs = wiki.invoke({"query": question})
    return {"documents": [docs], "question": question}

def route_question(state):
    question = state["question"]
    source = question_router.invoke({"question": question})
    return "wiki_search" if source.datasource == "wiki_search" else "vectorstore"

workflow = StateGraph(GraphState)
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("vectorstore", retrieve)
workflow.add_conditional_edges(START, route_question, {
    "wiki_search": "wiki_search",
    "vectorstore": "vectorstore"
})
workflow.add_edge("wiki_search", END)
workflow.add_edge("vectorstore", END)
app = workflow.compile()

# Streamlit UI
def main():
    st.title("AI Knowledge Assistant")
    st.write("Ask me anything about AI agents, prompt engineering, adversarial attacks, or general knowledge!")
    
    # User input
    user_question = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Processing your question..."):
                # Run the workflow
                result = app.invoke({
                    "question": user_question,
                    "generation": "",
                    "documents": []
                })
                
                # Display results
                st.subheader("Answer:")
                if "document" in result:
                    # Vector store results
                    for doc in result["document"]:
                        st.write(doc.page_content)
                        st.markdown("---")
                elif "documents" in result:
                    # Wikipedia results
                    st.write(result["documents"][0])

if __name__ == "__main__":
    main()
