# pipeline.py

import os
os.environ['CQLENG_ALLOW_SCHEMA_MANAGEMENT'] = '1'

import eventlet
eventlet.monkey_patch()

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
docs = [
    WebBaseLoader(url).load()
    for url in [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
    ]
]
doc_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
docs_split = text_splitter.split_documents(doc_list)

# Vector store setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
astra_vector_store = Cassandra(embedding=embeddings, session=None, keyspace=None, table_name="qa_mini_demo")
astra_vector_store.add_documents(docs_split)
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
retreiver = astra_vector_store.as_retriever()

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
    document = retreiver.invoke(question)
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
