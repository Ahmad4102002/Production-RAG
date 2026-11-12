from dotenv import load_dotenv
from importlib.metadata import version
import os 

load_dotenv()
KNOWLEDGE_BASE = os.getenv("KNOWLEDGE_BASE")

from langchain_core import __version__ as core_version
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from openai import OpenAI
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
from langchain_core.documents import Document
from langchain_chroma import Chroma 
import tempfile 
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import numpy as np
from langchain_ollama import OllamaEmbeddings

def main():

    # creating temporary vector store instance

    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":2})

    data = vector_store.get(include=["embeddings", "documents", "metadatas"])

    print("Number of chunks:", len(data["ids"]))

    for i in range(len(data["ids"])):
        print("\n--- Chunk", i, "---")
        print("ID:", data["ids"][i])
        print("Text:", data["documents"][i])
        print("Metadata:", data["metadatas"][i])
        print("Vector length:", len(data["embeddings"][i]))
        print("First 5 numbers:", data["embeddings"][i][:5])
        
    #initialising embedding model instance 
    #     
    llm = init_chat_model(
    model="claude-haiku-5-5",
    model_provider="anthropic",
    )

    # creatign system prompt

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are a helpful assistant that answers questions using only the "
        "context provided below.\n\n"
        "Rules:\n"
        "- Base your answer strictly on the context. Do not use outside knowledge.\n"
        "- If the context does not contain the answer, say \"I couldn't find that "
        "in the provided documents.\" Do not guess.\n"
        "- Be clear and concise. Use bullet points for lists or steps.\n"
        "- If the context contains conflicting information, point it out.\n\n"
        "Context:\n{context}"),
        ("human", "{question}"),
    ])

    # rag chain syntax
    rag_chain = (
        {"context": retriever | format_docs, "question" : RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is LangChain",
        "who created LangChain",
        "Did jenson hwang get fired from Nvidia"
    ]

    print("BASIC RAG DEMO STARTING")

    for each in questions:
        
        answer = rag_chain.invoke(each)
        print("QUESTION: " + each)
        print("ANSWER: " + answer)




def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def create_kb():
    # model instance that will be passed
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    #checkign vector norm
    print("LINE NO 102")
    v = np.array(embedding_model.embed_query("What is LangChain"))
    print(np.linalg.norm(v))

    
    # splitting the doc
    splitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE,
                   metadata={"source" : "langchain_knowledge_base.md"})

    chunks = splitter.split_documents([doc])

    # create vector store 

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=tempfile.mkdtemp(),
    )

    return vector_store

if __name__ == "__main__":
    main()

