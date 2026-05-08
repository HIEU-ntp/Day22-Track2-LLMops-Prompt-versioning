"""
Step 1 — LangSmith-instrumented RAG Pipeline

This script builds a simple RAG pipeline and instruments it with LangSmith tracing.
Every query is traced with input/output/latency visible in the LangSmith UI.
"""

import os
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

import config
from qa_pairs import QA_PAIRS

# Setup LangSmith tracing BEFORE importing other components
config.setup_langsmith()

print("=" * 70)
print("  Step 1: LangSmith RAG Pipeline")
print("=" * 70)


def build_vectorstore():
    """Load knowledge base, split into chunks, and index with FAISS"""
    print("\n📖 Building FAISS vectorstore...")
    
    # Read knowledge base
    text = config.KNOWLEDGE_BASE_PATH.read_text()
    print(f"   Loaded {len(text)} characters from knowledge base")
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"   Split into {len(chunks)} chunks")
    
    # Create embeddings and build vectorstore
    embeddings = config.get_embeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print(f"   ✅ FAISS vectorstore ready ({len(chunks)} chunks indexed)")
    
    return vectorstore


def build_rag_chain(vectorstore):
    """Build RAG chain: retriever → prompt → LLM → output parser"""
    print("\n🔗 Building RAG chain...")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Define RAG prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful AI assistant. Answer the user's question using ONLY "
            "the provided context. If the context does not contain sufficient information, "
            "say 'I don't have enough information in the knowledge base to answer that.'  "
            "\n\nContext:\n{context}"
        )),
        ("human", "{question}"),
    ])
    
    # Get LLM
    llm = config.get_llm()
    
    # Helper to format retrieved docs
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build chain using LCEL
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    print("   ✅ RAG chain ready")
    return chain, retriever


# Traced query function - this creates LangSmith traces
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """Run the RAG chain on a single question"""
    return chain.invoke(question)


def main():
    try:
        # Build vectorstore and chain
        vectorstore = build_vectorstore()
        chain, retriever = build_rag_chain(vectorstore)
        
        # Process all 50 questions
        print(f"\n🔍 Running {len(QA_PAIRS)} queries through RAG pipeline...")
        print("-" * 70)
        
        for i, qa in enumerate(QA_PAIRS, 1):
            question = qa["question"]
            try:
                answer = ask(chain, question)
                print(f"[{i:02d}/{len(QA_PAIRS)}] Q: {question[:65]}")
                print(f"            A: {answer[:70]}...")
                
                if i % 5 == 0:
                    print(f"    ({i} queries processed)")
            except Exception as e:
                print(f"[{i:02d}] ❌ Error: {str(e)[:60]}")
        
        print("-" * 70)
        print(f"\n✅ Step 1 Complete: {len(QA_PAIRS)} traces sent to LangSmith")
        print(f"   Project: {config.LANGSMITH_PROJECT}")
        print(f"   View traces at: https://smith.langchain.com")
        
    except Exception as e:
        print(f"\n❌ Error in Step 1: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
