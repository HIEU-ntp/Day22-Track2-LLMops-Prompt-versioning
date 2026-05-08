"""
Step 2 — Prompt Hub & A/B Routing

This script demonstrates Prompt Hub integration and deterministic A/B routing.
Two prompt versions are pushed to LangSmith Prompt Hub and queries are routed
deterministically based on request ID hashing.
"""

import hashlib
import os
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable, Client

import config
from qa_pairs import QA_PAIRS

# Setup LangSmith
config.setup_langsmith()

print("=" * 70)
print("  Step 2: Prompt Hub A/B Routing")
print("=" * 70)

# Prompt names for Hub
PROMPT_V1_NAME = "day22-rag-prompt-v1-concise"
PROMPT_V2_NAME = "day22-rag-prompt-v2-structured"


def get_prompts():
    """Define two distinct system prompts"""
    
    # V1: Concise answers (2-4 sentences)
    system_v1 = (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the provided context. Keep answers concise (2-4 sentences). "
        "If the context lacks sufficient information, say: 'I don't have enough information.'\n\n"
        "Context:\n{context}"
    )
    prompt_v1 = ChatPromptTemplate.from_messages([
        ("system", system_v1),
        ("human", "{question}"),
    ])
    
    # V2: Structured expert answers (3-5 sentences with reasoning)
    system_v2 = (
        "You are an expert AI assistant. Provide structured, accurate answers with reasoning.\n\n"
        "Instructions:\n"
        "1. Read the context carefully.\n"
        "2. Identify key facts relevant to the question.\n"
        "3. Provide a clear, well-organized answer (3-5 sentences).\n"
        "4. Explicitly state if context lacks sufficient information.\n\n"
        "Context:\n{context}"
    )
    prompt_v2 = ChatPromptTemplate.from_messages([
        ("system", system_v2),
        ("human", "{question}"),
    ])
    
    return {
        PROMPT_V1_NAME: (prompt_v1, "Concise answers (2-4 sentences)"),
        PROMPT_V2_NAME: (prompt_v2, "Structured expert answers (3-5 sentences)"),
    }


def push_prompts_to_hub(client, prompts):
    """Push both prompt versions to LangSmith Prompt Hub"""
    print("\n📤 Pushing prompts to LangSmith Prompt Hub...")
    
    pushed = {}
    for name, (template, desc) in prompts.items():
        try:
            url = client.push_prompt(name, object=template, description=desc)
            pushed[name] = template
            print(f"   ✅ Pushed '{name}'")
        except Exception as e:
            # If push fails, use local version
            print(f"   ⚠️  Failed to push '{name}': {str(e)[:60]}")
            pushed[name] = template
    
    return pushed


def pull_prompts_from_hub(client, prompts):
    """Pull prompts from Hub, fallback to local if unavailable"""
    print("\n📥 Pulling prompts from LangSmith Prompt Hub...")
    
    pulled = {}
    for name in prompts.keys():
        try:
            template = client.pull_prompt(name)
            pulled[name] = template
            print(f"   ✅ Pulled '{name}' from Hub")
        except Exception as e:
            # Fallback to local version
            pulled[name] = prompts[name][0]
            print(f"   ℹ️  Using local fallback for '{name}'")
    
    return pulled


def get_prompt_version(request_id: str) -> str:
    """Deterministically route to V1 or V2 based on request ID hash"""
    # Hash the request_id and convert to integer
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    
    # Even → V1, Odd → V2
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


def build_vectorstore():
    """Reuse from step 1"""
    print("\n📖 Building FAISS vectorstore...")
    text = config.KNOWLEDGE_BASE_PATH.read_text()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    embeddings = config.get_embeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print(f"   ✅ Vectorstore ready ({len(chunks)} chunks)")
    
    return vectorstore


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """Run RAG with given prompt version"""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })
    
    return {"question": question, "answer": answer, "version": version}


def main():
    try:
        # Get LangSmith client
        client = Client(api_key=config.LANGSMITH_API_KEY)
        
        # Get prompts
        prompts = get_prompts()
        
        # Push to Hub
        pushed = push_prompts_to_hub(client, prompts)
        
        # Pull from Hub
        pulled = pull_prompts_from_hub(client, prompts)
        
        # Build vectorstore and retriever
        vectorstore = build_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = config.get_llm()
        
        # Process queries with A/B routing
        print(f"\n🔀 Running {len(QA_PAIRS)} queries with A/B routing...")
        print("-" * 70)
        
        v1_count = 0
        v2_count = 0
        
        for i, qa in enumerate(QA_PAIRS, 1):
            question = qa["question"]
            request_id = f"req-{i:04d}"
            
            # Deterministic routing
            version_key = get_prompt_version(request_id)
            version_label = "v1" if version_key == PROMPT_V1_NAME else "v2"
            prompt = pulled[version_key]
            
            # Track routing
            if version_label == "v1":
                v1_count += 1
            else:
                v2_count += 1
            
            try:
                result = ask_ab(retriever, llm, prompt, question, version_label)
                print(f"[{i:02d}/{len(QA_PAIRS)}] [prompt-{version_label}] {question[:50]}...")
            except Exception as e:
                print(f"[{i:02d}] ❌ Error: {str(e)[:50]}")
        
        print("-" * 70)
        print(f"\n✅ Step 2 Complete: {len(QA_PAIRS)} A/B routed queries")
        print(f"   V1 (concise):         {v1_count} queries")
        print(f"   V2 (structured):      {v2_count} queries")
        print(f"   Project:              {config.LANGSMITH_PROJECT}")
        print(f"   View in Prompt Hub:   https://smith.langchain.com/hub")
        
    except Exception as e:
        print(f"\n❌ Error in Step 2: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
