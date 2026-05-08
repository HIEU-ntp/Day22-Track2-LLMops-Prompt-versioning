"""
Step 3 — RAGAS Evaluation

This script evaluates both prompt versions (V1: concise, V2: structured)
using RAGAS metrics. It compares faithfulness, answer_relevancy, context_recall,
and context_precision.

⏰ NOTE: This step takes 15-25 minutes due to LLM-based metric computations.
"""

import json
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

import config
from qa_pairs import QA_PAIRS

# Setup LangSmith
config.setup_langsmith()

print("=" * 70)
print("  Step 3: RAGAS Evaluation")
print("=" * 70)
print("\n⏰ This step takes 15-25 minutes. Please be patient...")


def get_prompts():
    """Define the two prompt versions"""
    system_v1 = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "Keep answers concise (2-4 sentences). If insufficient, say 'I don't have enough information.'\n\n"
        "Context:\n{context}"
    )
    prompt_v1 = ChatPromptTemplate.from_messages([
        ("system", system_v1),
        ("human", "{question}"),
    ])
    
    system_v2 = (
        "You are an expert AI assistant. Provide structured, accurate answers with reasoning.\n\n"
        "Instructions:\n"
        "1. Read the context carefully.\n"
        "2. Identify key facts.\n"
        "3. Provide clear, organized answer (3-5 sentences).\n"
        "4. State if context is insufficient.\n\n"
        "Context:\n{context}"
    )
    prompt_v2 = ChatPromptTemplate.from_messages([
        ("system", system_v2),
        ("human", "{question}"),
    ])
    
    return {"v1": prompt_v1, "v2": prompt_v2}


def build_vectorstore():
    """Build FAISS vectorstore from knowledge base"""
    print("\n📖 Building FAISS vectorstore...")
    text = config.KNOWLEDGE_BASE_PATH.read_text()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    embeddings = config.get_embeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    print(f"   ✅ Vectorstore ready ({len(chunks)} chunks)")
    return vectorstore


def run_rag(retriever, llm, prompt, question: str) -> dict:
    """Run RAG and capture answer + contexts"""
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    context_str = "\n\n".join(contexts)
    
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context_str,
        "question": question
    })
    
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompts, version: str) -> list:
    """Run all 50 questions through a prompt version"""
    print(f"\n🔄 Running {len(QA_PAIRS)} queries with prompt {version}...")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = config.get_llm()
    prompt = prompts[version]
    
    results = []
    for i, qa in enumerate(QA_PAIRS, 1):
        try:
            rag_output = run_rag(retriever, llm, prompt, qa["question"])
            results.append({
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": rag_output["answer"],
                "contexts": rag_output["contexts"],
            })
            
            if i % 10 == 0:
                print(f"   [{i:2d}/50]")
        except Exception as e:
            print(f"   [{i:2d}] ❌ Error: {str(e)[:40]}")
            # Use empty answer on error
            results.append({
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": "",
                "contexts": [],
            })
    
    print(f"   ✅ Collected {len(results)} outputs")
    return results


def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """Convert RAG results to RAGAS EvaluationDataset"""
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


def run_ragas_eval(rag_results: list, version: str) -> dict:
    """Evaluate RAG outputs with 4 RAGAS metrics"""
    print(f"\n📐 Evaluating prompt {version} (this takes 5-15 min)...")
    
    dataset = build_ragas_dataset(rag_results)
    
    # Create LLM and embeddings for RAGAS to use
    llm_eval = config.get_llm()
    emb_eval = config.get_embeddings()
    
    # Run evaluation
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=llm_eval,
            embeddings=emb_eval,
        )
        
        # Extract and compute mean scores
        scores = {}
        for metric_name in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            raw_scores = result[metric_name]
            # Filter None values
            valid_scores = [v for v in raw_scores if v is not None]
            mean_score = float(np.mean(valid_scores)) if valid_scores else 0.0
            scores[metric_name] = mean_score
            
            # Mark if faithfulness meets target
            if metric_name == "faithfulness" and mean_score >= 0.8:
                print(f"   ⭐ {metric_name:25s}: {mean_score:.4f} ✅")
            else:
                print(f"   • {metric_name:25s}: {mean_score:.4f}")
        
        return scores
    except Exception as e:
        print(f"   ❌ Evaluation error: {e}")
        return {}


def main():
    try:
        # Build vectorstore and get prompts
        vectorstore = build_vectorstore()
        prompts = get_prompts()
        
        # Collect outputs for both versions
        print("\n" + "=" * 70)
        print("  Collecting RAG outputs for V1 and V2")
        print("=" * 70)
        
        v1_results = collect_rag_outputs(vectorstore, prompts, "v1")
        v2_results = collect_rag_outputs(vectorstore, prompts, "v2")
        
        # Run RAGAS evaluation
        print("\n" + "=" * 70)
        print("  Running RAGAS Evaluation")
        print("=" * 70)
        
        v1_scores = run_ragas_eval(v1_results, "v1")
        v2_scores = run_ragas_eval(v2_results, "v2")
        
        # Print comparison table
        print("\n" + "=" * 70)
        print("  Evaluation Results Comparison")
        print("=" * 70)
        print()
        print(f"{'Metric':<30} {'V1 (Concise)':<20} {'V2 (Structured)':<20}")
        print("-" * 70)
        
        for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            v1_score = v1_scores.get(metric, 0.0)
            v2_score = v2_scores.get(metric, 0.0)
            winner = "← V1" if v1_score > v2_score else "← V2" if v2_score > v1_score else "tie"
            print(f"{metric:<30} {v1_score:<20.4f} {v2_score:<20.4f} {winner}")
        
        # Check if target met
        best_faith = max(
            v1_scores.get("faithfulness", 0.0),
            v2_scores.get("faithfulness", 0.0)
        )
        
        if best_faith >= 0.8:
            print(f"\n✅ Target met: faithfulness ≥ 0.8 (best: {best_faith:.4f})")
        else:
            print(f"\n⚠️  Target not met: faithfulness < 0.8 (best: {best_faith:.4f})")
        
        # Save report
        report = {
            "timestamp": str(Path.cwd()),
            "v1_scores": v1_scores,
            "v2_scores": v2_scores,
            "target_faithfulness": 0.8,
            "target_met": best_faith >= 0.8,
        }
        
        report_path = config.RAGAS_REPORT_PATH
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Report saved to: {report_path}")
        print(f"\n✅ Step 3 Complete!")
        
    except Exception as e:
        print(f"\n❌ Error in Step 3: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
