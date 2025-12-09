"""
DELIVERABLE 4: Testing & Evaluation
Test suite for evaluating RAG chatbot performance
"""
import time
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

from src.retriever import RAGRetriever
from src.generator import ResponseGenerator
from src.ingestion import DocumentIngestion


class RAGEvaluator:
    """Evaluate RAG chatbot performance"""
    
    def __init__(self):
        """Initialize evaluator with retriever and generator"""
        print("🔧 Initializing evaluator...")
        self.ingestion = DocumentIngestion()
        self.retriever = RAGRetriever()
        self.generator = ResponseGenerator()
        
        # Test questions (customize based on your domain)
        self.test_questions = [
            {
                "question": "What is machine learning?",
                "expected_topic": "machine learning definition",
                "category": "definition"
            },
            {
                "question": "How does deep learning work?",
                "expected_topic": "deep learning process",
                "category": "explanation"
            },
            {
                "question": "What are neural networks?",
                "expected_topic": "neural networks",
                "category": "definition"
            },
            {
                "question": "Explain supervised learning",
                "expected_topic": "supervised learning",
                "category": "explanation"
            },
            {
                "question": "What is unsupervised learning?",
                "expected_topic": "unsupervised learning",
                "category": "definition"
            },
            {
                "question": "How do you train a model?",
                "expected_topic": "model training",
                "category": "process"
            },
            {
                "question": "What is overfitting?",
                "expected_topic": "overfitting",
                "category": "concept"
            },
            {
                "question": "Explain gradient descent",
                "expected_topic": "optimization",
                "category": "algorithm"
            },
            {
                "question": "What are the types of machine learning?",
                "expected_topic": "ML types",
                "category": "taxonomy"
            },
            {
                "question": "How does reinforcement learning work?",
                "expected_topic": "reinforcement learning",
                "category": "explanation"
            },
            {
                "question": "What is natural language processing?",
                "expected_topic": "NLP",
                "category": "definition"
            },
            {
                "question": "Explain computer vision",
                "expected_topic": "computer vision",
                "category": "definition"
            },
            {
                "question": "What are hyperparameters?",
                "expected_topic": "hyperparameters",
                "category": "concept"
            },
            {
                "question": "How do you evaluate a model?",
                "expected_topic": "model evaluation",
                "category": "process"
            },
            {
                "question": "What is cross-validation?",
                "expected_topic": "validation techniques",
                "category": "method"
            },
            {
                "question": "Explain feature engineering",
                "expected_topic": "feature engineering",
                "category": "process"
            },
            {
                "question": "What is a convolutional neural network?",
                "expected_topic": "CNN",
                "category": "architecture"
            },
            {
                "question": "How does transfer learning work?",
                "expected_topic": "transfer learning",
                "category": "technique"
            },
            {
                "question": "What is the difference between AI and ML?",
                "expected_topic": "AI vs ML",
                "category": "comparison"
            },
            {
                "question": "Explain backpropagation",
                "expected_topic": "backpropagation",
                "category": "algorithm"
            }
        ]
    
    def evaluate_single_query(
        self,
        question: str,
        expected_topic: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate a single query
        
        Args:
            question: Test question
            expected_topic: Expected topic/theme
            top_k: Number of chunks to retrieve
            
        Returns:
            Evaluation results dictionary
        """
        print(f"\n📝 Testing: {question}")
        
        # Measure retrieval time
        start_time = time.time()
        chunks, avg_similarity = self.retriever.retrieve_with_scores(question, top_k=top_k)
        retrieval_time = time.time() - start_time
        
        # Measure generation time
        start_time = time.time()
        result = self.generator.generate(question, chunks)
        generation_time = time.time() - start_time
        
        total_time = retrieval_time + generation_time
        
        # Evaluate retrieval
        retrieved = len(chunks) > 0
        sources_found = result["sources"] if result["sources"] else []
        
        # Manual relevance check (would need human evaluation in production)
        # For now, we consider it relevant if chunks were retrieved
        relevant = retrieved and avg_similarity > 0.5
        
        evaluation = {
            "question": question,
            "expected_topic": expected_topic,
            "answer": result["answer"][:200] + "..." if len(result["answer"]) > 200 else result["answer"],
            "full_answer": result["answer"],
            "retrieved_chunks": len(chunks),
            "sources": ", ".join(sources_found) if sources_found else "None",
            "avg_similarity": round(avg_similarity, 3),
            "confidence": result["confidence"],
            "relevant": "Yes" if relevant else "No",
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3),
            "total_time": round(total_time, 3),
            "status": result["status"]
        }
        
        print(f"   ✓ Retrieved: {len(chunks)} chunks")
        print(f"   ✓ Similarity: {avg_similarity:.3f}")
        print(f"   ✓ Time: {total_time:.2f}s")
        
        return evaluation
    
    def run_evaluation(self, top_k: int = 3) -> pd.DataFrame:
        """
        Run evaluation on all test questions
        
        Args:
            top_k: Number of chunks to retrieve
            
        Returns:
            DataFrame with evaluation results
        """
        print("\n" + "="*70)
        print("🧪 STARTING RAG CHATBOT EVALUATION")
        print("="*70)
        
        # Check collection status
        status = self.retriever.check_collection_status()
        print(f"\n📊 Collection Status: {status['message']}")
        
        if status["document_count"] == 0:
            print("\n❌ ERROR: No documents in collection!")
            print("Please process documents before running evaluation.")
            return pd.DataFrame()
        
        print(f"\n📋 Testing {len(self.test_questions)} questions...")
        
        # Run evaluation
        results = []
        for idx, test_case in enumerate(self.test_questions, 1):
            print(f"\n[{idx}/{len(self.test_questions)}]", end=" ")
            
            try:
                result = self.evaluate_single_query(
                    question=test_case["question"],
                    expected_topic=test_case["expected_topic"],
                    top_k=top_k
                )
                result["category"] = test_case["category"]
                results.append(result)
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                results.append({
                    "question": test_case["question"],
                    "expected_topic": test_case["expected_topic"],
                    "category": test_case["category"],
                    "answer": f"Error: {str(e)}",
                    "relevant": "Error",
                    "status": "error"
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Calculate metrics
        self.print_summary(df)
        
        return df
    
    def print_summary(self, df: pd.DataFrame):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print("📊 EVALUATION SUMMARY")
        print("="*70)
        
        # Overall metrics
        total_questions = len(df)
        relevant_count = len(df[df["relevant"] == "Yes"])
        accuracy = (relevant_count / total_questions * 100) if total_questions > 0 else 0
        
        print(f"\n✅ Overall Performance:")
        print(f"   • Total Questions: {total_questions}")
        print(f"   • Relevant Answers: {relevant_count}")
        print(f"   • Accuracy: {accuracy:.1f}%")
        
        # Time metrics
        if "total_time" in df.columns:
            avg_time = df["total_time"].mean()
            max_time = df["total_time"].max()
            min_time = df["total_time"].min()
            
            print(f"\n⏱️  Response Time:")
            print(f"   • Average: {avg_time:.2f}s")
            print(f"   • Minimum: {min_time:.2f}s")
            print(f"   • Maximum: {max_time:.2f}s")
        
        # Retrieval metrics
        if "avg_similarity" in df.columns:
            avg_similarity = df["avg_similarity"].mean()
            print(f"\n🎯 Retrieval Quality:")
            print(f"   • Average Similarity: {avg_similarity:.3f}")
        
        # Status breakdown
        if "status" in df.columns:
            success_count = len(df[df["status"] == "success"])
            error_count = len(df[df["status"] == "error"])
            
            print(f"\n📈 Status:")
            print(f"   • Success: {success_count}")
            print(f"   • Errors: {error_count}")
        
        # Failure cases
        failed_df = df[df["relevant"] == "No"]
        if len(failed_df) > 0:
            print(f"\n❌ Failed Queries ({len(failed_df)}):")
            for idx, row in failed_df.iterrows():
                print(f"   • {row['question']}")
        
        print("\n" + "="*70)
    
    def save_results(self, df: pd.DataFrame, output_path: str = "./tests/test_results.xlsx"):
        """Save evaluation results to Excel"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main results
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # Summary statistics
            summary_data = {
                "Metric": [
                    "Total Questions",
                    "Relevant Answers",
                    "Accuracy (%)",
                    "Average Response Time (s)",
                    "Average Similarity"
                ],
                "Value": [
                    len(df),
                    len(df[df["relevant"] == "Yes"]),
                    round(len(df[df["relevant"] == "Yes"]) / len(df) * 100, 2) if len(df) > 0 else 0,
                    round(df["total_time"].mean(), 3) if "total_time" in df.columns else "N/A",
                    round(df["avg_similarity"].mean(), 3) if "avg_similarity" in df.columns else "N/A"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Failed queries
            failed_df = df[df["relevant"] == "No"][["question", "expected_topic", "answer"]]
            failed_df.to_excel(writer, sheet_name='Failed_Queries', index=False)
        
        print(f"\n💾 Results saved to: {output_path}")
    
    def analyze_failures(self, df: pd.DataFrame):
        """Analyze failure cases in detail"""
        failed_df = df[df["relevant"] == "No"]
        
        if len(failed_df) == 0:
            print("\n🎉 No failures to analyze!")
            return
        
        print("\n" + "="*70)
        print(f"🔍 FAILURE ANALYSIS ({len(failed_df)} cases)")
        print("="*70)
        
        for idx, row in failed_df.iterrows():
            print(f"\n❌ Failed Query #{idx + 1}")
            print(f"   Question: {row['question']}")
            print(f"   Expected Topic: {row['expected_topic']}")
            print(f"   Retrieved Chunks: {row.get('retrieved_chunks', 'N/A')}")
            print(f"   Similarity: {row.get('avg_similarity', 'N/A')}")
            print(f"   Answer: {row['answer'][:150]}...")
            
            # Suggest improvements
            if row.get('retrieved_chunks', 0) == 0:
                print("   💡 Suggestion: Add more documents on this topic")
            elif row.get('avg_similarity', 0) < 0.5:
                print("   💡 Suggestion: Improve query or document chunking")


def main():
    """Run evaluation"""
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Run evaluation
    results_df = evaluator.run_evaluation(top_k=3)
    
    if not results_df.empty:
        # Save results
        evaluator.save_results(results_df)
        
        # Analyze failures
        evaluator.analyze_failures(results_df)
        
        print("\n✅ Evaluation complete!")
    else:
        print("\n❌ Evaluation failed. Please check collection status.")


if __name__ == "__main__":
    main()
