
"""
Main Pipeline
Orchestrates the complete UNSPSC classification workflow
"""

import warnings
warnings.filterwarnings('ignore')

from src.DataLoader import DataLoader
from src.TextPreprocessor import TextCleaner, ProductFeatureExtractor
from src.RetrievalSystem import VectorStoreBuilder, HybridRetriever, CrossEncoderReranker, ScoreMerger
from src.HierarchicalRouter import DocumentInformedHierarchicalRouter
from src.TrainDataGenerator import TrainingDataGenerator
from src.ProductClassifier import ProductClassifier
from src.Evaluator import ProductEvaluator


def main():
    print("\n" + "="*70)
    print("UNSPSC PRODUCT CLASSIFICATION PIPELINE")
    print("="*70 + "\n")
    
    # ============================================================================
    # STEP 1: Load Data
    # ============================================================================
    print("Step 1: Loading data...")
    loader = DataLoader(data_dir="data")
    catalog_df = loader.load_catalog()
    queries_df = loader.load_queries()
    products_df = loader.load_products()
    
    loader.validate_data(catalog_df, queries_df, products_df)
    
    # ============================================================================
    # STEP 2: Initialize Components
    # ============================================================================
    print("\n Step 2: Initializing components...")
    
    # Text processing
    cleaner = TextCleaner()
    feature_extractor = ProductFeatureExtractor()
    
    # Vector store
    vs_builder = VectorStoreBuilder(embedding_model="all-MiniLM-L6-v2")
    
    # Check if FAISS index exists, otherwise build it
    # try:
    #     # vectorstore = vs_builder.load_vectorstore("G:/Dylog_Internship_Assessments/src/faiss_index")
    #     vectorstore=vs_builder.load_vectorstore()
    # except:
    #     print("Building FAISS index (first time only)...")
    #     vectorstore = vs_builder.build_vectorstore(catalog_df, "faiss_index")

    vectorstore=vs_builder.load_vectorstore()
    
    # Retrieval system
    retriever = HybridRetriever(catalog_df, vectorstore, cleaner)
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    merger = ScoreMerger(alpha=0.4, beta=0.6)
    
    # Hierarchical router
    hierarchical_router = DocumentInformedHierarchicalRouter(catalog_df)
    
    # ============================================================================
    # STEP 3: Generate Training Data (Output 1)
    # ============================================================================
    print("\n Step 3: Generating training data...")
    
    training_generator = TrainingDataGenerator(catalog_df, queries_df, cleaner)
    triples_file, pairs_file = training_generator.save_training_data(output_dir="output")
    
    print(f"\n Output 1 Generated:")
    print(f"   - Triples: {triples_file}")
    print(f"   - Pairs: {pairs_file}")
    
    # ============================================================================
    # STEP 4: Initialize Product Classifier
    # ============================================================================
    print("\n Step 4: Initializing classifier...")
    
    classifier = ProductClassifier(
        retriever=retriever,
        reranker=reranker,
        merger=merger,
        hierarchical_router=hierarchical_router,
        feature_extractor=feature_extractor
    )
    
    # Precompute features
    classifier.precompute_features(catalog_df)
    
    # ============================================================================
    # STEP 5: Classify Products (Output 2)
    # ============================================================================
    print("\n Step 5: Classifying products...")
    
    # Classify all products
    classification_results = classifier.classify_batch(products_df, top_k=10)
    
    # Save predictions
    predictions_file = classifier.save_predictions(
        classification_results,
        # output_file="output/Output2_predictions.json"
    )
    
    print(f"\Output 2 Generated:")
    print(f"   - Predictions: {predictions_file}")
    
    # ============================================================================
    # STEP 6: Evaluate Performance
    # ============================================================================
    print("\n Step 6: Evaluating performance...")
    
    evaluator = ProductEvaluator(catalog_df, classifier)
    results = evaluator.evaluate(products_df, top_k_list=[1, 5, 10])
    
    evaluator.print_results(results)
    
    # ============================================================================
    # STEP 7: Sample Predictions
    # ============================================================================
    print("Sample Predictions (First 3 Products):\n")
    
    for i in range(min(3, len(classification_results))):
        result = classification_results[i]
        print(f"Product {i+1}:")
        print(f"  Description: {result['description'][:80]}...")
        print(f"  Ground Truth: {result.get('ground_truth', 'N/A')}")
        print(f"  Top 3 Predictions:")
        for pred in result['top_k_predictions'][:3]:
            print(f"    {pred['rank']}. {pred['commodity_name']} "
                  f"(Score: {pred['score']:.4f}, In Hierarchy: {pred['in_hierarchy']})")
        print()
    
    print("="*70)
    print(" PIPELINE COMPLETE")
    print("="*70)
    print("\n Output Files:")
    print("   1. output/Output1_triples.json  - Training triples")
    print("   2. output/Output1_pairs.json    - Training pairs")
    print("   3. output/Output2_predictions.json - Product classifications")
    print()


if __name__ == "__main__":
    main()