
"""
Product Classification Module
Classifies products to UNSPSC codes with Top-K predictions
"""


import numpy as np
from typing import List, Dict
from pathlib import Path
import csv


class ProductClassifier:
    """Classifies product descriptions to UNSPSC commodities"""
    
    def __init__(self, retriever, reranker, merger, hierarchical_router, 
                 feature_extractor=None):
        self.retriever = retriever
        self.reranker = reranker
        self.merger = merger
        self.hierarchical_router = hierarchical_router
        self.feature_extractor = feature_extractor
        self.commodity_features = {}
    
    def precompute_features(self, catalog_df):
        """Precompute feature vectors for all commodities"""
        if not self.feature_extractor:
            return
        
        print("Computing commodity feature vectors...")
        for idx, row in catalog_df.iterrows():
            commodity_name = row['Commodity Name']
            self.commodity_features[commodity_name] = self.feature_extractor.extract_all(
                row['corpus']
            )
        
        print(f"Precomputed features for {len(self.commodity_features)} commodities")
    
    def classify_product(self, description: str, top_k: int = 10, 
                        retrieve_k: int = 50, rerank_k: int = 20) -> List[Dict]:
        """
        Classify a product description to top-k UNSPSC codes
        
        Args:
            description: Product description text
            top_k: Number of top predictions to return
            retrieve_k: Number of candidates to retrieve
            rerank_k: Number of candidates to rerank
        
        Returns:
            List of top-k predictions with scores
        """
        # Step 1: Hybrid retrieval
        candidates = self.retriever.retrieve(description, top_k=retrieve_k)
        
        if not candidates:
            return []
        
        # Step 2: Hierarchical routing
        hierarchy_info = self.hierarchical_router.route_from_documents(
            candidates, top_k=5
        )
        
        # Step 3: Cross-encoder reranking
        candidates = self.reranker.rerank(description, candidates, top_n=rerank_k)
        
        # Step 4: Score merging
        candidates = self.merger.merge(candidates)
        
        # Step 5: Apply hierarchy boost
        candidates = self.hierarchical_router.apply_hierarchy_boost(
            candidates, hierarchy_info, boost_factor=1.3
        )
        
        # Step 6: Feature similarity boost (if available)
        if self.feature_extractor and self.commodity_features:
            product_features = self.feature_extractor.extract_all(description)
            
            for candidate in candidates:
                commodity_name = candidate.get('commodity_name', '')
                commodity_feats = self.commodity_features.get(
                    commodity_name, 
                    np.zeros_like(product_features)
                )
                
                # Cosine-like similarity
                feature_sim = 1 - np.linalg.norm(product_features - commodity_feats) / (
                    1 + np.linalg.norm(product_features) + np.linalg.norm(commodity_feats)
                )
                
                # Final score: 70% retrieval + 30% features
                candidate['final_boosted_score'] = (
                    0.7 * candidate['boosted_score'] + 0.3 * feature_sim
                )
        else:
            for candidate in candidates:
                candidate['final_boosted_score'] = candidate['boosted_score']
        
        # Sort by final score
        candidates = sorted(
            candidates, 
            key=lambda x: x['final_boosted_score'], 
            reverse=True
        )
        
        # Return top-k with clean format
        results = []
        for i, candidate in enumerate(candidates[:top_k], 1):
            results.append({
                'rank': i,
                'commodity_code': int(candidate['commodity_code']),
                'commodity_name': candidate['commodity_name'],
                'score': float(candidate['final_boosted_score']),
                'segment': candidate['metadata']['Segment Name'],
                'family': candidate['metadata']['Family Name'],
                'class': candidate['metadata']['Class Name'],
                'in_hierarchy': candidate['in_hierarchy']
            })
        
        return results
    
    def classify_batch(self, products_df, top_k: int = 10) -> List[Dict]:
        """
        Classify multiple products
        
        Args:
            products_df: DataFrame with 'Original Description' column
            top_k: Number of predictions per product
        
        Returns:
            List of classification results
        """
        results = []
        
        for idx, row in products_df.iterrows():
            description = row['Original Description']
            
            predictions = self.classify_product(description, top_k=top_k)
            
            result = {
                'product_id': idx,
                'description': description,
                'top_k_predictions': predictions
            }
            
            # Add ground truth if available
            if 'UNSPSC Commodity Name' in row:
                result['ground_truth'] = row['UNSPSC Commodity Name']
            
            results.append(result)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1} products...")
        
        return results
    

    def save_predictions(self, results: List[Dict], output_file: str = "output/unspsc_candidates_dataset.csv"):
        """Save classification results to CSV"""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        if not results:
            print("No results to save")
            return output_path
        
        # Get all unique keys from the results to use as CSV headers
        fieldnames = list(results[0].keys())
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Saved predictions to {output_path}")
        return output_path