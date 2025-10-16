"""
Evaluation Module
Evaluates classification performance with hierarchical metrics
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict


class ProductEvaluator:
    """Evaluates retrieval performance with hierarchical metrics"""
    
    def __init__(self, catalog_df, classifier):
        self.catalog_df = catalog_df
        self.classifier = classifier
        
        # Build lookups
        self.code_to_hierarchy = {}
        self.name_to_code = {}
        
        for _, row in catalog_df.iterrows():
            code = row['Commodity Code']
            name = row['Commodity Name'].strip().lower()
            
            self.code_to_hierarchy[code] = {
                'Segment Name': row['Segment Name'],
                'Family Name': row['Family Name'],
                'Class Name': row['Class Name'],
                'Commodity Name': row['Commodity Name']
            }
            
            self.name_to_code[name] = code
        
        print(f"Evaluator initialized")
    
    def evaluate(self, products_df, top_k_list: List[int] = [1, 5, 10]) -> Dict:
        """
        Evaluate classification performance
        
        Args:
            products_df: Test products with ground truth
            top_k_list: List of K values to evaluate
        
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = defaultdict(int)
        precision_at_k = {k: [] for k in top_k_list}
        recall_at_k = {k: [] for k in top_k_list}
        
        segment_hits = {k: 0 for k in top_k_list}
        family_hits = {k: 0 for k in top_k_list}
        class_hits = {k: 0 for k in top_k_list}
        
        total = 0
        skipped = 0
        
        max_k = max(top_k_list)
        
        for idx, row in products_df.iterrows():
            description = row.get('Original Description', '')
            true_commodity_name = row.get('UNSPSC Commodity Name', '').strip().lower()
            
            if not description or not true_commodity_name:
                skipped += 1
                continue
            
            # Get true commodity code
            true_commodity_code = self.name_to_code.get(true_commodity_name)
            if not true_commodity_code:
                skipped += 1
                continue
            
            total += 1
            
            # Get true hierarchy
            true_hierarchy = self.code_to_hierarchy.get(true_commodity_code, {})
            true_segment = true_hierarchy.get('Segment Name', '')
            true_family = true_hierarchy.get('Family Name', '')
            true_class = true_hierarchy.get('Class Name', '')
            
            # Get predictions
            predictions = self.classifier.classify_product(
                description, 
                top_k=max_k
            )
            
            pred_codes = [p['commodity_code'] for p in predictions]
            pred_names = [p['commodity_name'].strip().lower() for p in predictions]
            
            # Evaluate for each K
            for k in top_k_list:
                top_k_codes = pred_codes[:k]
                top_k_names = pred_names[:k]
                
                # Commodity match
                commodity_match = (
                    true_commodity_code in top_k_codes or
                    true_commodity_name in top_k_names
                )
                
                metrics[f'top{k}'] += int(commodity_match)
                precision_at_k[k].append(int(commodity_match) / k)
                recall_at_k[k].append(int(commodity_match))
                
                # Hierarchy matching
                segment_match = False
                family_match = False
                class_match = False
                
                for pred_code in top_k_codes:
                    pred_hierarchy = self.code_to_hierarchy.get(pred_code, {})
                    
                    if pred_hierarchy.get('Segment Name') == true_segment:
                        segment_match = True
                    if pred_hierarchy.get('Family Name') == true_family:
                        family_match = True
                    if pred_hierarchy.get('Class Name') == true_class:
                        class_match = True
                
                segment_hits[k] += int(segment_match)
                family_hits[k] += int(family_match)
                class_hits[k] += int(class_match)
            
            if (total % 25) == 0:
                print(f"  Evaluated {total} products...")
        
        # Calculate final metrics
        results = {}
        for k in top_k_list:
            results[f"Top-{k} Accuracy"] = (metrics[f'top{k}'] / total) * 100
            results[f"Precision@{k}"] = np.mean(precision_at_k[k]) * 100
            results[f"Recall@{k}"] = np.mean(recall_at_k[k]) * 100
            results[f"Segment Acc @{k}"] = (segment_hits[k] / total) * 100
            results[f"Family Acc @{k}"] = (family_hits[k] / total) * 100
            results[f"Class Acc @{k}"] = (class_hits[k] / total) * 100
        
        results["Total Evaluated"] = total
        results["Skipped"] = skipped
        
        return results
    
    def print_results(self, results: Dict):
        """Pretty print evaluation results"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                if metric in ["Total Evaluated", "Skipped"]:
                    print(f"{metric:25s}: {value}")
                else:
                    print(f"{metric:25s}: {value:6.2f}%")
        
        print("="*70 + "\n")