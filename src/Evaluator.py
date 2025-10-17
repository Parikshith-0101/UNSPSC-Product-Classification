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
    
    def evaluate(self, products_df, top_k_list: List[int] = [1, 5, 10], 
                 confidence_threshold: float = 0.3) -> Dict:
        """
        Evaluate classification performance
        
        Args:
            products_df: Test products with ground truth
            top_k_list: List of K values to evaluate
            confidence_threshold: Minimum confidence score (0.0-1.0) to include prediction
        
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
        filtered_count = 0  # Track how many predictions were filtered by threshold
        
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
            
            # Apply confidence thresholding
            original_pred_count = len(predictions)
            predictions = [p for p in predictions if p.get('confidence', 1.0) >= confidence_threshold]
            
            if len(predictions) < original_pred_count:
                filtered_count += (original_pred_count - len(predictions))
            
            # Handle case where all predictions filtered out
            if not predictions:
                # Add zero scores for this sample
                for k in top_k_list:
                    precision_at_k[k].append(0.0)
                    recall_at_k[k].append(0.0)
                continue
            
            pred_codes = [p['commodity_code'] for p in predictions]
            pred_names = [p['commodity_name'].strip().lower() for p in predictions]
            
            # Evaluate for each K
            for k in top_k_list:
                # Adjust k if fewer predictions than k after filtering
                effective_k = min(k, len(pred_codes))
                
                top_k_codes = pred_codes[:effective_k]
                top_k_names = pred_names[:effective_k]
                
                # Commodity match
                commodity_match = (
                    true_commodity_code in top_k_codes or
                    true_commodity_name in top_k_names
                )
                
                metrics[f'top{k}'] += int(commodity_match)
                
                # Precision: relevant items / k (or effective_k if fewer predictions)
                precision_at_k[k].append(int(commodity_match) / effective_k if effective_k > 0 else 0.0)
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
            results[f"Top-{k} Accuracy"] = (metrics[f'top{k}'] / total) * 100 if total > 0 else 0.0
            results[f"Precision@{k}"] = np.mean(precision_at_k[k]) * 100 if precision_at_k[k] else 0.0
            results[f"Recall@{k}"] = np.mean(recall_at_k[k]) * 100 if recall_at_k[k] else 0.0
            results[f"Segment Acc @{k}"] = (segment_hits[k] / total) * 100 if total > 0 else 0.0
            results[f"Family Acc @{k}"] = (family_hits[k] / total) * 100 if total > 0 else 0.0
            results[f"Class Acc @{k}"] = (class_hits[k] / total) * 100 if total > 0 else 0.0
        
        results["Total Evaluated"] = total
        results["Skipped"] = skipped
        results["Confidence Threshold"] = confidence_threshold
        results["Predictions Filtered"] = filtered_count
        
        return results
    
    def print_results(self, results: Dict):
        """Pretty print evaluation results"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                if metric in ["Total Evaluated", "Skipped", "Predictions Filtered"]:
                    print(f"{metric:25s}: {value}")
                elif metric == "Confidence Threshold":
                    print(f"{metric:25s}: {value:.2f}")
                else:
                    print(f"{metric:25s}: {value:6.2f}%")
        
        print("="*70 + "\n")