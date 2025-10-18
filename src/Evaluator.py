# """
# Evaluation Module
# Evaluates classification performance with hierarchical metrics
# """

# import numpy as np
# from typing import Dict, List
# from collections import defaultdict


# class ProductEvaluator:
#     """Evaluates retrieval performance with hierarchical metrics"""
    
#     def __init__(self, catalog_df, classifier):
#         self.catalog_df = catalog_df
#         self.classifier = classifier
        
#         # Build lookups
#         self.code_to_hierarchy = {}
#         self.name_to_code = {}
        
#         for _, row in catalog_df.iterrows():
#             code = row['Commodity Code']
#             name = row['Commodity Name'].strip().lower()
            
#             self.code_to_hierarchy[code] = {
#                 'Segment Name': row['Segment Name'],
#                 'Family Name': row['Family Name'],
#                 'Class Name': row['Class Name'],
#                 'Commodity Name': row['Commodity Name']
#             }
            
#             self.name_to_code[name] = code
        
#         print(f"Evaluator initialized")
    
#     def evaluate(self, products_df, top_k_list: List[int] = [1, 5, 10], 
#                  confidence_threshold: float = 0.3) -> Dict:
#         """
#         Evaluate classification performance
        
#         Args:
#             products_df: Test products with ground truth
#             top_k_list: List of K values to evaluate
#             confidence_threshold: Minimum confidence score (0.0-1.0) to include prediction
        
#         Returns:
#             Dictionary of evaluation metrics
#         """
#         metrics = defaultdict(int)
#         precision_at_k = {k: [] for k in top_k_list}
#         recall_at_k = {k: [] for k in top_k_list}
        
#         segment_hits = {k: 0 for k in top_k_list}
#         family_hits = {k: 0 for k in top_k_list}
#         class_hits = {k: 0 for k in top_k_list}
        
#         total = 0
#         skipped = 0
#         filtered_count = 0  # Track how many predictions were filtered by threshold
        
#         max_k = max(top_k_list)
        
#         for idx, row in products_df.iterrows():
#             description = row.get('Original Description', '')
#             true_commodity_name = row.get('UNSPSC Commodity Name', '').strip().lower()
            
#             if not description or not true_commodity_name:
#                 skipped += 1
#                 continue
            
#             # Get true commodity code
#             true_commodity_code = self.name_to_code.get(true_commodity_name)
#             if not true_commodity_code:
#                 skipped += 1
#                 continue
            
#             total += 1
            
#             # Get true hierarchy
#             true_hierarchy = self.code_to_hierarchy.get(true_commodity_code, {})
#             true_segment = true_hierarchy.get('Segment Name', '')
#             true_family = true_hierarchy.get('Family Name', '')
#             true_class = true_hierarchy.get('Class Name', '')
            
#             # Get predictions
#             predictions = self.classifier.classify_product(
#                 description, 
#                 top_k=max_k
#             )
            
#             # Apply confidence thresholding
#             original_pred_count = len(predictions)
#             predictions = [p for p in predictions if p.get('confidence', 1.0) >= confidence_threshold]
            
#             if len(predictions) < original_pred_count:
#                 filtered_count += (original_pred_count - len(predictions))
            
#             # Handle case where all predictions filtered out
#             if not predictions:
#                 # Add zero scores for this sample
#                 for k in top_k_list:
#                     precision_at_k[k].append(0.0)
#                     recall_at_k[k].append(0.0)
#                 continue
            
#             pred_codes = [p['commodity_code'] for p in predictions]
#             pred_names = [p['commodity_name'].strip().lower() for p in predictions]
            
#             # Evaluate for each K
#             for k in top_k_list:
#                 # Adjust k if fewer predictions than k after filtering
#                 effective_k = min(k, len(pred_codes))
                
#                 top_k_codes = pred_codes[:effective_k]
#                 top_k_names = pred_names[:effective_k]
                
#                 # Commodity match
#                 commodity_match = (
#                     true_commodity_code in top_k_codes or
#                     true_commodity_name in top_k_names
#                 )
                
#                 metrics[f'top{k}'] += int(commodity_match)
                
#                 # Precision: relevant items / k (or effective_k if fewer predictions)
#                 precision_at_k[k].append(int(commodity_match) / effective_k if effective_k > 0 else 0.0)
#                 recall_at_k[k].append(int(commodity_match))
                
#                 # Hierarchy matching
#                 segment_match = False
#                 family_match = False
#                 class_match = False
                
#                 for pred_code in top_k_codes:
#                     pred_hierarchy = self.code_to_hierarchy.get(pred_code, {})
                    
#                     if pred_hierarchy.get('Segment Name') == true_segment:
#                         segment_match = True
#                     if pred_hierarchy.get('Family Name') == true_family:
#                         family_match = True
#                     if pred_hierarchy.get('Class Name') == true_class:
#                         class_match = True
                
#                 segment_hits[k] += int(segment_match)
#                 family_hits[k] += int(family_match)
#                 class_hits[k] += int(class_match)
            
#             if (total % 25) == 0:
#                 print(f"  Evaluated {total} products...")
        
#         # Calculate final metrics
#         results = {}
#         for k in top_k_list:
#             results[f"Top-{k} Accuracy"] = (metrics[f'top{k}'] / total) * 100 if total > 0 else 0.0
#             results[f"Precision@{k}"] = np.mean(precision_at_k[k]) * 100 if precision_at_k[k] else 0.0
#             results[f"Recall@{k}"] = np.mean(recall_at_k[k]) * 100 if recall_at_k[k] else 0.0
#             results[f"Segment Acc @{k}"] = (segment_hits[k] / total) * 100 if total > 0 else 0.0
#             results[f"Family Acc @{k}"] = (family_hits[k] / total) * 100 if total > 0 else 0.0
#             results[f"Class Acc @{k}"] = (class_hits[k] / total) * 100 if total > 0 else 0.0
        
#         results["Total Evaluated"] = total
#         results["Skipped"] = skipped
#         results["Confidence Threshold"] = confidence_threshold
#         results["Predictions Filtered"] = filtered_count
        
#         return results
    
#     def print_results(self, results: Dict):
#         """Pretty print evaluation results"""
#         print("\n" + "="*70)
#         print("EVALUATION RESULTS")
#         print("="*70)
        
#         for metric, value in results.items():
#             if isinstance(value, (int, float)):
#                 if metric in ["Total Evaluated", "Skipped", "Predictions Filtered"]:
#                     print(f"{metric:25s}: {value}")
#                 elif metric == "Confidence Threshold":
#                     print(f"{metric:25s}: {value:.2f}")
#                 else:
#                     print(f"{metric:25s}: {value:6.2f}%")
        
#         print("="*70 + "\n")
"""
Evaluation Module
Evaluates classification performance with hierarchical and ranking metrics
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import unicodedata
import re


def normalize_name(s: str) -> str:
    """Normalize commodity names for robust matching"""
    s = unicodedata.normalize('NFKD', s)
    s = re.sub(r'\W+', ' ', s)
    return s.strip().lower()


class ProductEvaluator:
    """Evaluates retrieval performance with hierarchical metrics"""

    def __init__(self, catalog_df, classifier):
        self.catalog_df = catalog_df
        self.classifier = classifier

        # Build lookup dictionaries
        self.code_to_hierarchy = {}
        self.name_to_code = {}
        self.code_to_name = {}

        for _, row in catalog_df.iterrows():
            code = row['Commodity Code']
            name = normalize_name(row['Commodity Name'])

            self.code_to_hierarchy[code] = {
                'Segment Name': row['Segment Name'],
                'Family Name': row['Family Name'],
                'Class Name': row['Class Name'],
                'Commodity Name': row['Commodity Name']
            }

            # Handle duplicate normalized names by keeping first occurrence
            if name not in self.name_to_code:
                self.name_to_code[name] = code
            
            self.code_to_name[code] = name

        print(f"Evaluator initialized with {len(self.code_to_hierarchy)} commodities")

    def _compute_dcg(self, relevances: List[int], k: int) -> float:
        """Compute Discounted Cumulative Gain at k"""
        dcg = 0.0
        for i, rel in enumerate(relevances[:k], start=1):
            dcg += rel / np.log2(i + 1)
        return dcg

    def _compute_ndcg(self, predicted_codes: List[str], true_code: str, k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain at k"""
        # Binary relevance: 1 if match, 0 otherwise
        relevances = [1 if code == true_code else 0 for code in predicted_codes]
        
        dcg = self._compute_dcg(relevances, k)
        
        # Ideal DCG: true item at position 1
        idcg = self._compute_dcg([1], k)
        
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate(
        self,
        products_df,
        top_k_list: List[int] = [1, 5, 10],
        confidence_threshold: float = 0.3
    ) -> Dict:
        """
        Evaluate classification performance

        Args:
            products_df: Test products with ground truth
            top_k_list: List of K values to evaluate
            confidence_threshold: Minimum confidence score to include prediction

        Returns:
            Dictionary of evaluation metrics
        """

        # Initialize metrics
        hits_at_k = {k: 0 for k in top_k_list}
        precision_at_k = {k: [] for k in top_k_list}
        ndcg_at_k = {k: [] for k in top_k_list}
        
        segment_hits = {k: 0 for k in top_k_list}
        family_hits = {k: 0 for k in top_k_list}
        class_hits = {k: 0 for k in top_k_list}

        mrr_scores = []
        
        total = 0
        skipped = 0
        no_predictions = 0
        filtered_all = 0
        total_predictions_filtered = 0
        insufficient_predictions = {k: 0 for k in top_k_list}

        max_k = max(top_k_list)

        for _, row in products_df.iterrows():
            description = row.get('Original Description', '')
            true_commodity_name = normalize_name(row.get('UNSPSC Commodity Name', ''))

            if not description or not true_commodity_name:
                skipped += 1
                continue

            # Map ground truth name to code
            true_commodity_code = self.name_to_code.get(true_commodity_name)
            if not true_commodity_code:
                skipped += 1
                continue

            total += 1

            # Get hierarchy for ground truth
            true_hierarchy = self.code_to_hierarchy.get(true_commodity_code, {})
            true_segment = true_hierarchy.get('Segment Name', '')
            true_family = true_hierarchy.get('Family Name', '')
            true_class = true_hierarchy.get('Class Name', '')

            # Get predictions from classifier
            predictions = self.classifier.classify_product(description, top_k=max_k)
            
            if not predictions:
                no_predictions += 1
                for k in top_k_list:
                    precision_at_k[k].append(0.0)
                    ndcg_at_k[k].append(0.0)
                mrr_scores.append(0.0)
                continue

            # Apply confidence threshold
            original_pred_count = len(predictions)
            predictions = [
                p for p in predictions
                if p.get('confidence', 1.0) >= confidence_threshold
            ]
            total_predictions_filtered += (original_pred_count - len(predictions))

            if not predictions:
                filtered_all += 1
                for k in top_k_list:
                    precision_at_k[k].append(0.0)
                    ndcg_at_k[k].append(0.0)
                mrr_scores.append(0.0)
                continue

            # Deduplicate predictions by commodity code (keep first occurrence)
            seen = set()
            unique_preds = []
            for p in predictions:
                code = p.get('commodity_code')
                if code and code not in seen:
                    seen.add(code)
                    unique_preds.append(p)
            predictions = unique_preds

            pred_codes = [p.get('commodity_code') for p in predictions]

            # Compute MRR (Mean Reciprocal Rank)
            reciprocal_rank = 0.0
            for rank, code in enumerate(pred_codes, start=1):
                if code == true_commodity_code:
                    reciprocal_rank = 1.0 / rank
                    break
            mrr_scores.append(reciprocal_rank)

            # Evaluate for each K
            for k in top_k_list:
                # Get top-k predictions
                top_k_codes = pred_codes[:k]
                
                # Track if we have fewer predictions than k
                if len(top_k_codes) < k:
                    insufficient_predictions[k] += 1

                # Check if true commodity is in top-k
                commodity_match = true_commodity_code in top_k_codes
                
                if commodity_match:
                    hits_at_k[k] += 1

                # Precision@k: Always divide by k for consistency
                # This represents: "if I take k items, what fraction are relevant?"
                precision_at_k[k].append(int(commodity_match) / k)
                
                # NDCG@k: Ranking quality metric
                ndcg_at_k[k].append(self._compute_ndcg(pred_codes, true_commodity_code, k))

                # Hierarchy matching (binary per sample)
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

            if total % 50 == 0:
                print(f"  Evaluated {total} products...")

        # Aggregate results
        results = {}
        
        # Overall statistics
        results["Total Evaluated"] = total
        results["Skipped (missing data)"] = skipped
        results["No Predictions"] = no_predictions
        results["All Predictions Filtered"] = filtered_all
        results["Total Predictions Filtered"] = total_predictions_filtered
        results["Confidence Threshold"] = confidence_threshold
        
        # Ranking metrics
        results["MRR (Mean Reciprocal Rank)"] = np.mean(mrr_scores) if mrr_scores else 0.0
        
        # Per-k metrics
        for k in sorted(top_k_list):
            # Accuracy (Hit Rate)
            results[f"Accuracy@{k}"] = (hits_at_k[k] / total * 100) if total > 0 else 0.0
            
            # Precision
            results[f"Precision@{k}"] = (np.mean(precision_at_k[k]) * 100) if precision_at_k[k] else 0.0
            
            # NDCG
            results[f"NDCG@{k}"] = (np.mean(ndcg_at_k[k]) * 100) if ndcg_at_k[k] else 0.0
            
            # Hierarchical metrics
            results[f"Segment Match@{k}"] = (segment_hits[k] / total * 100) if total > 0 else 0.0
            results[f"Family Match@{k}"] = (family_hits[k] / total * 100) if total > 0 else 0.0
            results[f"Class Match@{k}"] = (class_hits[k] / total * 100) if total > 0 else 0.0
            
            # Track insufficient predictions
            results[f"Samples with <{k} predictions"] = insufficient_predictions[k]

        return results

    def print_results(self, results: Dict):
        """Pretty print evaluation results"""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        
        # Print summary statistics first
        summary_keys = [
            "Total Evaluated", "Skipped (missing data)", "No Predictions",
            "All Predictions Filtered", "Total Predictions Filtered", "Confidence Threshold"
        ]
        
        print("\nSummary Statistics:")
        print("-" * 80)
        for key in summary_keys:
            if key in results:
                value = results[key]
                if key == "Confidence Threshold":
                    print(f"  {key:35s}: {value:.2f}")
                else:
                    print(f"  {key:35s}: {value}")
        
        # Print ranking metrics
        print("\nRanking Metrics:")
        print("-" * 80)
        if "MRR (Mean Reciprocal Rank)" in results:
            print(f"  {'MRR (Mean Reciprocal Rank)':35s}: {results['MRR (Mean Reciprocal Rank)']:.4f}")
        
        # Determine k values
        k_values = []
        for key in results:
            if key.startswith("Accuracy@"):
                k = int(key.split("@")[1])
                k_values.append(k)
        k_values = sorted(set(k_values))
        
        # Print per-k metrics grouped
        for k in k_values:
            print(f"\nMetrics @ k={k}:")
            print("-" * 80)
            
            metrics_order = [
                f"Accuracy@{k}",
                f"Precision@{k}",
                f"NDCG@{k}",
                f"Segment Match@{k}",
                f"Family Match@{k}",
                f"Class Match@{k}",
                f"Samples with <{k} predictions"
            ]
            
            for metric in metrics_order:
                if metric in results:
                    value = results[metric]
                    if isinstance(value, float) and "predictions" not in metric:
                        print(f"  {metric:35s}: {value:6.2f}%")
                    else:
                        print(f"  {metric:35s}: {value}")
        
        print("=" * 80 + "\n")

    def analyze_errors(
        self,
        products_df,
        top_k: int = 5,
        confidence_threshold: float = 0.3,
        max_errors: int = 20
    ) -> List[Dict]:
        """
        Analyze misclassified products for error analysis
        
        Args:
            products_df: Test products
            top_k: Number of predictions to analyze
            confidence_threshold: Minimum confidence
            max_errors: Maximum errors to return
            
        Returns:
            List of error cases with details
        """
        errors = []
        
        for _, row in products_df.iterrows():
            if len(errors) >= max_errors:
                break
                
            description = row.get('Original Description', '')
            true_commodity_name = normalize_name(row.get('UNSPSC Commodity Name', ''))
            
            if not description or not true_commodity_name:
                continue
            
            true_commodity_code = self.name_to_code.get(true_commodity_name)
            if not true_commodity_code:
                continue
            
            predictions = self.classifier.classify_product(description, top_k=top_k)
            if not predictions:
                continue
            
            predictions = [
                p for p in predictions
                if p.get('confidence', 1.0) >= confidence_threshold
            ]
            
            if not predictions:
                continue
            
            pred_codes = [p.get('commodity_code') for p in predictions]
            
            # Only record if it's a miss
            if true_commodity_code not in pred_codes:
                true_hierarchy = self.code_to_hierarchy[true_commodity_code]
                
                error_case = {
                    'description': description,
                    'true_code': true_commodity_code,
                    'true_name': true_hierarchy['Commodity Name'],
                    'true_class': true_hierarchy['Class Name'],
                    'true_family': true_hierarchy['Family Name'],
                    'predictions': []
                }
                
                for p in predictions[:top_k]:
                    pred_code = p['commodity_code']
                    pred_hierarchy = self.code_to_hierarchy.get(pred_code, {})
                    error_case['predictions'].append({
                        'code': pred_code,
                        'name': pred_hierarchy.get('Commodity Name', 'Unknown'),
                        'confidence': p.get('confidence', 0.0),
                        'class': pred_hierarchy.get('Class Name', 'Unknown')
                    })
                
                errors.append(error_case)
        
        return errors

    def print_error_analysis(self, errors: List[Dict]):
        """Pretty print error analysis"""
        print("\n" + "=" * 80)
        print(f"ERROR ANALYSIS (showing {len(errors)} misclassifications)")
        print("=" * 80)
        
        for i, error in enumerate(errors, 1):
            print(f"\n[Error {i}]")
            print(f"Description: {error['description'][:100]}...")
            print(f"True: {error['true_name']} ({error['true_code']})")
            print(f"      Class: {error['true_class']}, Family: {error['true_family']}")
            print(f"Predictions:")
            for j, pred in enumerate(error['predictions'], 1):
                print(f"  {j}. {pred['name']} ({pred['code']}) - conf: {pred['confidence']:.3f}")
                print(f"     Class: {pred['class']}")
            print("-" * 80)