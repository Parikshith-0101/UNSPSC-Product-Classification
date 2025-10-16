
"""
Training Data Generator Module
Generates training triples and pairs from search queries
"""

import json
import random
from typing import List, Dict
from pathlib import Path


class TrainingDataGenerator:
    """Generates training data for reranker/biencoder fine-tuning"""
    
    def __init__(self, catalog_df, queries_df, cleaner):
        self.catalog_df = catalog_df
        self.queries_df = queries_df
        self.cleaner = cleaner
        
        # Build commodity lookup
        self.name_to_corpus = {}
        for _, row in catalog_df.iterrows():
            name = row['Commodity Name'].strip().lower()
            self.name_to_corpus[name] = row['corpus']
        
        print(f"Training data generator ready")
    
    def generate_triples(self, num_negatives: int = 5) -> List[Dict]:
        """
        Generate training triples: (query, positive, negative)
        
        Args:
            num_negatives: Number of hard negatives per query
        
        Returns:
            List of training triples
        """
        triples = []
        
        for idx, row in self.queries_df.iterrows():
            query = row['Search Query']
            true_commodity = row['UNSPSC Commodity Name'].strip().lower()
            
            # Get positive document
            positive_corpus = self.name_to_corpus.get(true_commodity)
            if not positive_corpus:
                continue
            
            # Sample hard negatives (from same segment/family)
            true_row = self.catalog_df[
                self.catalog_df['Commodity Name'].str.lower() == true_commodity
            ]
            
            if len(true_row) == 0:
                continue
            
            true_segment = true_row.iloc[0]['Segment Name']
            true_family = true_row.iloc[0]['Family Name']
            
            # Get candidates from same family (hard negatives)
            same_family = self.catalog_df[
                (self.catalog_df['Family Name'] == true_family) &
                (self.catalog_df['Commodity Name'].str.lower() != true_commodity)
            ]
            
            # If not enough in family, sample from same segment
            if len(same_family) < num_negatives:
                same_segment = self.catalog_df[
                    (self.catalog_df['Segment Name'] == true_segment) &
                    (self.catalog_df['Commodity Name'].str.lower() != true_commodity)
                ]
                negatives = same_segment.sample(
                    min(num_negatives, len(same_segment))
                )
            else:
                negatives = same_family.sample(num_negatives)
            
            # Create triples
            for _, neg_row in negatives.iterrows():
                triple = {
                    'query': self.cleaner.clean(query),
                    'positive': positive_corpus,
                    'negative': neg_row['corpus'],
                    'positive_commodity': true_commodity,
                    'negative_commodity': neg_row['Commodity Name'].lower()
                }
                triples.append(triple)
        
        print(f" Generated {len(triples)} training triples")
        return triples
    
    def generate_pairs(self) -> List[Dict]:
        """
        Generate training pairs: (query, document, label)
        
        Returns:
            List of training pairs with labels
        """
        pairs = []
        
        for idx, row in self.queries_df.iterrows():
            query = row['Search Query']
            true_commodity = row['UNSPSC Commodity Name'].strip().lower()
            
            # Get positive document
            positive_corpus = self.name_to_corpus.get(true_commodity)
            if not positive_corpus:
                continue
            
            # Positive pair
            pairs.append({
                'query': self.cleaner.clean(query),
                'document': positive_corpus,
                'label': 1,
                'commodity': true_commodity
            })
            
            # Sample random negatives
            neg_samples = self.catalog_df[
                self.catalog_df['Commodity Name'].str.lower() != true_commodity
            ].sample(min(5, len(self.catalog_df)))
            
            for _, neg_row in neg_samples.iterrows():
                pairs.append({
                    'query': self.cleaner.clean(query),
                    'document': neg_row['corpus'],
                    'label': 0,
                    'commodity': neg_row['Commodity Name'].lower()
                })
        
        # Shuffle pairs
        #random.shuffle(pairs)  -optional if you dont want randomness
        
        print(f"--------Generated {len(pairs)} training pairs------")
        return pairs
    
    def save_training_data(self, output_dir: str = "output"):
        """Generate and save both triples and pairs"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate triples
        triples = self.generate_triples(num_negatives=5)
        triples_file = output_path / "unspsc_training_triples.jsonl"
        with open(triples_file, 'w') as f:
            json.dump(triples, f, indent=2)
        print(f"Saved triples to {triples_file}")
        
        # Generate pairs
        pairs = self.generate_pairs()
        pairs_file = output_path / "unspsc_training_pairs.jsonl"
        with open(pairs_file, 'w') as f:
            json.dump(pairs, f, indent=2)
        print(f"Saved pairs to {pairs_file}")
        
        return triples_file, pairs_file