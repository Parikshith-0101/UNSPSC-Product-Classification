"""
Data Loading Module
Handles loading and preprocessing of UNSPSC catalog, queries, and products
"""

import pandas as pd
from pathlib import Path

class DataLoader:
    """Loads and validates all required datasets"""
    
    def __init__(self, data_dir: str = "data"):
        # Automatically resolve path relative to this file’s directory
        self.data_dir = (Path(__file__).resolve().parent.parent / data_dir).resolve()
        
    def load_catalog(self) -> pd.DataFrame:
        """Load UNSPSC catalog with hierarchy"""
        file_path = self.data_dir / "dylog_unspsc_data.csv"
        df = pd.read_csv(file_path)
        
        # Create corpus column for retrieval
        df['corpus'] = (
            df["Segment Name"] + " " +
            df["Family Name"] + " " +
            df["Class Name"] + " " +
            df["Commodity Name"]
        ).str.lower()
        
        print(f"Loaded catalog: {len(df)} commodities")
        return df
    
    def load_queries(self) -> pd.DataFrame:
        """Load training queries"""
        file_path = self.data_dir / "dylog_search_queries_unspsc.csv"
        df = pd.read_csv(file_path)
        print(f"Loaded queries: {len(df)} entries")
        return df
    
    def load_products(self) -> pd.DataFrame:
        """Load test products"""
        file_path = self.data_dir / "dylog_sample_product_unspsc.csv"
        df = pd.read_csv(file_path)
        print(f"Loaded products: {len(df)} entries")
        return df
    
    def validate_data(self, catalog_df: pd.DataFrame, 
                     queries_df: pd.DataFrame, 
                     products_df: pd.DataFrame) -> dict:
        """Validate data consistency"""
        catalog_names = set(catalog_df['Commodity Name'].str.strip().str.lower())
        
        query_names = set(queries_df['UNSPSC Commodity Name'].str.strip().str.lower())
        query_match = len(query_names.intersection(catalog_names))
        
        product_names = set(products_df['UNSPSC Commodity Name'].str.strip().str.lower())
        product_match = len(product_names.intersection(catalog_names))
        
        stats = {
            'catalog_commodities': len(catalog_names),
            'query_commodities': len(query_names),
            'query_match_rate': query_match / len(query_names) * 100,
            'product_commodities': len(product_names),
            'product_match_rate': product_match / len(product_names) * 100
        }
        
        print("\n Data Validation:")
        print(f"   Queries match rate: {stats['query_match_rate']:.1f}%")
        print(f"   Products match rate: {stats['product_match_rate']:.1f}%")
        
        return stats
