"""
Hierarchical Router Module
Uses retrieved documents to constrain hierarchy search and boost scores
"""

from typing import List, Dict, Set


class DocumentInformedHierarchicalRouter:
    """Routes predictions through UNSPSC hierarchy using retrieved documents"""
    
    def __init__(self, catalog_df):
        self.catalog = catalog_df
        self.build_hierarchy()
    
    def build_hierarchy(self):
        """Build hierarchy mappings"""
        # Segment → Families
        self.segment_to_families = {}
        for _, row in self.catalog.iterrows():
            seg = row['Segment Name']
            fam = row['Family Name']
            if seg not in self.segment_to_families:
                self.segment_to_families[seg] = set()
            self.segment_to_families[seg].add(fam)
        
        # Family → Classes
        self.family_to_classes = {}
        for _, row in self.catalog.iterrows():
            fam = row['Family Name']
            cls = row['Class Name']
            if fam not in self.family_to_classes:
                self.family_to_classes[fam] = set()
            self.family_to_classes[fam].add(cls)
        
        # Class → Commodities
        self.class_to_commodities = {}
        for _, row in self.catalog.iterrows():
            cls = row['Class Name']
            comm = row['Commodity Name']
            if cls not in self.class_to_commodities:
                self.class_to_commodities[cls] = set()
            self.class_to_commodities[cls].add(comm)
        
        print(f"Hierarchy built: {len(self.segment_to_families)} segments, "
              f"{len(self.family_to_classes)} families, "
              f"{len(self.class_to_commodities)} classes")
    
    def route_from_documents(self, retrieved_candidates: List[Dict], 
                            top_k: int = 5) -> Dict:
        """
        Extract hierarchy constraints from top-k retrieved documents
        
        Args:
            retrieved_candidates: List of retrieved documents with metadata
            top_k: Number of top documents to analyze
        
        Returns:
            Dict with constrained hierarchy paths
        """
        top_docs = retrieved_candidates[:top_k]
        
        segment_counts = {}
        family_counts = {}
        class_counts = {}
        
        # Count hierarchy elements in top documents
        for doc in top_docs:
            metadata = doc.get('metadata', {})
            
            segment = metadata.get('Segment Name')
            family = metadata.get('Family Name')
            cls = metadata.get('Class Name')
            
            if segment:
                segment_counts[segment] = segment_counts.get(segment, 0) + 1
            if family:
                family_counts[family] = family_counts.get(family, 0) + 1
            if cls:
                class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Get top segments by frequency
        top_segments = sorted(segment_counts.items(), 
                            key=lambda x: x[1], reverse=True)
        
        # Get valid families constrained by top segments
        valid_families = set()
        for segment, _ in top_segments:
            valid_families.update(self.segment_to_families.get(segment, set()))
        
        top_families = [(f, family_counts.get(f, 0)) for f in valid_families]
        top_families = sorted(top_families, key=lambda x: x[1], reverse=True)
        
        # Get valid classes constrained by top families
        valid_classes = set()
        for family, _ in top_families:
            valid_classes.update(self.family_to_classes.get(family, set()))
        
        top_classes = [(c, class_counts.get(c, 0)) for c in valid_classes]
        top_classes = sorted(top_classes, key=lambda x: x[1], reverse=True)
        
        # Get valid commodities constrained by top classes
        valid_commodities = set()
        for cls, _ in top_classes:
            valid_commodities.update(self.class_to_commodities.get(cls, set()))
        
        return {
            'top_segments': dict(top_segments[:3]),
            'top_families': dict(top_families[:5]),
            'top_classes': dict(top_classes[:5]),
            'valid_commodities': valid_commodities,
            'num_constrained': len(valid_commodities)
        }
    
    def apply_hierarchy_boost(self, candidates: List[Dict], 
                             hierarchy_info: Dict,
                             boost_factor: float = 1.3) -> List[Dict]:
        """
        Boost scores for candidates within hierarchy constraints
        
        Args:
            candidates: List of candidates with scores
            hierarchy_info: Output from route_from_documents()
            boost_factor: Multiplier for candidates in hierarchy
        
        Returns:
            Candidates with hierarchy-boosted scores
        """
        valid_commodities = hierarchy_info['valid_commodities']
        
        for candidate in candidates:
            commodity_name = candidate.get('commodity_name', '')
            in_hierarchy = commodity_name in valid_commodities
            
            if in_hierarchy:
                candidate['hierarchy_boost'] = boost_factor
            else:
                candidate['hierarchy_boost'] = 1.0
            
            # Apply boost to final score
            candidate['boosted_score'] = (
                candidate.get('final_score', 0) * candidate['hierarchy_boost']
            )
            candidate['in_hierarchy'] = in_hierarchy
        
        # Re-sort by boosted score
        return sorted(candidates, key=lambda x: x['boosted_score'], reverse=True)