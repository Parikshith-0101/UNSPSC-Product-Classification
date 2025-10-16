"""
Text Processing Module
Handles text cleaning, normalization, and feature extraction
"""

import re
import numpy as np

class TextCleaner:
    """Cleans and normalizes product descriptions"""
    
    def __init__(self):
        self.mappings = {
            "pvc": "polyvinyl chloride",
            "ss": "stainless steel", 
            "cs": "carbon steel",
            "vlv": "valve",
            "adpt": "adapter",
            "bush": "bushing",
            "cplg": "coupling",
            "ftg": "fitting",
            "tee": "t-junction"
        }
    
    def clean(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower().strip()
        
        # Apply mappings
        for k, v in self.mappings.items():
            text = text.replace(k, v)
        
        # Remove special characters
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class ProductFeatureExtractor:
    """Extracts structured features from product descriptions"""
    
    def __init__(self):
        self.material_patterns = {
            'stainless_steel': r'\b(ss|stainless steel|304|316)\b',
            'carbon_steel': r'\b(cs|carbon steel|mild steel)\b',
            'pvc': r'\b(pvc|polyvinyl chloride)\b',
            'copper': r'\b(copper|cu)\b',
            'aluminum': r'\b(aluminum|al)\b',
            'galvanized': r'\b(galvanized|hdg|hot dip)\b',
            'brass': r'\b(brass)\b',
        }
        
        self.volume_pattern = r'(\d+(?:\.\d+)?)\s*(?:gal|gallon|liter|l|ml|cc)'
        self.pressure_pattern = r'(\d+(?:\.\d+)?)\s*(?:psi|bar|pa|kpa)'
        self.power_pattern = r'(\d+(?:\.\d+)?)\s*(?:kw|hp|kbtu|btu|w|watt)'
        self.voltage_pattern = r'(\d+(?:\.\d+)?)\s*(?:v|volt|ac|dc)'
    
    def extract_all(self, text: str) -> np.ndarray:
        """Extract all features as a vector"""
        text_lower = text.lower()
        
        # Material features
        materials = [bool(re.search(pattern, text_lower)) 
                    for pattern in self.material_patterns.values()]
        
        # Spec features
        vol_match = re.search(self.volume_pattern, text, re.IGNORECASE)
        pres_match = re.search(self.pressure_pattern, text, re.IGNORECASE)
        pow_match = re.search(self.power_pattern, text, re.IGNORECASE)
        volt_match = re.search(self.voltage_pattern, text, re.IGNORECASE)
        
        specs = [
            bool(vol_match),
            float(vol_match.group(1)) if vol_match else 0.0,
            bool(pres_match),
            float(pres_match.group(1)) if pres_match else 0.0,
            bool(pow_match),
            float(pow_match.group(1)) if pow_match else 0.0,
            bool(volt_match),
            float(volt_match.group(1)) if volt_match else 0.0,
        ]
        
        # Keyword features
        keywords = [
            any(w in text_lower for w in ['heater', 'heating', 'furnace', 'boiler']),
            any(w in text_lower for w in ['cooler', 'cooling', 'chiller', 'fan']),
            any(w in text_lower for w in ['pipe', 'piping', 'tubing', 'coupling', 'fitting']),
            any(w in text_lower for w in ['valve', 'vlv', 'gate', 'check']),
            any(w in text_lower for w in ['electrical', 'wire', 'cable', 'panel']),
            any(w in text_lower for w in ['plumbing', 'toilet', 'sink', 'faucet']),
        ]
        
        return np.array(materials + specs + keywords, dtype=np.float32)