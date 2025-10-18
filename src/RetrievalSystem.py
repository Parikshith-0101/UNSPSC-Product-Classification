
"""
Retrieval System Module
Implements hybrid retrieval combining FAISS semantic search and BM25 lexical search
"""
import os
import numpy as np
import torch
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreBuilder:
    """Builds and manages FAISS vector store"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
    def build_vectorstore(self, catalog_df, save_path: str = "faiss_index"):
        """Build FAISS index from catalog"""
        chunks = []
        for _, row in catalog_df.iterrows():
            doc = Document(
                page_content=row["corpus"],
                metadata={
                    "Segment Code": row["Segment Code"],
                    "Segment Name": row["Segment Name"],
                    "Family Code": row["Family Code"],
                    "Family Name": row["Family Name"],
                    "Class Code": row["Class Code"],
                    "Class Name": row["Class Name"],
                    "Commodity Code": row["Commodity Code"],
                    "Commodity Name": row["Commodity Name"],
                }
            )
            chunks.append(doc)
        
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )
        
        vectorstore.save_local(save_path)
        print(f" FAISS index saved to {save_path}")
        return vectorstore
   
    def load_vectorstore(self, index_path: str = r"src\faiss.index"):
        """Load existing FAISS index"""
        vectorstore = FAISS.load_local(
        index_path,                 # pass the folder, not the .faiss file
        self.embeddings,
        allow_dangerous_deserialization=True
        )
        print(f"Loaded vectorstore with {len(vectorstore.docstore._dict)} documents")
        return vectorstore



class HybridRetriever:
    """Combines semantic (FAISS) and lexical (BM25) retrieval"""
    
    def __init__(self, catalog_df, vectorstore, cleaner):
        self.catalog_df = catalog_df
        self.vectorstore = vectorstore
        self.cleaner = cleaner
        
        # Build BM25 index
        self.tokenized_docs = [
            self.cleaner.clean(doc).split() 
            for doc in catalog_df['corpus'].tolist()
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print("---- BM25 index ready--------")
    
    def retrieve(self, query: str, top_k: int = 50) -> List[Dict]:
        """Retrieve top-k candidates using hybrid search with ANN (FAISS)"""
        query_clean = self.cleaner.clean(query)
        
        # Semantic retrieval using FAISS ANN (approximate nearest neighbor)
        sem_results = self._ann_search(query_clean, k=top_k)
        sem_candidates = self._convert_results(sem_results)
        
        # Lexical retrieval (BM25)
        tokenized_query = query_clean.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(-bm25_scores)[:top_k]
        
        # Merge results
        corpus_to_candidate = {c['doc_text']: c for c in sem_candidates}
        
        for idx in top_indices:
            corpus_text = self.catalog_df.iloc[idx]['corpus']
            lex_score = float(bm25_scores[idx])
            
            if corpus_text in corpus_to_candidate:
                corpus_to_candidate[corpus_text]['lex_score'] = lex_score
            else:
                corpus_to_candidate[corpus_text] = {
                    'doc_text': corpus_text,
                    'commodity_name': self.catalog_df.iloc[idx]['Commodity Name'],
                    'commodity_code': self.catalog_df.iloc[idx]['Commodity Code'],
                    'metadata': self.catalog_df.iloc[idx].to_dict(),
                    'sem_score': 0.0,
                    'lex_score': lex_score
                }
        
        # Ensure all candidates have both scores
        candidates = list(corpus_to_candidate.values())
        for c in candidates:
            c.setdefault('sem_score', 0.0)
            c.setdefault('lex_score', 0.0)
            c.setdefault('commodity_name', c['metadata'].get('Commodity Name', ''))
            c.setdefault('commodity_code', c['metadata'].get('Commodity Code', ''))
        
        return candidates

    def _ann_search(self, query: str, k: int = 50) -> List[Tuple]:
        """Perform approximate nearest neighbor search using FAISS"""
        # Get query embedding using the same embeddings model
        query_embedding = self.vectorstore.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Access FAISS index directly for ANN search
        faiss_index = self.vectorstore.index
        
        # Search: FAISS returns distances and indices
        distances, indices = faiss_index.search(query_vector, k)
        
        # Extract documents from vectorstore's docstore
        docstore = self.vectorstore.docstore
        index_to_docid = self.vectorstore.index_to_docstore_id
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            doc_id = index_to_docid[idx]
            doc = docstore.search(doc_id)
            distance = float(distances[0][i])
            
            # Convert L2 distance to similarity score
            # Lower distance = higher similarity
            similarity_score = 1 / (1 + distance)
            
            results.append((doc, similarity_score))
        return results

    def _convert_results(self, results: List[Tuple]) -> List[Dict]:
        """Convert FAISS ANN results to dict format"""
        converted = []
        for doc, score in results:
            entry = {
                "doc_text": doc.page_content,
                "metadata": doc.metadata,
                "sem_score": float(score),
            }
            converted.append(entry)
        return converted


class CrossEncoderReranker:
    """Reranks candidates using cross-encoder model"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Cross-encoder loaded on {self.device}")
    
    def rerank(self, query: str, candidates: List[Dict], top_n: int = 20) -> List[Dict]:
        """Rerank candidates using cross-encoder"""
        if not candidates:
            return []
        
        candidate_texts = [c['doc_text'] for c in candidates]
        
        encodings = self.tokenizer(
            [query] * len(candidate_texts),
            candidate_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            scores = self.model(**encodings).logits.squeeze(-1).cpu().numpy()
        
        for i, cand in enumerate(candidates):
            cand['cross_score'] = float(scores[i])
        
        reranked = sorted(candidates, key=lambda x: x['cross_score'], reverse=True)
        return reranked[:top_n]


class ScoreMerger:
    """Combines hybrid retrieval scores with cross-encoder scores"""
    
    def __init__(self, alpha: float = 0.4, beta: float = 0.6):
        self.alpha = alpha
        self.beta = beta
    
    def merge(self, candidates: List[Dict]) -> List[Dict]:
        """Merge and normalize scores"""
        if not candidates:
            return []
        
        sem_scores = np.array([c.get('sem_score', 0) for c in candidates])
        lex_scores = np.array([c.get('lex_score', 0) for c in candidates])
        cross_scores = np.array([c.get('cross_score', 0) for c in candidates])
        
        hybrid_scores = self._normalize(sem_scores + lex_scores)
        cross_scores_norm = self._normalize(cross_scores)
        
        final_scores = self.alpha * hybrid_scores + self.beta * cross_scores_norm
        
        for i, c in enumerate(candidates):
            c['final_score'] = float(final_scores[i])
        
        return sorted(candidates, key=lambda x: x['final_score'], reverse=True)
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores"""
        scores = np.array(scores, dtype=np.float32)
        if scores.size == 0:
            return scores
        
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s == 0:
            return np.ones_like(scores)
        
        return (scores - min_s) / (max_s - min_s)