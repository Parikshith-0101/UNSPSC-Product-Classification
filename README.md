# UNSPSC Vector Search Classifier 

AI-powered product categorization system that maps noisy product descriptions to UNSPSC hierarchy codes using embeddings and FAISS vector search.


# Installation & Setup:
Step 1 — Clone the repository
git clone https://github.com/<your-username>/unspsc-vectorsearch-classifier.git
cd unspsc-vectorsearch-classifier
Step 2 — Install dependencies
python -m venv venv
venv\Scripts\activate     # (Windows)
# or source venv/bin/activate (Linux/Mac)
Step 3 — Run the main pipeline
pip install -r requirements.txt
python -m src.MainPipeline


# Sample Output

When you run the pipeline, you’ll see output similar to this:

Loaded queries: 50 entries
Loaded products: 100 entries

 Data Validation:
   Queries match rate: 100.0%
   Products match rate: 100.0%

 Step 2: Initializing components...
Loaded vectorstore with 71502 documents
---- BM25 index ready--------
Cross-encoder loaded on cpu
Hierarchy built: 57 segments, 465 families, 5313 classes

 Step 3: Generating training data...
Training data generator ready
 Generated 250 training triples
Saved triples to output\Output1_triples.json
--------Generated 300 training pairs------
Saved pairs to output\Output1_pairs.json

 Output 1 Generated:
   - Triples: output\Output1_triples.json
   - Pairs: output\Output1_pairs.json

 Step 4: Initializing classifier...
Computing commodity feature vectors...
Precomputed features for 71502 commodities

 Step 5: Classifying products...
  Processed 10 products...
  ...
  Processed 100 products...

Saved predictions to output\Output2_predictions.json

 Step 6: Evaluating performance...
Evaluator initialized

======================================================================
EVALUATION RESULTS
======================================================================
Top-1 Accuracy           :  21.00%
Segment Acc @1           :  44.00%
Family Acc @1            :  35.00%
Class Acc @1             :  26.00%
Top-5 Accuracy           :  31.00%
Top-10 Accuracy          :  36.00%
======================================================================

Sample Predictions (First 3 Products)
...
======================================================================
 PIPELINE COMPLETE
======================================================================

 Output Files:
   1. output/Output1_triples.json         - Training triples
   2. output/Output1_pairs.json           - Training pairs
   3. output/Output2_predictions.json     - Final product predictions
"# UNSPSC-Product-Classifier" 
"# UNSPSC-Product-Classification" 
