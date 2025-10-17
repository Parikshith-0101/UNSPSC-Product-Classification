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


======================================================================
UNSPSC PRODUCT CLASSIFICATION PIPELINE
======================================================================

Step 1: Loading data...
Loaded catalog: 71502 commodities
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
Saved triples to output\unspsc_training_triples.jsonl
--------Generated 300 training pairs------
Saved pairs to output\unspsc_training_pairs.jsonl

 Output 1 Generated:
   - Triples: output\unspsc_training_triples.jsonl
   - Pairs: output\unspsc_training_pairs.jsonl

 Step 4: Initializing classifier...
Computing commodity feature vectors...
Precomputed features for 71502 commodities

 Step 5: Classifying products...
  Processed 10 products...
  Processed 20 products...
  Processed 30 products...
  Processed 40 products...
  Processed 50 products...
  Processed 60 products...
  Processed 70 products...
  Processed 80 products...
  Processed 90 products...
  Processed 100 products...
Saved predictions to output\unspsc_candidates_dataset.csv
\Output 2 Generated:
   - Predictions: output\unspsc_candidates_dataset.csv

 Step 6: Evaluating performance...
Evaluator initialized
  Evaluated 25 products...
  Evaluated 50 products...
  Evaluated 75 products...
  Evaluated 100 products...

======================================================================
EVALUATION RESULTS
======================================================================
Top-1 Accuracy           :  21.00%
Precision@1              :  21.00%
Recall@1                 :  21.00%
Segment Acc @1           :  43.00%
Family Acc @1            :  34.00%
Class Acc @1             :  26.00%
Top-5 Accuracy           :  31.00%
Precision@5              :   6.20%
Recall@5                 :  31.00%
Segment Acc @5           :  59.00%
Family Acc @5            :  48.00%
Class Acc @5             :  39.00%
Top-10 Accuracy          :  36.00%
Precision@10             :   3.60%
Recall@10                :  36.00%
Segment Acc @10          :  64.00%
Family Acc @10           :  54.00%
Class Acc @10            :  45.00%
Total Evaluated          : 100
Skipped                  : 0
Confidence Threshold     : 0.30
Predictions Filtered     : 0
======================================================================

Sample Predictions (First 3 Products):

Product 1:
  Description: BW RG2PV75H6X 75 GAL LP GAS POWER VENTED WATER HEATER STANDARD W/SIDE CONNECTION...
  Ground Truth: Domestic water heaters
  Top 3 Predictions:
    1. Gas fueled fireplace B vent (Score: 0.5516, In Hierarchy: True)
    2. Space heaters (Score: 0.3815, In Hierarchy: True)
    3. Tungsten W (Score: 0.3202, In Hierarchy: False)

Product 2:
  Description: VERSIPRO BISC ELG CFWC WOOD SEAT SI SW-985 ORANGE CITRUS CRAZY CLEAN 19OZ SPRAYW...
  Ground Truth: Toilet seat
  Top 3 Predictions:
    1. Orange concentrate (Score: 0.8321, In Hierarchy: False)
    2. Orange juice (Score: 0.7846, In Hierarchy: False)
    3. Dried organic limequat oranges (Score: 0.7201, In Hierarchy: False)

Product 3:
  Description: 2 3000# Forged Steel Threaded Half Coupling FPT x FPT...
  Ground Truth: Forged steel pipe half coupling
  Top 3 Predictions:
    1. Forged steel pipe half coupling (Score: 1.0000, In Hierarchy: False)
    2. Coupling half (Score: 0.8685, In Hierarchy: True)
    3. Half lap coupling (Score: 0.8266, In Hierarchy: True)

======================================================================
 PIPELINE COMPLETE
======================================================================

 Output Files:
   1. output/Output1_triples.json  - Training triples
   2. output/Output1_pairs.json    - Training pairs
   3. output/Output2_predictions.json - Product classifications