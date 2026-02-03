import json
import argparse
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from sklearn.metrics import accuracy_score

def calculate_metrics(file_path):
    print(f"Loading results from: {file_path}")
    
    # 1. Load the data
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    # Handle different json formats (if it's a list or a dict with 'samples')
    if isinstance(data, dict) and 'samples' in data:
        records = data['samples']
    elif isinstance(data, list):
        records = data
    else:
        print("Error: Unknown JSON format.")
        return

    # 2. Extract References (Truth) and Hypotheses (Predictions)
    references = []
    hypotheses = []
    
    # For Exact Match Accuracy
    clean_refs = []
    clean_preds = []

    print(f"Evaluating {len(records)} samples...")

    for item in records:
        # Get raw text
        ref_text = item.get('answer', '').strip()
        pred_text = item.get('prediction', '').strip() or item.get('generated_report', '').strip()

        # simple tokenizer (whitespace) for BLEU
        references.append([ref_text.split()])
        hypotheses.append(pred_text.split())
        
        # Clean text for strict accuracy (lowercase, remove 'Report:' prefix)
        r_clean = ref_text.lower().replace('report:', '').strip()
        p_clean = pred_text.lower().replace('report:', '').strip()
        
        clean_refs.append(r_clean)
        clean_preds.append(p_clean)

    # 3. Calculate Scores
    
    # BLEU Scores (1-gram to 4-gram)
    # Weights: (1,0,0,0) means only single words. (0.25, 0.25, 0.25, 0.25) includes phrases.
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    # ROUGE Scores (Measures overlap, good for medical reports)
    rouge = Rouge()
    # Rouge might fail if empty strings, so we filter
    try:
        rouge_scores = rouge.get_scores(clean_preds, clean_refs, avg=True)
        rouge_l = rouge_scores['rouge-l']['f'] # F1-score of Longest Common Subsequence
    except:
        rouge_l = 0.0

    # Exact Accuracy (How often was it PERFECT?)
    accuracy = accuracy_score(clean_refs, clean_preds)
    
    # Clinical Accuracy Approximation 
    # (Checks if key terms like 'sinus' appear in both)
    correct_content = 0
    for r, p in zip(clean_refs, clean_preds):
        # Very basic check: if the main diagnosis word matches
        if r in p or p in r: 
            correct_content += 1
    content_accuracy = correct_content / len(clean_refs)

    # 4. Print Report
    print("-" * 30)
    print("FINAL VALIDATION METRICS")
    print("-" * 30)
    print(f"Exact Match Accuracy: {accuracy:.4f}  (Matches perfectly)")
    print(f"Content Overlap Acc:  {content_accuracy:.4f}  (Partial match)")
    print(f"BLEU-1 Score:         {bleu_1:.4f}    (Word choice)")
    print(f"BLEU-4 Score:         {bleu_4:.4f}    (Phrasing)")
    print(f"ROUGE-L Score:        {rouge_l:.4f}   (Summary quality)")
    print("-" * 30)

if __name__ == "__main__":
    # Point this to your actual result file
    # Based on your previous command, it is inside ./output/inference_results_full/
    # You might need to check the filename, often it is 'predictions.json' or 'results.json'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Path to results JSON')
    args = parser.parse_args()
    
    calculate_metrics(args.file)