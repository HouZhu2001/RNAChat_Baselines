"""
Evaluation utilities for RNA inference scripts
"""
import json
import numpy as np
from typing import List, Dict, Any
from evaluation_metrics import compute_simcse, compute_bleu, compute_rouge, compute_meteor, compute_word_overlap

def get_simcse(model_path: str, func_text: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute SimCSE similarity scores for a list of text entries
    
    Args:
        model_path: Path to SimCSE model
        func_text: List of dictionaries containing 'predict_func' and 'correct_func' keys
    
    Returns:
        Dictionary containing similarity scores
    """
    predictions = [entry['predict_func'] for entry in func_text]
    references = [entry['correct_func'] for entry in func_text]
    
    try:
        simcse_score = compute_simcse(predictions, references, model_path)
    except Exception as e:
        print(f"Warning: SimCSE computation failed: {e}")
        simcse_score = 0.0
    
    return {
        'simcse_similarity': simcse_score,
        'num_samples': len(func_text)
    }

def get_simcse_llm_param(model_path: str, func_text: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics including SimCSE for LLM outputs
    
    Args:
        model_path: Path to SimCSE model
        func_text: List of dictionaries containing 'predict_func' and 'correct_func' keys
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    predictions = [entry['predict_func'] for entry in func_text]
    references = [entry['correct_func'] for entry in func_text]
    
    # Compute SimCSE similarity
    try:
        simcse_score = compute_simcse(predictions, references, model_path)
    except Exception as e:
        print(f"Warning: SimCSE computation failed: {e}")
        simcse_score = 0.0
    
    # Compute other metrics
    try:
        bleu_score = compute_bleu(predictions, references)
    except Exception as e:
        print(f"Warning: BLEU computation failed: {e}")
        bleu_score = 0.0
    
    try:
        rouge_scores = compute_rouge(predictions, references)
    except Exception as e:
        print(f"Warning: ROUGE computation failed: {e}")
        rouge_scores = {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
    
    try:
        meteor_score = compute_meteor(predictions, references)
    except Exception as e:
        print(f"Warning: METEOR computation failed: {e}")
        meteor_score = 0.0
    
    try:
        word_overlap_score = compute_word_overlap(predictions, references)
    except Exception as e:
        print(f"Warning: Word overlap computation failed: {e}")
        word_overlap_score = 0.0
    
    return {
        'simcse_similarity': simcse_score,
        'bleu_score': bleu_score,
        'rouge_scores': rouge_scores,
        'meteor_score': meteor_score,
        'word_overlap_score': word_overlap_score,
        'num_samples': len(func_text)
    }

