"""
Evaluation Metrics for RNAChat Baselines
Implements BLEU, SimCSE, and statistical testing
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
from scipy import stats
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# BLEU METRIC IMPLEMENTATION
# ============================================================================

def compute_ngrams(sequence: List[str], n: int) -> Counter:
    """Compute n-grams from a sequence"""
    ngrams = Counter()
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i:i+n])
        ngrams[ngram] += 1
    return ngrams


def modified_precision(candidate: List[str], references: List[List[str]], n: int) -> float:
    """
    Compute modified n-gram precision
    
    Args:
        candidate: Tokenized candidate sentence
        references: List of tokenized reference sentences
        n: n-gram order
    """
    candidate_ngrams = compute_ngrams(candidate, n)
    
    if sum(candidate_ngrams.values()) == 0:
        return 0.0
    
    max_counts = Counter()
    for reference in references:
        reference_ngrams = compute_ngrams(reference, n)
        for ngram in candidate_ngrams:
            max_counts[ngram] = max(max_counts[ngram], reference_ngrams[ngram])
    
    clipped_counts = {
        ngram: min(count, max_counts[ngram])
        for ngram, count in candidate_ngrams.items()
    }
    
    return sum(clipped_counts.values()) / sum(candidate_ngrams.values())


def brevity_penalty(candidate_length: int, reference_length: int) -> float:
    """Compute BLEU brevity penalty"""
    if candidate_length > reference_length:
        return 1.0
    elif candidate_length == 0:
        return 0.0
    else:
        return np.exp(1 - reference_length / candidate_length)


def compute_bleu_single(candidate: str, reference: str, max_n: int = 4) -> List[float]:
    """
    Compute BLEU scores for a single candidate-reference pair
    
    Args:
        candidate: Predicted text
        reference: Ground truth text
        max_n: Maximum n-gram order (default: 4 for BLEU-1 to BLEU-4)
    
    Returns:
        List of BLEU scores [BLEU-1, BLEU-2, BLEU-3, BLEU-4]
    """
    # Tokenize (simple whitespace tokenization)
    candidate_tokens = candidate.lower().split()
    reference_tokens = reference.lower().split()
    
    # Compute n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        prec = modified_precision(candidate_tokens, [reference_tokens], n)
        precisions.append(prec)
    
    # Compute brevity penalty
    bp = brevity_penalty(len(candidate_tokens), len(reference_tokens))
    
    # Compute BLEU scores
    bleu_scores = []
    for n in range(1, max_n + 1):
        if precisions[n-1] == 0:
            bleu_scores.append(0.0)
        else:
            # Geometric mean of precisions up to n
            geometric_mean = np.exp(np.mean([np.log(p) if p > 0 else -np.inf 
                                             for p in precisions[:n]]))
            bleu_scores.append(bp * geometric_mean)
    
    return bleu_scores


def compute_bleu(predictions: List[str], references: List[str], 
                 max_n: int = 4) -> List[float]:
    """
    Compute average BLEU scores across all predictions
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        max_n: Maximum n-gram order
    
    Returns:
        List of average BLEU scores [BLEU-1, BLEU-2, BLEU-3, BLEU-4]
    """
    all_bleu_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = compute_bleu_single(pred, ref, max_n)
        all_bleu_scores.append(scores)
    
    # Average across all examples
    avg_bleu = np.mean(all_bleu_scores, axis=0)
    return avg_bleu.tolist()


# ============================================================================
# SimCSE METRIC IMPLEMENTATION
# ============================================================================

class SimCSEEvaluator:
    """SimCSE-based semantic similarity evaluator"""
    
    def __init__(self, model_name='princeton-nlp/sup-simcse-roberta-large', device='cuda'):
        """
        Initialize SimCSE evaluator
        
        Args:
            model_name: Pre-trained SimCSE model
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def encode(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode sentences into embeddings
        
        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for encoding
        
        Returns:
            Tensor of embeddings [num_sentences, embedding_dim]
        """
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]
            
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def compute_similarity(self, embeddings1: torch.Tensor, 
                          embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between two sets of embeddings
        
        Args:
            embeddings1: First set of embeddings [n, dim]
            embeddings2: Second set of embeddings [n, dim]
        
        Returns:
            Cosine similarities [n]
        """
        # Normalize embeddings
        embeddings1_norm = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        embeddings2_norm = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
        
        # Compute cosine similarity
        similarities = (embeddings1_norm * embeddings2_norm).sum(dim=1)
        return similarities


def compute_simcse(predictions: List[str], references: List[str], 
                   model_name='princeton-nlp/sup-simcse-roberta-large',
                   device='cuda') -> float:
    """
    Compute average SimCSE similarity between predictions and references
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        model_name: SimCSE model to use
        device: Device to run on
    
    Returns:
        Average SimCSE similarity score
    """
    evaluator = SimCSEEvaluator(model_name, device)
    
    # Encode all sentences
    pred_embeddings = evaluator.encode(predictions)
    ref_embeddings = evaluator.encode(references)
    
    # Compute similarities
    similarities = evaluator.compute_similarity(pred_embeddings, ref_embeddings)
    
    return similarities.mean().item()


# ============================================================================
# STATISTICAL SIGNIFICANCE TESTING
# ============================================================================

def paired_ttest(scores1: List[float], scores2: List[float], 
                 alternative='two-sided') -> Tuple[float, float]:
    """
    Perform paired t-test between two sets of scores
    
    Args:
        scores1: Scores from model 1
        scores2: Scores from model 2
        alternative: 'two-sided', 'greater', or 'less'
    
    Returns:
        t-statistic and p-value
    """
    t_stat, p_value = stats.ttest_rel(scores1, scores2, alternative=alternative)
    return t_stat, p_value


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply Bonferroni correction for multiple comparisons
    
    Args:
        p_values: List of p-values
    
    Returns:
        Corrected p-values
    """
    n = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    return corrected


def bootstrap_confidence_interval(scores: List[float], n_bootstrap: int = 1000,
                                  confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for scores
    
    Args:
        scores: List of scores
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95)
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    scores = np.array(scores)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    mean = np.mean(scores)
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return mean, lower, upper


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

class ComprehensiveEvaluator:
    """Comprehensive evaluator for all metrics and statistical tests"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.simcse_evaluator = SimCSEEvaluator(device=device)
    
    def evaluate_single_model(self, predictions: List[str], 
                             references: List[str],
                             compute_ci: bool = True) -> Dict:
        """
        Evaluate a single model with all metrics
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            compute_ci: Whether to compute confidence intervals
        
        Returns:
            Dictionary of metrics
        """
        results = {}
        
        # BLEU scores
        bleu_scores = compute_bleu(predictions, references)
        results['BLEU-1'] = bleu_scores[0]
        results['BLEU-2'] = bleu_scores[1]
        results['BLEU-3'] = bleu_scores[2]
        results['BLEU-4'] = bleu_scores[3]
        
        # SimCSE score
        simcse_score = compute_simcse(predictions, references, device=self.device)
        results['SimCSE'] = simcse_score
        
        # Compute confidence intervals if requested
        if compute_ci:
            # Compute individual BLEU scores for CI
            individual_bleu = []
            for pred, ref in zip(predictions, references):
                individual_bleu.append(compute_bleu_single(pred, ref))
            
            # BLEU-4 confidence interval
            bleu4_scores = [scores[3] for scores in individual_bleu]
            mean, lower, upper = bootstrap_confidence_interval(bleu4_scores)
            results['BLEU-4_CI'] = f"{mean:.4f} [{lower:.4f}, {upper:.4f}]"
            
            # SimCSE individual scores for CI
            pred_embeddings = self.simcse_evaluator.encode(predictions)
            ref_embeddings = self.simcse_evaluator.encode(references)
            simcse_scores = self.simcse_evaluator.compute_similarity(
                pred_embeddings, ref_embeddings
            ).numpy()
            
            mean, lower, upper = bootstrap_confidence_interval(simcse_scores)
            results['SimCSE_CI'] = f"{mean:.4f} [{lower:.4f}, {upper:.4f}]"
        
        return results
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict:
        """
        Compare multiple models with statistical significance testing
        
        Args:
            model_results: Dictionary of {model_name: {metric: [scores]}}
        
        Returns:
            Dictionary with comparison results and p-values
        """
        comparison = {}
        
        # Extract reference model (e.g., RNAChat)
        reference_model = 'RNAChat'
        if reference_model not in model_results:
            reference_model = list(model_results.keys())[0]
        
        ref_scores = model_results[reference_model]
        
        # Compare each model against reference
        p_values_all = []
        for model_name, scores in model_results.items():
            if model_name == reference_model:
                continue
            
            comparison[model_name] = {}
            
            # Compare BLEU-4
            if 'BLEU-4_individual' in scores and 'BLEU-4_individual' in ref_scores:
                t_stat, p_val = paired_ttest(
                    ref_scores['BLEU-4_individual'],
                    scores['BLEU-4_individual'],
                    alternative='greater'
                )
                comparison[model_name]['BLEU-4_pvalue'] = p_val
                p_values_all.append(p_val)
            
            # Compare SimCSE
            if 'SimCSE_individual' in scores and 'SimCSE_individual' in ref_scores:
                t_stat, p_val = paired_ttest(
                    ref_scores['SimCSE_individual'],
                    scores['SimCSE_individual'],
                    alternative='greater'
                )
                comparison[model_name]['SimCSE_pvalue'] = p_val
                p_values_all.append(p_val)
        
        # Apply Bonferroni correction
        if p_values_all:
            corrected_p = bonferroni_correction(p_values_all)
            comparison['bonferroni_corrected'] = corrected_p
        
        return comparison


# ============================================================================
# ADDITIONAL METRICS
# ============================================================================

def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L score (Longest Common Subsequence)
    
    Args:
        prediction: Predicted text
        reference: Reference text
    
    Returns:
        ROUGE-L F1 score
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    # Dynamic programming for LCS
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    if lcs_length == 0:
        return 0.0
    
    precision = lcs_length / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_meteor(prediction: str, reference: str) -> float:
    """
    Simplified METEOR score (without WordNet synonyms)
    
    Args:
        prediction: Predicted text
        reference: Reference text
    
    Returns:
        METEOR score
    """
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    
    # Compute matches
    matches = len(pred_tokens & ref_tokens)
    
    if matches == 0:
        return 0.0
    
    precision = matches / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = matches / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f_mean = (precision * recall) / (0.9 * precision + 0.1 * recall)
    return f_mean


def compute_bertscore(predictions: List[str], references: List[str],
                     model_name='microsoft/deberta-xlarge-mnli',
                     device='cuda') -> Tuple[float, float, float]:
    """
    Compute BERTScore
    
    Args:
        predictions: List of predictions
        references: List of references
        model_name: BERT model to use
        device: Device to run on
    
    Returns:
        (precision, recall, f1) averaged over all examples
    """
    try:
        from bert_score import score
        P, R, F1 = score(predictions, references, 
                        model_type=model_name, 
                        device=device,
                        verbose=False)
        return P.mean().item(), R.mean().item(), F1.mean().item()
    except ImportError:
        print("bert_score not installed. Install with: pip install bert-score")
        return 0.0, 0.0, 0.0


# ============================================================================
# RESULT VISUALIZATION
# ============================================================================

def create_results_table(results: Dict[str, Dict], metrics: List[str] = None) -> str:
    """
    Create a formatted table of results
    
    Args:
        results: Dictionary of {model_name: {metric: score}}
        metrics: List of metrics to include (default: all)
    
    Returns:
        Formatted table string
    """
    if metrics is None:
        # Extract all unique metrics
        metrics = set()
        for model_results in results.values():
            metrics.update(model_results.keys())
        metrics = sorted(list(metrics))
    
    # Create header
    header = "| Model | " + " | ".join(metrics) + " |"
    separator = "|" + "|".join(["-" * (len(m) + 2) for m in ["Model"] + metrics]) + "|"
    
    # Create rows
    rows = []
    for model_name, model_results in sorted(results.items()):
        row = f"| {model_name} |"
        for metric in metrics:
            score = model_results.get(metric, 0.0)
            if isinstance(score, float):
                row += f" {score:.4f} |"
            else:
                row += f" {score} |"
        rows.append(row)
    
    return "\n".join([header, separator] + rows)


def save_results_latex(results: Dict[str, Dict], filename: str):
    """
    Save results as LaTeX table
    
    Args:
        results: Dictionary of {model_name: {metric: score}}
        filename: Output filename
    """
    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'SimCSE']
    
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{l" + "c" * len(metrics) + "}")
    latex.append("\\toprule")
    latex.append("Model & " + " & ".join(metrics) + " \\\\")
    latex.append("\\midrule")
    
    for model_name, model_results in sorted(results.items()):
        row = model_name.replace("_", "\\_")
        for metric in metrics:
            score = model_results.get(metric, 0.0)
            if isinstance(score, float):
                row += f" & {score:.4f}"
            else:
                row += f" & {score}"
        row += " \\\\"
        latex.append(row)
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Comprehensive baseline comparison results.}")
    latex.append("\\label{tab:baseline_results}")
    latex.append("\\end{table}")
    
    with open(filename, 'w') as f:
        f.write("\n".join(latex))


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def run_comprehensive_evaluation(model_predictions: Dict[str, List[str]],
                                 references: List[str]) -> Dict:
    """
    Run comprehensive evaluation on all models
    
    Args:
        model_predictions: Dictionary of {model_name: [predictions]}
        references: List of reference texts
    
    Returns:
        Dictionary of all results
    """
    evaluator = ComprehensiveEvaluator()
    
    all_results = {}
    
    for model_name, predictions in model_predictions.items():
        print(f"\nEvaluating {model_name}...")
        
        # Basic metrics
        results = evaluator.evaluate_single_model(predictions, references)
        
        # Additional metrics
        rouge_scores = [compute_rouge_l(pred, ref) 
                       for pred, ref in zip(predictions, references)]
        results['ROUGE-L'] = np.mean(rouge_scores)
        
        meteor_scores = [compute_meteor(pred, ref)
                        for pred, ref in zip(predictions, references)]
        results['METEOR'] = np.mean(meteor_scores)
        
        # Store individual scores for statistical testing
        individual_bleu = [compute_bleu_single(pred, ref)[3]  # BLEU-4
                          for pred, ref in zip(predictions, references)]
        results['BLEU-4_individual'] = individual_bleu
        
        pred_embeddings = evaluator.simcse_evaluator.encode(predictions)
        ref_embeddings = evaluator.simcse_evaluator.encode(references)
        simcse_scores = evaluator.simcse_evaluator.compute_similarity(
            pred_embeddings, ref_embeddings
        ).numpy().tolist()
        results['SimCSE_individual'] = simcse_scores
        
        all_results[model_name] = results
    
    # Statistical comparisons
    print("\nPerforming statistical significance testing...")
    comparisons = evaluator.compare_models(all_results)
    
    # Print results table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(create_results_table(all_results, 
                               ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 
                                'SimCSE', 'ROUGE-L', 'METEOR']))
    
    # Save results
    import json
    with open('comprehensive_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    with open('statistical_comparisons.json', 'w') as f:
        json.dump(comparisons, f, indent=2)
    
    save_results_latex(all_results, 'baseline_results.tex')
    
    print("\nResults saved to:")
    print("  - comprehensive_results.json")
    print("  - statistical_comparisons.json")
    print("  - baseline_results.tex")
    
    return all_results, comparisons


if __name__ == '__main__':
    # Example usage
    print("Evaluation Metrics Module")
    print("="*80)
    
    # Test with sample data
    predictions = [
        "This RNA functions as a regulatory molecule in gene expression.",
        "The RNA is involved in protein synthesis and translation."
    ]
    
    references = [
        "This RNA molecule functions as a regulator of gene expression.",
        "This RNA participates in protein synthesis through translation."
    ]
    
    # Test BLEU
    print("\nTesting BLEU scores...")
    bleu_scores = compute_bleu(predictions, references)
    print(f"BLEU-1: {bleu_scores[0]:.4f}")
    print(f"BLEU-2: {bleu_scores[1]:.4f}")
    print(f"BLEU-3: {bleu_scores[2]:.4f}")
    print(f"BLEU-4: {bleu_scores[3]:.4f}")
    
    # Test SimCSE
    print("\nTesting SimCSE score...")
    simcse_score = compute_simcse(predictions, references)
    print(f"SimCSE: {simcse_score:.4f}")
    
    # Test ROUGE-L
    print("\nTesting ROUGE-L...")
    rouge_scores = [compute_rouge_l(p, r) for p, r in zip(predictions, references)]
    print(f"ROUGE-L: {np.mean(rouge_scores):.4f}")
    
    print("\nAll metrics tested successfully!")