"""
Analyze and Compare All Baseline Results
Creates tables and figures for manuscript

Usage: python analyze_results.py --results_dir ./
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_results(results_dir):
    """Load all result JSON files"""
    results = {}
    
    result_files = list(Path(results_dir).glob('results_*.json'))
    
    for file in result_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                model_name = data['model']
                results[model_name] = data['metrics']
                print(f"Loaded: {model_name}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return results


def create_comparison_table(results):
    """Create comparison table"""
    df = pd.DataFrame(results).T
    
    # Sort by BLEU-4
    df = df.sort_values('BLEU-4', ascending=False)
    
    print("\n" + "="*100)
    print("BASELINE COMPARISON TABLE")
    print("="*100)
    print(df.to_string(float_format=lambda x: f'{x:.4f}'))
    print("="*100)
    
    # Statistics
    print(f"\nBest BLEU-4: {df['BLEU-4'].idxmax()} = {df['BLEU-4'].max():.4f}")
    print(f"Best SimCSE: {df['SimCSE'].idxmax()} = {df['SimCSE'].max():.4f}")
    print(f"Avg BLEU-4: {df['BLEU-4'].mean():.4f} ± {df['BLEU-4'].std():.4f}")
    print(f"Avg SimCSE: {df['SimCSE'].mean():.4f} ± {df['SimCSE'].std():.4f}")
    
    # Save
    df.to_csv('baseline_comparison.csv')
    print("\nSaved to: baseline_comparison.csv")
    
    return df


def create_latex_table(df, output_file='table_baselines.tex'):
    """Generate LaTeX table"""
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Comprehensive baseline comparison. Bold indicates best performance.}")
    latex.append("\\label{tab:baselines}")
    latex.append("\\begin{tabular}{lccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Model} & \\textbf{BLEU-1} & \\textbf{BLEU-2} & \\textbf{BLEU-3} & \\textbf{BLEU-4} & \\textbf{SimCSE} \\\\")
    latex.append("\\midrule")
    
    # Find best scores
    best_bleu1 = df['BLEU-1'].max()
    best_bleu2 = df['BLEU-2'].max()
    best_bleu3 = df['BLEU-3'].max()
    best_bleu4 = df['BLEU-4'].max()
    best_simcse = df['SimCSE'].max()
    
    for model, row in df.iterrows():
        model_name = model.replace('_', '\\_')
        b1 = f"\\textbf{{{row['BLEU-1']:.4f}}}" if abs(row['BLEU-1'] - best_bleu1) < 0.0001 else f"{row['BLEU-1']:.4f}"
        b2 = f"\\textbf{{{row['BLEU-2']:.4f}}}" if abs(row['BLEU-2'] - best_bleu2) < 0.0001 else f"{row['BLEU-2']:.4f}"
        b3 = f"\\textbf{{{row['BLEU-3']:.4f}}}" if abs(row['BLEU-3'] - best_bleu3) < 0.0001 else f"{row['BLEU-3']:.4f}"
        b4 = f"\\textbf{{{row['BLEU-4']:.4f}}}" if abs(row['BLEU-4'] - best_bleu4) < 0.0001 else f"{row['BLEU-4']:.4f}"
        sc = f"\\textbf{{{row['SimCSE']:.4f}}}" if abs(row['SimCSE'] - best_simcse) < 0.0001 else f"{row['SimCSE']:.4f}"
        
        latex.append(f"{model_name} & {b1} & {b2} & {b3} & {b4} & {sc} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"LaTeX table saved to: {output_file}")


def plot_comparison(df, output_dir='figures'):
    """Create comparison plots"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Bar chart comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'SimCSE']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        sorted_df = df.sort_values(metric)
        colors = ['#FF6B6B' if i == len(sorted_df)-1 else '#4ECDC4' 
                 for i in range(len(sorted_df))]
        
        ax.barh(range(len(sorted_df)), sorted_df[metric], color=colors, alpha=0.8)
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df.index, fontsize=9)
        ax.set_xlabel('Score', fontsize=10)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add values
        for i, (model, score) in enumerate(zip(sorted_df.index, sorted_df[metric])):
            ax.text(score + 0.005, i, f'{score:.3f}', va='center', fontsize=8)
    
    # Remove extra subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/baseline_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}/baseline_comparison.png', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir}/baseline_comparison.pdf")
    plt.close()
    
    # 2. Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt='.4f', cmap='YlOrRd', linewidths=0.5, ax=ax)
    ax.set_title('Baseline Performance Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/baseline_heatmap.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}/baseline_heatmap.png', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir}/baseline_heatmap.pdf")
    plt.close()
    
    # 3. Radar chart (top 5 models)
    top5 = df.nlargest(5, 'BLEU-4')
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'SimCSE']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    # Normalize to 0-1
    max_vals = df[metrics].max()
    
    for model in top5.index:
        values = (top5.loc[model, metrics] / max_vals).tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Top 5 Models - Multi-Metric Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/baseline_radar.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}/baseline_radar.png', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir}/baseline_radar.pdf")
    plt.close()


def calculate_improvements(df, reference_model='rnachat'):
    """Calculate improvement over baselines"""
    if reference_model not in df.index:
        print(f"\nWarning: {reference_model} not found. Add RNAChat results to compute improvements.")
        return
    
    ref_metrics = df.loc[reference_model]
    
    print(f"\n{'='*100}")
    print(f"RNACHAT IMPROVEMENT OVER BASELINES")
    print(f"{'='*100}")
    
    improvements = []
    for model in df.index:
        if model == reference_model:
            continue
        
        bleu4_imp = (ref_metrics['BLEU-4'] - df.loc[model, 'BLEU-4']) / df.loc[model, 'BLEU-4'] * 100
        simcse_imp = (ref_metrics['SimCSE'] - df.loc[model, 'SimCSE']) / df.loc[model, 'SimCSE'] * 100
        
        print(f"{model:20s}: BLEU-4 +{bleu4_imp:6.1f}%  |  SimCSE +{simcse_imp:6.1f}%")
        improvements.append({
            'model': model,
            'BLEU-4_improvement': bleu4_imp,
            'SimCSE_improvement': simcse_imp
        })
    
    print(f"{'='*100}")
    
    # Save improvements
    imp_df = pd.DataFrame(improvements)
    imp_df.to_csv('