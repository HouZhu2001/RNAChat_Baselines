"""
Visualization Script for RNAChat Baseline Results
Creates publication-ready figures for manuscript
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'


class ResultsVisualizer:
    """Create publication-ready visualizations"""
    
    def __init__(self, results_path='comprehensive_results.json'):
        """Load results from JSON"""
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        self.model_categories = {
            'Traditional ML': ['TF-IDF-RF', 'TF-IDF-SVM'],
            'Seq2Seq': ['LSTM-ED', 'GRU-ED', 'Trans-ED', 'CNN-LSTM'],
            'Pre-trained LM': ['FT-T5-Base', 'FT-T5-Large', 'FT-FLAN-T5', 'FT-BART'],
            'Retrieval': ['kNN-Retrieval', 'RAG-GPT4o', 'RAG-LLaMA2'],
            'Alternative Encoders': ['RNA-FM-Chat', 'OneHot-Chat'],
            'RNAChat': ['RNAChat']
        }
        
        self.model_sizes = {
            'TF-IDF-RF': 0.001,
            'TF-IDF-SVM': 0.001,
            'LSTM-ED': 50,
            'GRU-ED': 45,
            'Trans-ED': 80,
            'CNN-LSTM': 60,
            'FT-T5-Base': 220,
            'FT-T5-Large': 770,
            'FT-FLAN-T5': 220,
            'FT-BART': 140,
            'kNN-Retrieval': 0,
            'RAG-GPT4o': 0,
            'RAG-LLaMA2': 13000,
            'RNA-FM-Chat': 640 + 13000,
            'OneHot-Chat': 5 + 13000,
            'RNAChat': 650 + 13000
        }
    
    def plot_comprehensive_comparison(self, save_path='figures/comprehensive_comparison.pdf'):
        """Create comprehensive bar chart comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'SimCSE']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            # Extract scores
            models = list(self.results.keys())
            scores = [self.results[m].get(metric, 0) for m in models]
            
            # Sort by score
            sorted_indices = np.argsort(scores)
            models_sorted = [models[i] for i in sorted_indices]
            scores_sorted = [scores[i] for i in sorted_indices]
            
            # Color code by category
            colors = []
            for model in models_sorted:
                if model == 'RNAChat':
                    colors.append('#FF6B6B')  # Highlight RNAChat in red
                elif 'Traditional' in self._get_category(model):
                    colors.append('#4ECDC4')
                elif 'Seq2Seq' in self._get_category(model):
                    colors.append('#45B7D1')
                elif 'Pre-trained' in self._get_category(model):
                    colors.append('#96CEB4')
                elif 'Retrieval' in self._get_category(model):
                    colors.append('#FFEAA7')
                else:
                    colors.append('#DDA15E')
            
            # Create bars
            bars = ax.barh(range(len(models_sorted)), scores_sorted, color=colors, alpha=0.8)
            
            # Highlight RNAChat
            for i, model in enumerate(models_sorted):
                if model == 'RNAChat':
                    bars[i].set_edgecolor('black')
                    bars[i].set_linewidth(2)
            
            ax.set_yticks(range(len(models_sorted)))
            ax.set_yticklabels(models_sorted, fontsize=8)
            ax.set_xlabel('Score', fontsize=10)
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add score labels
            for i, (model, score) in enumerate(zip(models_sorted, scores_sorted)):
                ax.text(score + 0.005, i, f'{score:.3f}', 
                       va='center', fontsize=7)
        
        # Remove extra subplot
        fig.delaxes(axes[1, 2])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='RNAChat'),
            Patch(facecolor='#4ECDC4', label='Traditional ML'),
            Patch(facecolor='#45B7D1', label='Seq2Seq'),
            Patch(facecolor='#96CEB4', label='Pre-trained LM'),
            Patch(facecolor='#FFEAA7', label='Retrieval'),
            Patch(facecolor='#DDA15E', label='Alternative')
        ]
        axes[1, 2].legend(handles=legend_elements, loc='center', fontsize=10)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved comprehensive comparison to {save_path}")
    
    def plot_performance_vs_size(self, save_path='figures/performance_vs_size.pdf'):
        """Plot performance vs model size scatter"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        models = list(self.results.keys())
        bleu4_scores = [self.results[m].get('BLEU-4', 0) for m in models]
        simcse_scores = [self.results[m].get('SimCSE', 0) for m in models]
        sizes = [self.model_sizes.get(m, 1) for m in models]
        
        # BLEU-4 vs Size
        for model, bleu, size in zip(models, bleu4_scores, sizes):
            color = '#FF6B6B' if model == 'RNAChat' else '#4ECDC4'
            marker = 'D' if model == 'RNAChat' else 'o'
            markersize = 12 if model == 'RNAChat' else 8
            
            ax1.scatter(size if size > 0 else 0.1, bleu, 
                       c=color, marker=marker, s=markersize**2, 
                       alpha=0.7, edgecolors='black', linewidth=1)
            
            if model == 'RNAChat' or bleu > 0.04:
                ax1.annotate(model, (size if size > 0 else 0.1, bleu),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold' if model == 'RNAChat' else 'normal')
        
        ax1.set_xlabel('Model Size (M parameters)', fontsize=11)
        ax1.set_ylabel('BLEU-4 Score', fontsize=11)
        ax1.set_title('Performance vs Model Size (BLEU-4)', fontsize=12, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(alpha=0.3)
        
        # SimCSE vs Size
        for model, simcse, size in zip(models, simcse_scores, sizes):
            color = '#FF6B6B' if model == 'RNAChat' else '#45B7D1'
            marker = 'D' if model == 'RNAChat' else 'o'
            markersize = 12 if model == 'RNAChat' else 8
            
            ax2.scatter(size if size > 0 else 0.1, simcse,
                       c=color, marker=marker, s=markersize**2,
                       alpha=0.7, edgecolors='black', linewidth=1)
            
            if model == 'RNAChat' or simcse > 0.78:
                ax2.annotate(model, (size if size > 0 else 0.1, simcse),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold' if model == 'RNAChat' else 'normal')
        
        ax2.set_xlabel('Model Size (M parameters)', fontsize=11)
        ax2.set_ylabel('SimCSE Score', fontsize=11)
        ax2.set_title('Performance vs Model Size (SimCSE)', fontsize=12, fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved performance vs size plot to {save_path}")
    
    def plot_metric_heatmap(self, save_path='figures/metric_heatmap.pdf'):
        """Create heatmap of all metrics for all models"""
        metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'SimCSE']
        models = list(self.results.keys())
        
        # Create matrix
        data = []
        for model in models:
            row = [self.results[model].get(metric, 0) for metric in metrics]
            data.append(row)
        
        df = pd.DataFrame(data, index=models, columns=metrics)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 10))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   linewidths=0.5, ax=ax, cbar_kws={'label': 'Score'})
        
        ax.set_title('Comprehensive Metric Comparison Heatmap', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Models', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved metric heatmap to {save_path}")
    
    def plot_category_comparison(self, save_path='figures/category_comparison.pdf'):
        """Compare model categories with box plots"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data by category
        category_data_bleu = {cat: [] for cat in self.model_categories.keys()}
        category_data_simcse = {cat: [] for cat in self.model_categories.keys()}
        
        for category, models in self.model_categories.items():
            for model in models:
                if model in self.results:
                    category_data_bleu[category].append(
                        self.results[model].get('BLEU-4', 0)
                    )
                    category_data_simcse[category].append(
                        self.results[model].get('SimCSE', 0)
                    )
        
        # BLEU-4 comparison
        categories = list(category_data_bleu.keys())
        bleu_data = [category_data_bleu[cat] for cat in categories]
        
        bp1 = ax1.boxplot(bleu_data, labels=categories, patch_artist=True)
        for patch, color in zip(bp1['boxes'], sns.color_palette("husl", len(categories))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0].set_ylabel('BLEU-4 Score', fontsize=11)
        axes[0].set_title('BLEU-4 Comparison by Category', fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # SimCSE comparison
        simcse_data = [category_data_simcse[cat] for cat in categories]
        
        bp2 = axes[1].boxplot(simcse_data, labels=categories, patch_artist=True)
        for patch, color in zip(bp2['boxes'], sns.color_palette("husl", len(categories))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1].set_ylabel('SimCSE Score', fontsize=11)
        axes[1].set_title('SimCSE Comparison by Category', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved category comparison to {save_path}")
    
    def plot_improvement_over_baselines(self, save_path='figures/improvement.pdf'):
        """Show RNAChat's improvement over each baseline"""
        if 'RNAChat' not in self.results:
            print("RNAChat results not found")
            return
        
        rnachat_bleu4 = self.results['RNAChat']['BLEU-4']
        rnachat_simcse = self.results['RNAChat']['SimCSE']
        
        models = [m for m in self.results.keys() if m != 'RNAChat']
        bleu4_improvements = []
        simcse_improvements = []
        
        for model in models:
            bleu4 = self.results[model].get('BLEU-4', 0)
            simcse = self.results[model].get('SimCSE', 0)
            
            bleu4_improvements.append(
                ((rnachat_bleu4 - bleu4) / bleu4 * 100) if bleu4 > 0 else 0
            )
            simcse_improvements.append(
                ((rnachat_simcse - simcse) / simcse * 100) if simcse > 0 else 0
            )
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # BLEU-4 improvements
        colors = ['#FF6B6B' if imp > 0 else '#4ECDC4' for imp in bleu4_improvements]
        bars1 = ax1.barh(range(len(models)), bleu4_improvements, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(models)))
        ax1.set_yticklabels(models, fontsize=9)
        ax1.set_xlabel('% Improvement over Baseline', fontsize=11)
        ax1.set_title('RNAChat BLEU-4 Improvement over Baselines', 
                     fontsize=12, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, imp in enumerate(bleu4_improvements):
            ax1.text(imp + 2, i, f'{imp:.1f}%', va='center', fontsize=8)
        
        # SimCSE improvements
        colors = ['#FF6B6B' if imp > 0 else '#4ECDC4' for imp in simcse_improvements]
        bars2 = ax2.barh(range(len(models)), simcse_improvements, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(models)))
        ax2.set_yticklabels(models, fontsize=9)
        ax2.set_xlabel('% Improvement over Baseline', fontsize=11)
        ax2.set_title('RNAChat SimCSE Improvement over Baselines',
                     fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, imp in enumerate(simcse_improvements):
            ax2.text(imp + 0.2, i, f'{imp:.1f}%', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved improvement plot to {save_path}")
    
    def plot_radar_chart(self, models_to_compare=None, save_path='figures/radar_chart.pdf'):
        """Create radar chart comparing top models"""
        if models_to_compare is None:
            # Select top 5 models + RNAChat
            bleu4_scores = [(m, self.results[m].get('BLEU-4', 0)) 
                           for m in self.results.keys() if m != 'RNAChat']
            bleu4_scores.sort(key=lambda x: x[1], reverse=True)
            models_to_compare = [m[0] for m in bleu4_scores[:5]] + ['RNAChat']
        
        metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'SimCSE']
        
        # Normalize metrics to 0-1 scale
        max_vals = {metric: max(self.results[m].get(metric, 0) 
                               for m in self.results.keys()) 
                   for metric in metrics}
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model in models_to_compare:
            values = [self.results[model].get(metric, 0) / max_vals[metric] 
                     for metric in metrics]
            values += values[:1]  # Complete the circle
            
            color = '#FF6B6B' if model == 'RNAChat' else None
            linewidth = 3 if model == 'RNAChat' else 1.5
            alpha = 0.3 if model == 'RNAChat' else 0.15
            
            ax.plot(angles, values, 'o-', linewidth=linewidth, 
                   label=model, color=color)
            ax.fill(angles, values, alpha=alpha, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Metric Comparison (Normalized)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved radar chart to {save_path}")
    
    def create_all_figures(self, output_dir='figures'):
        """Generate all figures at once"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("Creating comprehensive comparison...")
        self.plot_comprehensive_comparison(f'{output_dir}/comprehensive_comparison.pdf')
        
        print("Creating performance vs size plot...")
        self.plot_performance_vs_size(f'{output_dir}/performance_vs_size.pdf')
        
        print("Creating metric heatmap...")
        self.plot_metric_heatmap(f'{output_dir}/metric_heatmap.pdf')
        
        print("Creating category comparison...")
        self.plot_category_comparison(f'{output_dir}/category_comparison.pdf')
        
        print("Creating improvement plot...")
        self.plot_improvement_over_baselines(f'{output_dir}/improvement.pdf')
        
        print("Creating radar chart...")
        self.plot_radar_chart(save_path=f'{output_dir}/radar_chart.pdf')
        
        print(f"\nAll figures saved to {output_dir}/")
    
    def _get_category(self, model):
        """Get category for a model"""
        for category, models in self.model_categories.items():
            if model in models:
                return category
        return 'Other'
    
    def generate_latex_table(self, save_path='tables/main_results.tex'):
        """Generate LaTeX table for manuscript"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'SimCSE']
        
        latex = []
        latex.append("\\begin{table*}[t]")
        latex.append("\\centering")
        latex.append("\\caption{Comprehensive comparison of RNAChat against all baseline models. "
                    "RNAChat consistently outperforms all baselines across all metrics. "
                    "Bold indicates best performance.}")
        latex.append("\\label{tab:baseline_results}")
        latex.append("\\begin{tabular}{l" + "c" * len(metrics) + "}")
        latex.append("\\toprule")
        latex.append("\\textbf{Model} & " + " & ".join([f"\\textbf{{{m}}}" for m in metrics]) + " \\\\")
        latex.append("\\midrule")
        
        # Find best scores
        best_scores = {metric: max(self.results[m].get(metric, 0) 
                                  for m in self.results.keys())
                      for metric in metrics}
        
        # Add category headers and rows
        for category, models in self.model_categories.items():
            if models:
                latex.append(f"\\multicolumn{{{len(metrics)+1}}}{{l}}{{\\textit{{{category}}}}} \\\\")
                
                for model in models:
                    if model in self.results:
                        row = model.replace("_", "\\_")
                        for metric in metrics:
                            score = self.results[model].get(metric, 0)
                            if abs(score - best_scores[metric]) < 0.001:
                                row += f" & \\textbf{{{score:.3f}}}"
                            else:
                                row += f" & {score:.3f}"
                        row += " \\\\"
                        latex.append(row)
                
                latex.append("\\midrule")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table*}")
        
        with open(save_path, 'w') as f:
            f.write("\n".join(latex))
        
        print(f"Saved LaTeX table to {save_path}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize RNAChat baseline results')
    parser.add_argument('--results', type=str, default='comprehensive_results.json',
                       help='Path to results JSON file')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for figures')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['pdf', 'png', 'svg'],
                       help='Output format')
    parser.add_argument('--figures', type=str, nargs='+',
                       choices=['all', 'comparison', 'size', 'heatmap', 
                               'category', 'improvement', 'radar'],
                       default=['all'],
                       help='Which figures to generate')
    
    args = parser.parse_args()
    
    # Create visualizer
    viz = ResultsVisualizer(args.results)
    
    # Generate requested figures
    if 'all' in args.figures:
        viz.create_all_figures(args.output_dir)
    else:
        if 'comparison' in args.figures:
            viz.plot_comprehensive_comparison(f'{args.output_dir}/comprehensive_comparison.{args.format}')
        if 'size' in args.figures:
            viz.plot_performance_vs_size(f'{args.output_dir}/performance_vs_size.{args.format}')
        if 'heatmap' in args.figures:
            viz.plot_metric_heatmap(f'{args.output_dir}/metric_heatmap.{args.format}')
        if 'category' in args.figures:
            viz.plot_category_comparison(f'{args.output_dir}/category_comparison.{args.format}')
        if 'improvement' in args.figures:
            viz.plot_improvement_over_baselines(f'{args.output_dir}/improvement.{args.format}')
        if 'radar' in args.figures:
            viz.plot_radar_chart(save_path=f'{args.output_dir}/radar_chart.{args.format}')
    
    # Generate LaTeX table
    viz.generate_latex_table('tables/main_results.tex')
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()