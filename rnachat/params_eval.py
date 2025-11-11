from datetime import datetime
import torch
import json
from pathlib import Path
from collections import defaultdict

def analyze_all_checkpoints(model, checkpoint_paths, save_report=True, report_dir="/RNAChat/rnachat/param_analysis/checkpoint_analysis"):
    """
    Complete analysis of model vs multiple checkpoints with loading recommendations
    
    Args:
        model: Your PyTorch model
        checkpoint_paths: List of checkpoint file paths or dict {"name": "path"}
        save_report: Whether to save detailed reports to files
        report_dir: Directory to save reports
    
    Returns:
        dict: Complete analysis results with loading recommendations
    """
    
    # Setup
    if save_report:
        Path(report_dir).mkdir(exist_ok=True)
    
    if isinstance(checkpoint_paths, list):
        checkpoints = {f"checkpoint_{i}": path for i, path in enumerate(checkpoint_paths)}
    else:
        checkpoints = checkpoint_paths
    
    model_dict = model.state_dict()
    results = {
        'model_info': {},
        'checkpoints': {},
        'loading_recommendations': {},
        'summary': {}
    }
    
    # Analyze model
    print("="*80)
    print("COMPLETE CHECKPOINT ANALYSIS")
    print("="*80)
    
    model_lora_params = [k for k in model_dict.keys() if 'lora' in k.lower()]
    model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_total = sum(p.numel() for p in model.parameters())
    
    results['model_info'] = {
        'total_params': len(model_dict),
        'total_elements': model_total,
        'trainable_elements': model_trainable,
        'lora_params': len(model_lora_params),
        'param_patterns': get_param_patterns(model_dict),
        'lora_param_names': model_lora_params[:10]  # First 10 for reference
    }
    
    print(f"MODEL INFO:")
    print(f"  Parameters: {len(model_dict):,}")
    print(f"  Elements: {model_total:,}")
    print(f"  Trainable: {model_trainable:,} ({model_trainable/model_total*100:.1f}%)")
    print(f"  LoRA params: {len(model_lora_params)}")
    
    # Analyze each checkpoint
    for name, path in checkpoints.items():
        print(f"\n{'-'*60}")
        print(f"ANALYZING: {name} ({path})")
        print(f"{'-'*60}")
        
        try:
            # Load checkpoint
            ckpt = torch.load(path, map_location='cpu')
            ckpt_dict = ckpt['model'] if 'model' in ckpt else ckpt
            
            # Basic analysis
            ckpt_lora_params = [k for k in ckpt_dict.keys() if 'lora' in k.lower()]
            
            checkpoint_info = {
                'path': path,
                'total_params': len(ckpt_dict),
                'total_elements': sum(p.numel() for p in ckpt_dict.values()),
                'lora_params': len(ckpt_lora_params),
                'param_patterns': get_param_patterns(ckpt_dict),
                'checkpoint_keys': list(ckpt.keys()) if isinstance(ckpt, dict) else ['raw_state_dict']
            }
            
            # Key comparison
            model_keys = set(model_dict.keys())
            ckpt_keys = set(ckpt_dict.keys())
            
            common_keys = model_keys & ckpt_keys
            only_in_model = model_keys - ckpt_keys
            only_in_ckpt = ckpt_keys - model_keys
            
            # Shape compatibility analysis
            exact_matches = []
            shape_mismatches = []
            
            for key in common_keys:
                if model_dict[key].shape == ckpt_dict[key].shape:
                    exact_matches.append(key)
                else:
                    shape_mismatches.append({
                        'key': key,
                        'model_shape': tuple(model_dict[key].shape),
                        'ckpt_shape': tuple(ckpt_dict[key].shape)
                    })
            
            # LoRA analysis
            common_lora = set(model_lora_params) & set(ckpt_lora_params)
            lora_loadable = [k for k in common_lora if model_dict[k].shape == ckpt_dict[k].shape]
            
            # Key transformation analysis (for mismatched keys)
            transformation_suggestions = analyze_key_transformations(only_in_ckpt, only_in_model)
            
            analysis = {
                'checkpoint_info': checkpoint_info,
                'key_analysis': {
                    'common_keys': len(common_keys),
                    'only_in_model': len(only_in_model),
                    'only_in_checkpoint': len(only_in_ckpt),
                    'exact_matches': len(exact_matches),
                    'shape_mismatches': len(shape_mismatches)
                },
                'lora_analysis': {
                    'model_lora': len(model_lora_params),
                    'ckpt_lora': len(ckpt_lora_params),
                    'common_lora': len(common_lora),
                    'lora_loadable': len(lora_loadable)
                },
                'loadable_params': exact_matches,
                'shape_mismatches': shape_mismatches,
                'transformation_suggestions': transformation_suggestions
            }
            
            results['checkpoints'][name] = analysis
            
            # Print summary
            print(f"  Checkpoint params: {len(ckpt_dict):,}")
            print(f"  Common keys: {len(common_keys):,}")
            print(f"  Exact matches (loadable): {len(exact_matches):,}")
            print(f"  Shape mismatches: {len(shape_mismatches)}")
            print(f"  LoRA params loadable: {len(lora_loadable)}")
            
            if transformation_suggestions:
                print(f"  Potential key transformations: {len(transformation_suggestions)}")
            
        except Exception as e:
            print(f"  ERROR loading {path}: {e}")
            results['checkpoints'][name] = {'error': str(e)}
    
    # Generate loading recommendations
    print(f"\n{'='*80}")
    print("LOADING RECOMMENDATIONS")
    print(f"{'='*80}")
    
    recommendations = generate_loading_recommendations(model_dict, results['checkpoints'])
    results['loading_recommendations'] = recommendations
    
    for name, rec in recommendations.items():
        if 'error' in results['checkpoints'][name]:
            continue
            
        print(f"\n{name.upper()}:")
        print(f"  Loadability: {rec['loadability_score']:.1%}")
        print(f"  Loadable params: {rec['loadable_count']:,}")
        print(f"  Strategy: {rec['recommended_strategy']}")
        
        if rec['warnings']:
            print(f"  Warnings: {', '.join(rec['warnings'])}")
            
        if rec['code_example']:
            print(f"  Loading code:")
            for line in rec['code_example'].split('\n'):
                if line.strip():
                    print(f"    {line}")
    
    # Generate summary
    results['summary'] = generate_summary(results)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Best checkpoint for full loading: {results['summary']['best_overall']}")
    print(f"Best checkpoint for LoRA loading: {results['summary']['best_lora']}")
    print(f"Recommended loading order: {' -> '.join(results['summary']['loading_order'])}")
    
    # Save detailed reports
    if save_report:
        save_detailed_reports(results, report_dir)
        print(f"\nDetailed reports saved to: {report_dir}/")
    
    return results

def get_param_patterns(state_dict):
    """Analyze parameter name patterns"""
    patterns = defaultdict(int)
    for key in state_dict.keys():
        pattern = key.split('.')[0]
        patterns[pattern] += 1
    return dict(patterns)

def analyze_key_transformations(only_in_ckpt, only_in_model):
    """Suggest key transformations for mismatched parameter names"""
    transformations = []
    
    # Common transformation patterns
    transform_patterns = [
        ('base_model.model.', ''),
        ('base_model.', ''),
        ('.base_model.model.', '.'),
        ('.base_model.', '.'),
        ('model.model.', 'model.'),
    ]
    
    for ckpt_key in only_in_ckpt:
        for pattern_from, pattern_to in transform_patterns:
            if pattern_from in ckpt_key:
                transformed_key = ckpt_key.replace(pattern_from, pattern_to)
                if transformed_key in only_in_model:
                    transformations.append({
                        'original': ckpt_key,
                        'transformed': transformed_key,
                        'pattern': f"{pattern_from} -> {pattern_to}"
                    })
                    break
    
    return transformations

def generate_loading_recommendations(model_dict, checkpoint_analyses):
    """Generate specific loading recommendations for each checkpoint"""
    recommendations = {}
    
    for name, analysis in checkpoint_analyses.items():
        if 'error' in analysis:
            recommendations[name] = {'error': analysis['error']}
            continue
            
        loadable_count = analysis['key_analysis']['exact_matches']
        total_model_params = len(model_dict)
        lora_loadable = analysis['lora_analysis']['lora_loadable']
        
        loadability_score = loadable_count / total_model_params if total_model_params > 0 else 0
        
        # Determine strategy
        if loadability_score > 0.8:
            strategy = "direct_loading"
            code = f"""ckpt = torch.load('{analysis['checkpoint_info']['path']}', map_location='cpu')
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
msg = model.load_state_dict(state_dict, strict=False)
print(f"Loaded {{len(state_dict) - len(msg.missing_keys)}} parameters")"""
        
        elif lora_loadable > 0 and loadability_score > 0.1:
            strategy = "selective_loading"
            code = f"""ckpt = torch.load('{analysis['checkpoint_info']['path']}', map_location='cpu')
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
model_dict = model.state_dict()

# Load only compatible parameters
loadable_dict = {{k: v for k, v in state_dict.items() 
                 if k in model_dict and model_dict[k].shape == v.shape}}
msg = model.load_state_dict(loadable_dict, strict=False)
print(f"Loaded {{len(loadable_dict)}} parameters")"""
        
        elif analysis['transformation_suggestions']:
            strategy = "key_transformation"
            code = f"""ckpt = torch.load('{analysis['checkpoint_info']['path']}', map_location='cpu')
state_dict = ckpt['model'] if 'model' in ckpt else ckpt

# Apply key transformations
transformed_dict = {{}}
for key, value in state_dict.items():
    new_key = key
    # Apply your specific transformations here based on analysis
    transformed_dict[new_key] = value

msg = model.load_state_dict(transformed_dict, strict=False)"""
        
        else:
            strategy = "incompatible"
            code = "# This checkpoint is incompatible with your model structure"
        
        # Generate warnings
        warnings = []
        if analysis['key_analysis']['shape_mismatches'] > 0:
            warnings.append(f"{analysis['key_analysis']['shape_mismatches']} shape mismatches")
        if lora_loadable == 0 and analysis['lora_analysis']['ckpt_lora'] > 0:
            warnings.append("LoRA parameters not compatible")
        if loadability_score < 0.5:
            warnings.append("Low parameter compatibility")
        
        recommendations[name] = {
            'loadability_score': loadability_score,
            'loadable_count': loadable_count,
            'lora_loadable': lora_loadable,
            'recommended_strategy': strategy,
            'warnings': warnings,
            'code_example': code if strategy != "incompatible" else None
        }
    
    return recommendations

def generate_summary(results):
    """Generate overall summary and recommendations"""
    valid_checkpoints = {k: v for k, v in results['checkpoints'].items() if 'error' not in v}
    
    if not valid_checkpoints:
        return {'error': 'No valid checkpoints found'}
    
    # Find best overall checkpoint
    best_overall = max(valid_checkpoints.keys(), 
                      key=lambda k: results['loading_recommendations'][k]['loadability_score'])
    
    # Find best LoRA checkpoint
    best_lora = max(valid_checkpoints.keys(),
                   key=lambda k: results['loading_recommendations'][k]['lora_loadable'])
    
    # Suggest loading order
    loading_order = []
    if best_overall != best_lora:
        # If different, load general weights first, then LoRA
        loading_order = [best_overall, best_lora]
    else:
        loading_order = [best_overall]
    
    return {
        'best_overall': best_overall,
        'best_lora': best_lora,
        'loading_order': loading_order,
        'valid_checkpoints': len(valid_checkpoints),
        'total_checkpoints': len(results['checkpoints'])
    }

def save_detailed_reports(results, report_dir):
    """Save detailed analysis reports"""
    report_dir = Path(report_dir)
    
    # Save JSON report
    with open(report_dir / "complete_analysis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save human-readable report
    with open(report_dir / "analysis_report.txt", 'w') as f:
        f.write("COMPLETE CHECKPOINT ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Model info
        model_info = results['model_info']
        f.write(f"MODEL INFORMATION:\n")
        f.write(f"  Total parameters: {model_info['total_params']:,}\n")
        f.write(f"  Total elements: {model_info['total_elements']:,}\n")
        f.write(f"  Trainable elements: {model_info['trainable_elements']:,}\n")
        f.write(f"  LoRA parameters: {model_info['lora_params']}\n\n")
        
        # Checkpoint details
        for name, analysis in results['checkpoints'].items():
            if 'error' in analysis:
                f.write(f"{name.upper()}: ERROR - {analysis['error']}\n\n")
                continue
                
            f.write(f"{name.upper()}:\n")
            f.write(f"  Path: {analysis['checkpoint_info']['path']}\n")
            f.write(f"  Parameters: {analysis['checkpoint_info']['total_params']:,}\n")
            f.write(f"  Loadable: {analysis['key_analysis']['exact_matches']:,}\n")
            f.write(f"  Shape mismatches: {analysis['key_analysis']['shape_mismatches']}\n")
            f.write(f"  LoRA loadable: {analysis['lora_analysis']['lora_loadable']}\n")
            
            rec = results['loading_recommendations'][name]
            f.write(f"  Recommendation: {rec['recommended_strategy']}\n")
            f.write(f"  Loadability: {rec['loadability_score']:.1%}\n\n")
    
    # Save loading code
    with open(report_dir / "loading_code.py", 'w') as f:
        f.write("# Generated loading code based on analysis\n\n")
        for name, rec in results['loading_recommendations'].items():
            if rec.get('code_example'):
                f.write(f"# Loading from {name}\n")
                f.write(rec['code_example'])
                f.write("\n\n")

# Usage example:
"""
# Analyze your checkpoints
results = analyze_all_checkpoints(
    model=your_model,
    checkpoint_paths={
        'stage1': 'path/to/checkpoint1.pth',
        'peft': 'path/to/checkpoint2.pth'
    }
)

# The function will print everything and return detailed results
# Use results['loading_recommendations'] to get the exact loading code
"""




def list_detailed_param_comparison(model, checkpoint_paths, save_lists=True, output_dir="/RNAChat/rnachat/param_analysis/detailed_param_lists"):
    """
    List exactly which parameters are loadable, missing, and extra for each checkpoint
    
    Args:
        model: Your PyTorch model
        checkpoint_paths: Dict {"name": "path"} or list of paths
        save_lists: Whether to save lists to separate files
        output_dir: Directory to save the detailed lists
    
    Returns:
        dict: Detailed parameter lists for each checkpoint
    """
    
    if save_lists:
        Path(output_dir).mkdir(exist_ok=True)
    
    if isinstance(checkpoint_paths, list):
        checkpoints = {f"checkpoint_{i}": path for i, path in enumerate(checkpoint_paths)}
    else:
        checkpoints = checkpoint_paths
    
    model_dict = model.state_dict()
    model_keys = set(model_dict.keys())
    
    results = {}
    
    print("="*100)
    print("DETAILED PARAMETER COMPARISON")
    print("="*100)
    
    print(f"\nMODEL HAS {len(model_keys)} PARAMETERS:")
    print("-" * 50)
    
    # Group model parameters by component for better understanding
    model_components = {}
    for key in sorted(model_keys):
        component = key.split('.')[0]
        if component not in model_components:
            model_components[component] = []
        model_components[component].append(key)
    
    for component, params in model_components.items():
        print(f"  {component}: {len(params)} parameters")
        if len(params) <= 5:  # Show all if few parameters
            for param in params:
                print(f"    - {param}")
        else:  # Show first 3 and last 2
            for param in params[:3]:
                print(f"    - {param}")
            print(f"    - ... ({len(params)-5} more)")
            for param in params[-2:]:
                print(f"    - {param}")
        print()
    
    # Analyze each checkpoint
    for name, path in checkpoints.items():
        print(f"\n{'='*100}")
        print(f"ANALYZING {name.upper()}: {path}")
        print(f"{'='*100}")
        
        try:
            # Load checkpoint
            ckpt = torch.load(path, map_location='cpu')
            ckpt_dict = ckpt['model'] if 'model' in ckpt else ckpt
            ckpt_keys = set(ckpt_dict.keys())
            
            print(f"\nCHECKPOINT HAS {len(ckpt_keys)} PARAMETERS")
            
            # Calculate intersections
            loadable_params = []  # Common keys with same shape
            shape_mismatch_params = []  # Common keys with different shapes
            missing_in_checkpoint = model_keys - ckpt_keys  # In model but not in checkpoint
            extra_in_checkpoint = ckpt_keys - model_keys    # In checkpoint but not in model
            
            # Check shape compatibility for common keys
            common_keys = model_keys & ckpt_keys
            for key in common_keys:
                if model_dict[key].shape == ckpt_dict[key].shape:
                    loadable_params.append(key)
                else:
                    shape_mismatch_params.append({
                        'key': key,
                        'model_shape': tuple(model_dict[key].shape),
                        'ckpt_shape': tuple(ckpt_dict[key].shape)
                    })
            
            # Separate LoRA parameters for detailed analysis
            loadable_lora = [k for k in loadable_params if 'lora' in k.lower()]
            loadable_non_lora = [k for k in loadable_params if 'lora' not in k.lower()]
            
            missing_lora = [k for k in missing_in_checkpoint if 'lora' in k.lower()]
            missing_non_lora = [k for k in missing_in_checkpoint if 'lora' not in k.lower()]
            
            extra_lora = [k for k in extra_in_checkpoint if 'lora' in k.lower()]
            extra_non_lora = [k for k in extra_in_checkpoint if 'lora' not in k.lower()]
            
            # Store results
            checkpoint_result = {
                'checkpoint_path': path,
                'total_checkpoint_params': len(ckpt_keys),
                'loadable_params': sorted(loadable_params),
                'loadable_lora': sorted(loadable_lora),
                'loadable_non_lora': sorted(loadable_non_lora),
                'shape_mismatches': shape_mismatch_params,
                'missing_in_checkpoint': sorted(missing_in_checkpoint),
                'missing_lora': sorted(missing_lora),
                'missing_non_lora': sorted(missing_non_lora),
                'extra_in_checkpoint': sorted(extra_in_checkpoint),
                'extra_lora': sorted(extra_lora),
                'extra_non_lora': sorted(extra_non_lora)
            }
            
            results[name] = checkpoint_result
            
            # Print summary
            print(f"\n📊 SUMMARY for {name}:")
            print(f"  ✅ Loadable parameters: {len(loadable_params)} ({len(loadable_params)/len(model_keys)*100:.1f}% of model)")
            print(f"     - LoRA parameters: {len(loadable_lora)}")
            print(f"     - Other parameters: {len(loadable_non_lora)}")
            print(f"  ❌ Missing in checkpoint: {len(missing_in_checkpoint)} ({len(missing_in_checkpoint)/len(model_keys)*100:.1f}% of model)")
            print(f"     - LoRA parameters: {len(missing_lora)}")
            print(f"     - Other parameters: {len(missing_non_lora)}")
            print(f"  ➕ Extra in checkpoint: {len(extra_in_checkpoint)}")
            print(f"     - LoRA parameters: {len(extra_lora)}")
            print(f"     - Other parameters: {len(extra_non_lora)}")
            print(f"  ⚠️  Shape mismatches: {len(shape_mismatch_params)}")
            
            # Print detailed lists (truncated for console)
            def print_param_list(title, param_list, max_show=10):
                if not param_list:
                    print(f"\n{title}: None")
                    return
                    
                print(f"\n{title}: {len(param_list)} parameters")
                print("-" * 50)
                
                if len(param_list) <= max_show:
                    for param in param_list:
                        if isinstance(param, dict):  # Shape mismatch case
                            print(f"  - {param['key']}: model{param['model_shape']} vs ckpt{param['ckpt_shape']}")
                        else:
                            print(f"  - {param}")
                else:
                    for param in param_list[:max_show-2]:
                        if isinstance(param, dict):
                            print(f"  - {param['key']}: model{param['model_shape']} vs ckpt{param['ckpt_shape']}")
                        else:
                            print(f"  - {param}")
                    print(f"  - ... ({len(param_list)-(max_show-2)} more)")
                    for param in param_list[-(2):]:
                        if isinstance(param, dict):
                            print(f"  - {param['key']}: model{param['model_shape']} vs ckpt{param['ckpt_shape']}")
                        else:
                            print(f"  - {param}")
            
            # Print truncated lists for console
            print_param_list("🟢 LOADABLE PARAMETERS", loadable_params, 15)
            print_param_list("🔴 MISSING IN CHECKPOINT", list(missing_in_checkpoint), 15)
            print_param_list("🔵 EXTRA IN CHECKPOINT", list(extra_in_checkpoint), 15)
            print_param_list("🟡 SHAPE MISMATCHES", shape_mismatch_params, 10)
            
            # Save detailed lists to files
            if save_lists:
                save_checkpoint_lists(name, checkpoint_result, output_dir)
                
        except Exception as e:
            print(f"❌ ERROR loading {path}: {e}")
            results[name] = {'error': str(e)}
    
    # Print final comparison summary
    print(f"\n{'='*100}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*100}")
    
    for name, result in results.items():
        if 'error' in result:
            print(f"{name}: ERROR - {result['error']}")
            continue
            
        print(f"\n{name.upper()}:")
        print(f"  Loadable: {len(result['loadable_params']):4d} / {len(model_keys)} ({len(result['loadable_params'])/len(model_keys)*100:5.1f}%)")
        print(f"  Missing:  {len(result['missing_in_checkpoint']):4d} / {len(model_keys)} ({len(result['missing_in_checkpoint'])/len(model_keys)*100:5.1f}%)")
        print(f"  Extra:    {len(result['extra_in_checkpoint']):4d}")
        
        if result['loadable_lora']:
            print(f"  LoRA loadable: {len(result['loadable_lora'])} parameters")
    
    # Generate loading recommendations
    print(f"\n{'='*100}")
    print("LOADING RECOMMENDATIONS")
    print(f"{'='*100}")
    
    best_overall = max(results.keys(), key=lambda k: len(results[k].get('loadable_params', [])) if 'error' not in results[k] else 0)
    best_lora = max(results.keys(), key=lambda k: len(results[k].get('loadable_lora', [])) if 'error' not in results[k] else 0)
    
    print(f"🏆 Best overall coverage: {best_overall} ({len(results[best_overall]['loadable_params'])} params)")
    print(f"🧬 Best LoRA coverage: {best_lora} ({len(results[best_lora]['loadable_lora'])} LoRA params)")
    
    # Generate loading code
    for name, result in results.items():
        if 'error' in result or not result['loadable_params']:
            continue
            
        print(f"\n💻 Loading code for {name}:")
        print("```python")
        print(f"# Load {name} checkpoint")
        print(f"ckpt = torch.load('{result['checkpoint_path']}', map_location='cpu')")
        print("state_dict = ckpt['model'] if 'model' in ckpt else ckpt")
        print("model_dict = model.state_dict()")
        print()
        print("# Filter loadable parameters")
        print("loadable_dict = {}")
        print("for key, value in state_dict.items():")
        print("    if key in model_dict and model_dict[key].shape == value.shape:")
        print("        loadable_dict[key] = value")
        print()
        print("# Load parameters")
        print("msg = model.load_state_dict(loadable_dict, strict=False)")
        print(f"print(f'Loaded {{len(loadable_dict)}} parameters from {name}')")
        print(f"# Expected to load: {len(result['loadable_params'])} parameters")
        print("```")
    
    if save_lists:
        print(f"\n📁 Detailed parameter lists saved to: {output_dir}/")
    
    return results

def save_checkpoint_lists(checkpoint_name, result, output_dir):
    """Save detailed parameter lists to separate files"""
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save loadable parameters
    with open(checkpoint_dir / "loadable_params.txt", 'w') as f:
        f.write(f"LOADABLE PARAMETERS ({len(result['loadable_params'])} total)\n")
        f.write("="*60 + "\n\n")
        f.write("LoRA Parameters:\n")
        f.write("-"*30 + "\n")
        for param in result['loadable_lora']:
            f.write(f"{param}\n")
        f.write(f"\nOther Parameters:\n")
        f.write("-"*30 + "\n")
        for param in result['loadable_non_lora']:
            f.write(f"{param}\n")
    
    # Save missing parameters
    with open(checkpoint_dir / "missing_params.txt", 'w') as f:
        f.write(f"MISSING IN CHECKPOINT ({len(result['missing_in_checkpoint'])} total)\n")
        f.write("="*60 + "\n\n")
        f.write("Missing LoRA Parameters:\n")
        f.write("-"*30 + "\n")
        for param in result['missing_lora']:
            f.write(f"{param}\n")
        f.write(f"\nMissing Other Parameters:\n")
        f.write("-"*30 + "\n")
        for param in result['missing_non_lora']:
            f.write(f"{param}\n")
    
    # Save extra parameters
    with open(checkpoint_dir / "extra_params.txt", 'w') as f:
        f.write(f"EXTRA IN CHECKPOINT ({len(result['extra_in_checkpoint'])} total)\n")
        f.write("="*60 + "\n\n")
        f.write("Extra LoRA Parameters:\n")
        f.write("-"*30 + "\n")
        for param in result['extra_lora']:
            f.write(f"{param}\n")
        f.write(f"\nExtra Other Parameters:\n")
        f.write("-"*30 + "\n")
        for param in result['extra_non_lora']:
            f.write(f"{param}\n")
    
    # Save shape mismatches
    if result['shape_mismatches']:
        with open(checkpoint_dir / "shape_mismatches.txt", 'w') as f:
            f.write(f"SHAPE MISMATCHES ({len(result['shape_mismatches'])} total)\n")
            f.write("="*60 + "\n\n")
            for mismatch in result['shape_mismatches']:
                f.write(f"{mismatch['key']}:\n")
                f.write(f"  Model shape: {mismatch['model_shape']}\n")
                f.write(f"  Checkpoint shape: {mismatch['ckpt_shape']}\n\n")
    
    # Save summary
    with open(checkpoint_dir / "summary.txt", 'w') as f:
        f.write(f"SUMMARY FOR {checkpoint_name.upper()}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Checkpoint path: {result['checkpoint_path']}\n")
        f.write(f"Total checkpoint parameters: {result['total_checkpoint_params']}\n\n")
        f.write(f"Loadable parameters: {len(result['loadable_params'])}\n")
        f.write(f"  - LoRA: {len(result['loadable_lora'])}\n")
        f.write(f"  - Other: {len(result['loadable_non_lora'])}\n\n")
        f.write(f"Missing parameters: {len(result['missing_in_checkpoint'])}\n")
        f.write(f"  - LoRA: {len(result['missing_lora'])}\n")
        f.write(f"  - Other: {len(result['missing_non_lora'])}\n\n")
        f.write(f"Extra parameters: {len(result['extra_in_checkpoint'])}\n")
        f.write(f"  - LoRA: {len(result['extra_lora'])}\n")
        f.write(f"  - Other: {len(result['extra_non_lora'])}\n\n")
        f.write(f"Shape mismatches: {len(result['shape_mismatches'])}\n")

# Quick function for your specific case
def analyze_stage1_and_peft_params(model):
    """Quick analysis for your specific STAGE1 and PEFT checkpoints"""
    return list_detailed_param_comparison(
        model=model,
        checkpoint_paths={
            'STAGE1': 'rnachat/checkpoints/checkpoint_stage1.pth',
            'PEFT': 'rnachat/checkpoints/checkpoint_stage2.pth'
        },
        save_lists=True,
        output_dir="./detailed_param_analysis"
    )

# Usage example:
"""
# Get detailed parameter comparison
results = list_detailed_param_comparison(
    model=your_model,
    checkpoint_paths={
        'stage1': 'rnachat/checkpoints/checkpoint_stage1.pth',
        'peft': 'rnachat/checkpoints/checkpoint_stage2.pth'
    }
)

# This will print everything and save detailed files:
# ./detailed_param_lists/stage1/loadable_params.txt
# ./detailed_param_lists/stage1/missing_params.txt  
# ./detailed_param_lists/stage1/extra_params.txt
# ./detailed_param_lists/peft/loadable_params.txt
# ./detailed_param_lists/peft/missing_params.txt
# ./detailed_param_lists/peft/extra_params.txt
"""


def count_changed_params(model, original_state_dict):
    changed, unchanged = 0, 0
    for name, param in model.state_dict().items():
        if name in original_state_dict:
            # 比较内容是否相同
            if not torch.equal(param.cpu(), original_state_dict[name].cpu()):
                changed += param.numel()
            else:
                unchanged += param.numel()
        else:
            # 新增的权重（LoRA 之类）
            changed += param.numel()
    return changed, unchanged



def count_changed_layers(model, original_state_dict):
    layer_changed = defaultdict(bool)  # 记录每一层是否有改动

    for name, param in model.state_dict().items():
        layer_name = ".".join(name.split(".")[:5])  # 根据你的层命名结构调整切片深度
        if name in original_state_dict:
            if not torch.equal(param.cpu(), original_state_dict[name].cpu()):
                layer_changed[layer_name] = True
        else:
            # 新增的权重（比如 LoRA）直接算改动
            layer_changed[layer_name] = True

    changed_layers = sum(1 for changed in layer_changed.values() if changed)
    total_layers = len(layer_changed)
    unchanged_layers = total_layers - changed_layers

    return changed_layers, unchanged_layers, total_layers




def save_comprehensive_loading_report(stage1_msg, stage2_msg, model, 
                                    stage1_path, stage2_path, 
                                    output_file="/RNAChat/rnachat/param_analysis/comprehensive_load_report.json"):
    """
    Create a comprehensive loading report for both checkpoints
    """

    # Get model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = [name for name, _ in model.named_parameters() if 'lora' in name.lower()]
    model_keys = list(model.state_dict().keys())
    
    comprehensive_report = {
        "loading_timestamp": datetime.now().isoformat(),
        "stage1_checkpoint": {
            "path": str(stage1_path),
            "missing_keys": list(stage1_msg.missing_keys) if stage1_msg.missing_keys else [],
            "unexpected_keys": list(stage1_msg.unexpected_keys) if stage1_msg.unexpected_keys else [],
            "checkpoint_keys_count": 480,
            "missing_count": len(stage1_msg.missing_keys) if stage1_msg.missing_keys else 0,
            "unexpected_count": len(stage1_msg.unexpected_keys) if stage1_msg.unexpected_keys else 0,
            "load_successful": len(stage1_msg.missing_keys) == 0 and len(stage1_msg.unexpected_keys) == 0
        },
        "stage2_checkpoint": {
            "path": str(stage2_path),
            "missing_keys": list(stage2_msg.missing_keys) if stage2_msg.missing_keys else [],
            "unexpected_keys": list(stage2_msg.unexpected_keys) if stage2_msg.unexpected_keys else [],
            "checkpoint_keys_count": 200,
            "missing_count": len(stage2_msg.missing_keys) if stage2_msg.missing_keys else 0,
            "unexpected_count": len(stage2_msg.unexpected_keys) if stage2_msg.unexpected_keys else 0,
            "load_successful": len(stage2_msg.missing_keys) == 0 and len(stage2_msg.unexpected_keys) == 0
        },
        "model_statistics": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": round(trainable_params / total_params * 100, 2),
            "lora_parameters_count": len(lora_params),
            "lora_parameters_names": lora_params[:10],  # First 10 for brevity
            "total_model_keys": len(model_keys),
            "model_keys": model_keys
        },
        "overall_status": {
            "both_stages_successful": (
                len(stage1_msg.missing_keys) == 0 and len(stage1_msg.unexpected_keys) == 0 and
                len(stage2_msg.missing_keys) == 0 and len(stage2_msg.unexpected_keys) == 0
            ),
            "total_missing_keys": (
                len(stage1_msg.missing_keys) + len(stage2_msg.missing_keys) 
                if stage1_msg.missing_keys and stage2_msg.missing_keys else 0
            ),
            "total_unexpected_keys": (
                len(stage1_msg.unexpected_keys) + len(stage2_msg.unexpected_keys)
                if stage1_msg.unexpected_keys and stage2_msg.unexpected_keys else 0
            )
        }
    }
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(comprehensive_report, f, indent=2)
    
    print(f"Comprehensive load report saved to: {output_file}")
    return comprehensive_report

