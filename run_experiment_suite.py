"""
Experiment Suite Runner
Esegue set di esperimenti configurati e genera report comparativi
"""

import argparse
import yaml
import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import pandas as pd
from tabulate import tabulate

CONFIG_FILE = Path("src/configs/config.yaml")
RESULTS_DIR = Path("results/experiment_suites")

def load_config():
    """Carica configurazione"""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    with open(CONFIG_FILE, 'r') as f:
        return yaml.safe_load(f)

def run_single_experiment(exp_name, config):
    """Esegue un singolo esperimento"""
    print(f"▶️  Running {exp_name}...", end=" ", flush=True)
    
    cmd = [
        sys.executable,
        "main.py",
        exp_name
    ]
    
    try:
        # Usa Popen per streaming real-time dell'output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Cattura output mentre lo mostra
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # Mostra in real-time
            output_lines.append(line)
        
        process.wait(timeout=7200)
        full_output = ''.join(output_lines)
        
        if process.returncode == 0:
            print("✅ DONE")
            return parse_experiment_output(full_output, exp_name, config)
        else:
            print(f"❌ FAILED (exit code: {process.returncode})")
            return None
    except subprocess.TimeoutExpired:
        print("⏱️ TIMEOUT")
        process.kill()
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

def parse_experiment_output(output, exp_name, config):
    """Estrae metriche dall'output dell'esperimento"""
    lines = output.split('\n')
    
    metrics = {
        'experiment': exp_name,
        'config': config[exp_name]['params'],
        'train_accuracy': None,
        'train_precision': None,
        'train_recall': None,
        'train_f1': None,
        'test_accuracy': None,
        'test_precision': None,
        'test_recall': None,
        'test_f1': None,
        'mean_confidence': None,
        'mean_epistemic_uncertainty': None,
        'timestamp': datetime.now().isoformat()
    }
    
    # Parse average metrics
    for line in lines:
        if 'Average metrics across folds:' in line:
            idx = lines.index(line)
            for i in range(idx+1, min(idx+15, len(lines))):
                metric_line = lines[i]
                for key in ['train_accuracy', 'train_precision', 'train_recall', 'train_f1',
                           'test_accuracy', 'test_precision', 'test_recall', 'test_f1']:
                    if key in metric_line:
                        try:
                            value = float(metric_line.split(':')[1].strip())
                            metrics[key] = value
                        except:
                            pass
                # Parse MC Dropout metrics
                if 'mean_confidence:' in metric_line:
                    try:
                        value = float(metric_line.split(':')[1].strip())
                        metrics['mean_confidence'] = value
                    except:
                        pass
                if 'mean_epistemic_uncertainty:' in metric_line:
                    try:
                        value = float(metric_line.split(':')[1].strip())
                        metrics['mean_epistemic_uncertainty'] = value
                    except:
                        pass
            break
    
    return metrics

def save_results(suite_name, results):
    """Salva risultati in JSON"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"{suite_name}_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {result_file}")
    return result_file

def generate_comparison_table(results, suite_info):
    """Genera tabella comparativa"""
    if not results or all(r is None for r in results):
        print("\n❌ No valid results to compare")
        return
    
    print("\n" + "="*80)
    print(f"📊 EXPERIMENT SUITE: {suite_info['description']}")
    print("="*80)
    
    # Tabella configurazioni
    print("\n🔧 CONFIGURATIONS:")
    config_data = []
    for r in results:
        if r:
            config_data.append({
                'Experiment': r['experiment'],
                'Epochs': r['config'].get('epochs', 'N/A'),
                'Iterations': r['config'].get('n_iteration', 'N/A'),
                'Freeze': r['config'].get('frezze_layer', 'N/A'),
                'Modality': r['config'].get('mod', 'both'),
                'MC Dropout': '✅' if r['config'].get('use_mcdo', False) else '❌'
            })
    if config_data:
        print(tabulate(config_data, headers='keys', tablefmt='grid'))
    
    # Tabella metriche
    print("\n📈 RESULTS:")
    metrics_data = []
    for r in results:
        if r and r['test_accuracy'] is not None:
            row = {
                'Experiment': r['experiment'],
                'Train Acc': f"{r['train_accuracy']:.4f}" if r['train_accuracy'] else 'N/A',
                'Test Acc': f"{r['test_accuracy']:.4f}" if r['test_accuracy'] else 'N/A',
                'Test Prec': f"{r['test_precision']:.4f}" if r['test_precision'] else 'N/A',
                'Test Rec': f"{r['test_recall']:.4f}" if r['test_recall'] else 'N/A',
                'Test F1': f"{r['test_f1']:.4f}" if r['test_f1'] else 'N/A'
            }
            # Aggiungi metriche MC Dropout se disponibili
            if r.get('mean_confidence') is not None:
                row['Confidence'] = f"{r['mean_confidence']:.4f}"
            if r.get('mean_epistemic_uncertainty') is not None:
                row['Epistemic Unc.'] = f"{r['mean_epistemic_uncertainty']:.4f}"
            metrics_data.append(row)
    
    if metrics_data:
        print(tabulate(metrics_data, headers='keys', tablefmt='grid'))
        
        # Confronto diretto
        if len(metrics_data) >= 2:
            print("\n🔍 DIRECT COMPARISON:")
            baseline = results[0]
            for i in range(1, len(results)):
                comp = results[i]
                if comp and comp['test_accuracy'] and baseline['test_accuracy']:
                    diff_acc = comp['test_accuracy'] - baseline['test_accuracy']
                    diff_f1 = comp['test_f1'] - baseline['test_f1'] if comp['test_f1'] and baseline['test_f1'] else 0
                    
                    symbol = "📈" if diff_acc > 0 else "📉" if diff_acc < 0 else "➡️"
                    print(f"\n{comp['experiment']} vs {baseline['experiment']}:")
                    print(f"  {symbol} Accuracy: {diff_acc:+.4f} ({diff_acc*100:+.2f}%)")
                    print(f"  {symbol} F1 Score: {diff_f1:+.4f} ({diff_f1*100:+.2f}%)")
    
    # Focus points
    if 'comparison_focus' in suite_info:
        print("\n🎯 COMPARISON FOCUS:")
        for focus in suite_info['comparison_focus']:
            print(f"  • {focus}")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Run experiment suites")
    parser.add_argument("suite", help="Name of experiment suite to run")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    if 'experiment_suites' not in config:
        print("❌ No experiment_suites section in config")
        return
    
    if args.suite not in config['experiment_suites']:
        print(f"❌ Suite '{args.suite}' not found")
        print(f"\nAvailable suites:")
        for suite_name in config['experiment_suites'].keys():
            print(f"  • {suite_name}")
        return
    
    suite_info = config['experiment_suites'][args.suite]
    experiments = suite_info['experiments']
    
    print("\n" + "="*80)
    print(f"🚀 EXPERIMENT SUITE: {args.suite}")
    print("="*80)
    print(f"\nDescription: {suite_info['description']}")
    print(f"\nExperiments to run ({len(experiments)}):")
    for exp in experiments:
        exp_config = config.get(exp, {})
        params = exp_config.get('params', {})
        mcdo_status = "WITH MC Dropout" if params.get('use_mcdo', False) else "Baseline"
        print(f"  • {exp}: {params.get('n_iteration', '?')} iterations, {params.get('epochs', '?')} epochs - {mcdo_status}")
    
    if args.dry_run:
        print("\n🔍 Dry run mode - no experiments will be executed")
        return
    
    input("\n⚠️  Press ENTER to start experiments or Ctrl+C to cancel...")
    
    # Run experiments
    print("\n" + "="*80)
    print("🔄 EXECUTING EXPERIMENTS")
    print("="*80 + "\n")
    
    results = []
    for exp_name in experiments:
        result = run_single_experiment(exp_name, config)
        results.append(result)
    
    # Save results
    result_file = save_results(args.suite, results)
    
    # Generate comparison
    generate_comparison_table(results, suite_info)
    
    print(f"\n✨ Suite '{args.suite}' completed!")

if __name__ == "__main__":
    main()
