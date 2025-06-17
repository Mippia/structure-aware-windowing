#!/usr/bin/env python3

import subprocess
import pandas as pd
import json
import time
from datetime import datetime
import os
import sys
import argparse
import re

def run_single_experiment(method, feature, model, epochs=50, batch_size=32, folds=5, data_dir='gtzan_preprocessed', learning_rate=0.001):
    start_time = time.time()
    
    if model == 'mert':
        batch_size = min(batch_size, 4)
    
    cmd = [
        'python', 'train.py',
        '--method', method,
        '--feature', feature,
        '--model', model,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--folds', str(folds),
        '--data_dir', data_dir,
        '--learning_rate', str(learning_rate)
    ]
    
    print(f"Running: {method} + {feature} + {model}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            metrics = parse_output(result.stdout)
            return {
                'status': 'success',
                'method': method,
                'feature': feature,
                'model': model,
                'duration_minutes': duration / 60,
                'metrics': metrics,
                'output': result.stdout,
                'error': None
            }
        else:
            return {
                'status': 'failed',
                'method': method,
                'feature': feature,
                'model': model,
                'duration_minutes': duration / 60,
                'metrics': None,
                'output': result.stdout,
                'error': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'method': method,
            'feature': feature,
            'model': model,
            'duration_minutes': 120,
            'metrics': None,
            'output': None,
            'error': 'Timeout after 2 hours'
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            'status': 'error',
            'method': method,
            'feature': feature,
            'model': model,
            'duration_minutes': duration / 60,
            'metrics': None,
            'output': None,
            'error': str(e)
        }

def parse_output(output):
    metrics = {}
    
    patterns = {
        'accuracy': r'Accuracy: ([\d.]+)%',
        'f1': r'F1 Score: ([\d.]+)%',
        'precision': r'Precision: ([\d.]+)%',
        'recall': r'Recall: ([\d.]+)%'
    }
    
    for metric, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[metric] = float(match.group(1))
        else:
            metrics[metric] = None
    
    return metrics

def generate_experiment_configs():
    methods = ['method2_5sec', 'method3_2bar']
    features = ['mel', 'cqt']
    models = ['simple_cnn']
    
    configs = []
    
    for method in methods:
        for feature in features:
            for model in models:
                configs.append({
                    'method': method,
                    'feature': feature,
                    'model': model
                })
    
    for method in methods:
        configs.append({
            'method': method,
            'feature': 'mel',
            'model': 'mert'
        })
    
    return configs

def save_results(results, output_dir='experiment_results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_path = os.path.join(output_dir, f'all_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    summary_data = []
    for result in results:
        row = {
            'method': result['method'],
            'feature': result['feature'],
            'model': result['model'],
            'status': result['status'],
            'duration_minutes': result['duration_minutes']
        }
        
        if result['status'] == 'success' and result['metrics']:
            row['accuracy'] = result['metrics'].get('accuracy')
            row['f1'] = result['metrics'].get('f1')
            row['precision'] = result['metrics'].get('precision')
            row['recall'] = result['metrics'].get('recall')
        else:
            row['error'] = result.get('error', 'Unknown error')
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, f'summary_{timestamp}.csv')
    summary_df.to_csv(csv_path, index=False)
    
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        success_data = []
        for result in successful_results:
            metrics = result['metrics']
            success_data.append({
                'method': result['method'],
                'feature': result['feature'],
                'model': result['model'],
                'accuracy': metrics.get('accuracy'),
                'f1': metrics.get('f1'),
                'precision': metrics.get('precision'),
                'recall': metrics.get('recall'),
                'duration_minutes': result['duration_minutes']
            })
        
        success_df = pd.DataFrame(success_data)
        success_csv_path = os.path.join(output_dir, f'successful_experiments_{timestamp}.csv')
        success_df.to_csv(success_csv_path, index=False)
    
    return json_path, csv_path

def print_summary(results):
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print(f"\nSUMMARY:")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        best_result = max(successful, key=lambda x: x['metrics'].get('accuracy', 0))
        metrics = best_result['metrics']
        print(f"\nBest Result:")
        print(f"  {best_result['method']} + {best_result['feature']} + {best_result['model']}")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.2f}%")
        print(f"  F1: {metrics.get('f1', 0):.2f}%")
        print(f"  Precision: {metrics.get('precision', 0):.2f}%")
        print(f"  Recall: {metrics.get('recall', 0):.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--data_dir', default='gtzan_preprocessed')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--output_dir', default='experiment_results')
    
    args = parser.parse_args()
    
    if not os.path.exists('train.py'):
        print("Error: train.py not found")
        sys.exit(1)
    
    configs = generate_experiment_configs()
    total_experiments = len(configs)
    
    print(f"Total experiments: {total_experiments}")
    print(f"Configuration: epochs={args.epochs}, folds={args.folds}")
    
    results = []
    start_time = time.time()
    
    for i, config in enumerate(configs):
        print(f"\nExperiment {i+1}/{total_experiments}")
        
        result = run_single_experiment(
            method=config['method'],
            feature=config['feature'],
            model=config['model'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            folds=args.folds,
            data_dir=args.data_dir,
            learning_rate=args.learning_rate
        )
        
        results.append(result)
        
        if result['status'] == 'success':
            metrics = result['metrics']
            print(f"SUCCESS! Duration: {result['duration_minutes']:.1f} min")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.2f}%")
            print(f"  F1: {metrics.get('f1', 0):.2f}%")
        else:
            print(f"FAILED! {result.get('error', 'Unknown error')}")
    
    total_duration = time.time() - start_time
    print(f"\nAll experiments completed in {total_duration/3600:.1f} hours")
    
    save_results(results, args.output_dir)
    print_summary(results)

if __name__ == '__main__':
    main()