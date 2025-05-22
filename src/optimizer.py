"""
Parameter optimization module for Bitcoin Mean Reversion Statistical Analyzer.

This module contains the ParameterOptimizer class which implements grid search
functionality to find optimal parameters for the break analysis.
"""

import pandas as pd
import numpy as np
import itertools
import logging
import os
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """
    Parameter optimizer for the Bitcoin Mean Reversion Statistical Analyzer.
    
    This class implements grid search functionality to find optimal parameters
    for the break analysis based on various performance metrics.
    """
    
    def __init__(self, analyzer, evaluation_metric="profit_factor"):
        """
        Initialize the ParameterOptimizer with an analyzer instance.
        
        Args:
            analyzer: BreakAnalyzer instance
            evaluation_metric: Metric to use for evaluation ('profit_factor', 
                              'win_rate', 'expectancy', or 'sharpe_ratio')
        """
        self.analyzer = analyzer
        self.evaluation_metric = evaluation_metric
        self.results_history = []
    
    def calculate_performance_metrics(self, results_df):
        """
        Calculate performance metrics from analysis results.
        
        Args:
            results_df: DataFrame with analysis results
            
        Returns:
            dict: Performance metrics
        """
        if results_df.empty:
            logger.warning("No results to calculate metrics")
            return {
                "win_rate": 0,
                "profit_factor": 0,
                "expectancy": 0,
                "sharpe_ratio": 0,
                "total_trades": 0
            }
        
        # For each break type, calculate metrics
        metrics = {}
        
        for break_type_key in ['high', 'low']: # Changed from ['high_break', 'low_break']
            # Filter results for this break type
            type_df = results_df[results_df['break_type'] == break_type_key] # Use break_type_key
            
            if type_df.empty:
                continue
                
            # For high breaks, we assume reversal strategy (sell on break)
            # For low breaks, we assume reversal strategy (buy on break)
            
            if break_type_key == 'high': # Changed from 'high_break'
                # For high breaks (sell on break), profit comes from reversal
                wins = type_df[type_df['outcome'].isin(['reversal', 'reversal_then_continuation'])]
                losses = type_df[type_df['outcome'].isin(['continuation', 'continuation_then_reversal'])]
                
                # Define reward as max reversal percentage
                rewards = wins['max_reversal_pct']
                # Define risk as max continuation percentage
                risks = losses['max_continuation_pct']
            else: # low break
                # For low breaks (buy on break), profit comes from reversal
                wins = type_df[type_df['outcome'].isin(['reversal', 'reversal_then_continuation'])]
                losses = type_df[type_df['outcome'].isin(['continuation', 'continuation_then_reversal'])]
                
                # Define reward as max reversal percentage
                rewards = wins['max_reversal_pct']
                # Define risk as max continuation percentage
                risks = losses['max_continuation_pct']
            
            # Calculate metrics
            total_trades = len(type_df)
            win_count = len(wins)
            loss_count = len(losses)
            
            # Calculate win rate
            win_rate = win_count / total_trades if total_trades > 0 else 0
            
            # Calculate profit factor
            total_rewards = rewards.sum() if not rewards.empty else 0
            total_risks = risks.sum() if not risks.empty else 0
            profit_factor = total_rewards / total_risks if total_risks > 0 else 0
            
            # Calculate expectancy (average reward to risk ratio)
            avg_reward = rewards.mean() if not rewards.empty else 0
            avg_risk = risks.mean() if not risks.empty else 0
            expectancy = (win_rate * avg_reward) - ((1 - win_rate) * avg_risk)
            
            # Calculate Sharpe ratio (simplified)
            if not rewards.empty and not risks.empty:
                all_returns = pd.concat([rewards, -risks])
                sharpe_ratio = all_returns.mean() / all_returns.std() if all_returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
                
            metrics[break_type_key] = { # Use break_type_key
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "expectancy": expectancy,
                "sharpe_ratio": sharpe_ratio,
                "total_trades": total_trades
            }
        
        # Average metrics across break types
        combined_metrics = {
            "win_rate": np.mean([m.get("win_rate", 0) for m in metrics.values()]),
            "profit_factor": np.mean([m.get("profit_factor", 0) for m in metrics.values()]),
            "expectancy": np.mean([m.get("expectancy", 0) for m in metrics.values()]),
            "sharpe_ratio": np.mean([m.get("sharpe_ratio", 0) for m in metrics.values()]),
            "total_trades": sum([m.get("total_trades", 0) for m in metrics.values()])
        }
        
        return combined_metrics
    
    def grid_search(self, 
                  confirmation_thresholds=None,
                  continuation_thresholds=None, 
                  reversal_thresholds=None,
                  window_hours=None,
                  start_date=None,
                  end_date=None):
        """
        Perform grid search to find optimal parameters.
        
        Args:
            confirmation_thresholds: List of confirmation threshold values to test
            continuation_thresholds: List of continuation threshold values to test
            reversal_thresholds: List of reversal threshold values to test
            window_hours: List of window hours values to test
            start_date: Start date for analysis (format: 'YYYY-MM-DD')
            end_date: End date for analysis (format: 'YYYY-MM-DD')
            
        Returns:
            dict: Best parameters and their performance metrics
        """
        # Default parameter values if not provided
        if confirmation_thresholds is None:
            confirmation_thresholds = [0.05, 0.1, 0.2]
        if continuation_thresholds is None:
            continuation_thresholds = [0.5, 1.0, 1.5, 2.0]
        if reversal_thresholds is None:
            reversal_thresholds = [0.25, 0.5, 0.75, 1.0]
        if window_hours is None:
            window_hours = [4, 8, 12, 24]
            
        # Generate all combinations of parameters
        param_combinations = list(itertools.product(
            confirmation_thresholds,
            continuation_thresholds,
            reversal_thresholds,
            window_hours
        ))
        
        logger.info(f"Running grid search with {len(param_combinations)} parameter combinations")
        
        # Store results
        results = []
        best_metric = 0
        best_params = None
        best_metrics = None
        
        # Track progress
        total_combinations = len(param_combinations)
        start_time = time.time()
        
        for i, params in enumerate(param_combinations):
            confirmation_threshold, continuation_threshold, reversal_threshold, window_hour = params
            
            # Update analyzer parameters
            analysis_params = {
                'break_detection': {
                    'confirmation_threshold': confirmation_threshold
                },
                'analysis': {
                    'continuation_threshold': continuation_threshold,
                    'reversal_threshold': reversal_threshold,
                    'window_hours': window_hour
                }
            }
            
            # Log progress
            progress = (i+1) / total_combinations * 100
            elapsed = time.time() - start_time
            remaining = (elapsed / (i+1)) * (total_combinations - (i+1)) if i > 0 else 0
            
            logger.info(f"Progress: {progress:.1f}% - Testing parameters: {params} - "
                        f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
            
            # Run analysis with current parameters
            flat_analysis_params = {
                'confirmation_threshold': confirmation_threshold,
                'continuation_threshold': continuation_threshold,
                'reversal_threshold': reversal_threshold,
                'window_hours': window_hour
            }
            results_df = self.analyzer.analyze_key_level_breaks(
                start_date=start_date,
                end_date=end_date,
                **flat_analysis_params # Pass flat params directly
            )
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(results_df)
            
            # Store results
            result = {
                'params': {
                    'confirmation_threshold': confirmation_threshold,
                    'continuation_threshold': continuation_threshold,
                    'reversal_threshold': reversal_threshold,
                    'window_hours': window_hour
                },
                'metrics': metrics
            }
            results.append(result)
            
            # Check if this is the best so far
            current_metric = metrics[self.evaluation_metric]
            if current_metric > best_metric:
                best_metric = current_metric
                best_params = result['params']
                best_metrics = metrics
                logger.info(f"New best parameters found: {best_params} with {self.evaluation_metric} = {best_metric}")
        
        # Save results history
        self.results_history = results
        
        # Return best parameters
        return {
            'best_params': best_params,
            'best_metrics': best_metrics,
            'all_results': results
        }
    
    def save_results(self, results, output_file=None):
        """
        Save grid search results to a file.
        
        Args:
            results: Grid search results
            output_file: Path to output file (default: 'reports/optimization_results.json')
        """
        if output_file is None:
            # Create reports directory if it doesn't exist
            reports_dir = Path('reports')
            reports_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = reports_dir / f'optimization_results_{timestamp}.json'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved to {output_file}")
    
    def generate_report(self, results, output_path=None):
        """
        Generate an optimization report.
        
        Args:
            results: Grid search results
            output_path: Path to output file (default: 'reports/optimization_report.md')
        """
        if output_path is None:
            # Create reports directory if it doesn't exist
            reports_dir = Path('reports')
            reports_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = reports_dir / f'optimization_report_{timestamp}.md'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate report content
        report = "# Bitcoin Mean Reversion Parameter Optimization Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add best parameters section
        report += "## Best Parameters\n\n"
        report += "| Parameter | Value |\n"
        report += "|-----------|-------|\n"
        for param, value in results['best_params'].items():
            report += f"| {param} | {value} |\n"
        
        # Add best metrics section
        report += "\n## Performance Metrics with Best Parameters\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        for metric, value in results['best_metrics'].items():
            report += f"| {metric} | {value:.4f} |\n"
        
        # Add top parameters section
        report += "\n## Top 10 Parameter Combinations\n\n"
        
        # Sort results by evaluation metric
        sorted_results = sorted(
            results['all_results'], 
            key=lambda x: x['metrics'][self.evaluation_metric], 
            reverse=True
        )[:10]
        
        # Table header
        report += "| Confirmation | Continuation | Reversal | Window | "
        report += f"{self.evaluation_metric} | Win Rate | Total Trades |\n"
        report += "|--------------|--------------|----------|--------|"
        report += "--------------|----------|-------------|\n"
        
        # Table rows
        for result in sorted_results:
            params = result['params']
            metrics = result['metrics']
            report += (f"| {params['confirmation_threshold']} | "
                      f"{params['continuation_threshold']} | "
                      f"{params['reversal_threshold']} | "
                      f"{params['window_hours']} | "
                      f"{metrics[self.evaluation_metric]:.4f} | "
                      f"{metrics['win_rate']:.4f} | "
                      f"{metrics['total_trades']} |\n")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Optimization report saved to {output_path}")
        
        return output_path 