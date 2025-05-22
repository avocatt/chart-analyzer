"""
Main script for Bitcoin Mean Reversion Statistical Analyzer.

This script serves as the entry point for running the analyzer from the command line.
"""

import argparse
import os
import sys
import logging
from datetime import datetime
from pathlib import Path # Add this import

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import analyzer
from src.analyzer import BreakAnalyzer
from src.optimizer import ParameterOptimizer
from visualization.chart_generator import generate_report_charts

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Bitcoin Mean Reversion Statistical Analyzer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--minute-data',
        type=str,
        default='data/BTCUSDT_1m.csv',
        help='Path to 1-minute OHLCV data CSV file'
    )
    
    parser.add_argument(
        '--daily-data',
        type=str,
        default='data/BTCUSDT_1d.csv',
        help='Path to daily OHLCV data CSV file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/analyzer_config.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for analysis (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for analysis (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--confirmation-threshold',
        type=float,
        help='Minimum percentage beyond level to confirm break'
    )
    
    parser.add_argument(
        '--continuation-threshold',
        type=float,
        help='Minimum percentage move in break direction to count as continuation'
    )
    
    parser.add_argument(
        '--reversal-threshold',
        type=float,
        help='Minimum percentage move against break direction to count as reversal'
    )
    
    parser.add_argument(
        '--window-hours',
        type=int,
        help='Hours to analyze price action after a break'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='reports/analysis_report_{timestamp}.md',
        help='Path to save analysis report. Use {timestamp} for automatic timestamping.'
    )
    
    parser.add_argument(
        '--csv-output',
        type=str,
        help='Path to save analysis results as CSV'
    )
    
    parser.add_argument(
        '--charts',
        action='store_true',
        help='Generate visualization charts'
    )
    
    parser.add_argument(
        '--charts-dir',
        type=str,
        default='reports/charts',
        help='Directory to save generated charts'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Add parameter optimization arguments
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Run parameter optimization'
    )

    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Run analysis. This is the default action if --optimize is not specified.'
    )

    parser.add_argument(
        '--metric',
        type=str,
        choices=['profit_factor', 'win_rate', 'expectancy', 'sharpe_ratio'],
        default='profit_factor',
        help='Metric to optimize for'
    )
    
    parser.add_argument(
        '--confirmation-thresholds',
        type=float,
        nargs='+',
        default=[0.05, 0.1, 0.2],
        help='Confirmation threshold values to test'
    )
    
    parser.add_argument(
        '--continuation-thresholds',
        type=float,
        nargs='+',
        default=[0.5, 1.0, 1.5, 2.0],
        help='Continuation threshold values to test'
    )
    
    parser.add_argument(
        '--reversal-thresholds',
        type=float,
        nargs='+',
        default=[0.25, 0.5, 0.75, 1.0],
        help='Reversal threshold values to test'
    )
    
    parser.add_argument(
        '--window-hours-list',
        type=int,
        nargs='+',
        default=[4, 8, 12, 24],
        help='Window hours values to test'
    )
    
    return parser.parse_args()

def main():
    """Main function to run the analyzer."""
    args = parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Bitcoin Mean Reversion Statistical Analyzer")
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Process output path for analysis report
    output_path = args.output.replace('{timestamp}', timestamp)
    
    # Create analyzer
    analyzer = BreakAnalyzer(
        minute_data_path=args.minute_data,
        daily_data_path=args.daily_data,
        config_path=args.config
    )
    
    results_df_for_analysis_output = None # Will hold DF for report, CSV, charts, summary
    
    # If optimization is requested, run parameter optimization
    if args.optimize:
        logger.info("Running parameter optimization")
        
        # Create optimizer
        optimizer = ParameterOptimizer(analyzer, evaluation_metric=args.metric)
        
        # Run grid search
        opt_results = optimizer.grid_search(
            confirmation_thresholds=args.confirmation_thresholds,
            continuation_thresholds=args.continuation_thresholds,
            reversal_thresholds=args.reversal_thresholds,
            window_hours=args.window_hours_list,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Generate optimization report
        # Determine optimization report path - could be different from analysis report path
        opt_report_dir = Path(args.output).parent if '{timestamp}' in args.output else Path('reports')
        opt_report_dir.mkdir(parents=True, exist_ok=True)
        opt_report_filename = f"optimization_report_{timestamp}.md"
        # If args.output is a specific file not containing {timestamp}, use it for optimization report
        # This logic might need refinement based on desired behavior for --output with --optimize
        optimization_report_path = opt_report_dir / opt_report_filename
        if args.output and '{timestamp}' not in args.output and not Path(args.output).is_dir():
             optimization_report_path = args.output # User specified a fixed output file

        optimizer.generate_report(opt_results, output_path=str(optimization_report_path))
        
        # Save raw optimization results
        raw_opt_results_filename = f"optimization_results_{timestamp}.json"
        raw_optimization_results_path = opt_report_dir / raw_opt_results_filename
        optimizer.save_results(opt_results, output_file=str(raw_optimization_results_path))
        
        logger.info(f"Parameter optimization completed. Report: {optimization_report_path}, Results: {raw_optimization_results_path}")
        
        # If --analyze is also specified, run analysis using the best parameters found
        if args.analyze:
            logger.info("Using best parameters for analysis after optimization.")
            best_params = opt_results['best_params'] # This is a flat dict
            
            results_df_for_analysis_output = analyzer.analyze_key_level_breaks(
                start_date=args.start_date,
                end_date=args.end_date,
                **best_params # Pass flat best_params dict
            )
            
            # Generate analysis report (this will use the main output_path)
            if results_df_for_analysis_output is not None and not results_df_for_analysis_output.empty:
                logger.info(f"Generating analysis report (from optimized params) to {output_path}")
                analyzer.generate_report(results_df_for_analysis_output, output_path=output_path)
    
    # Standalone analysis or default action (if not optimizing, or if optimizing but --analyze was not also passed to trigger post-opt analysis)
    # This block runs if:
    # 1. --analyze is true AND --optimize is false (standalone analysis)
    # 2. Neither --optimize nor --analyze is true (default action is analysis)
    elif args.analyze or (not args.optimize and not args.analyze):
        action_type = "Standalone" if args.analyze else "Default"
        logger.info(f"Running {action_type} analysis.")
        
        # Prepare analysis parameters from CLI or config
        standalone_analysis_params = {}
        if args.confirmation_threshold is not None:
            standalone_analysis_params['confirmation_threshold'] = args.confirmation_threshold
        if args.continuation_threshold is not None:
            standalone_analysis_params['continuation_threshold'] = args.continuation_threshold
        if args.reversal_threshold is not None:
            standalone_analysis_params['reversal_threshold'] = args.reversal_threshold
        if args.window_hours is not None:
            standalone_analysis_params['window_hours'] = args.window_hours
        
        # Run analysis
        results_df_for_analysis_output = analyzer.analyze_key_level_breaks(
            start_date=args.start_date,
            end_date=args.end_date,
            **standalone_analysis_params
        )
        
        # Generate and save analysis report
        if results_df_for_analysis_output is not None and not results_df_for_analysis_output.empty:
            logger.info(f"Generating report for {action_type} analysis and saving to {output_path}")
            analyzer.generate_report(results_df_for_analysis_output, output_path=output_path)

    # Common output handling for any analysis run that produced results_df_for_analysis_output
    csv_path_final = None # To store actual path for summary
    charts_dir_final = None # To store actual path for summary

    if results_df_for_analysis_output is not None and not results_df_for_analysis_output.empty:
        # Save results to CSV if requested
        if args.csv_output:
            csv_path_final = args.csv_output.replace('{timestamp}', timestamp)
            logger.info(f"Saving results to CSV: {csv_path_final}")
            # Ensure directory exists
            Path(csv_path_final).parent.mkdir(parents=True, exist_ok=True)
            results_df_for_analysis_output.to_csv(csv_path_final, index=False)
        
        # Generate visualization charts if requested
        if args.charts:
            charts_dir_final = args.charts_dir.replace('{timestamp}', timestamp)
            logger.info(f"Generating charts in {charts_dir_final}")
            # Ensure directory exists
            Path(charts_dir_final).mkdir(parents=True, exist_ok=True)
            charts = generate_report_charts(results_df_for_analysis_output, charts_dir_final)
            logger.info(f"Generated {len(charts)} charts")
        
        logger.info("Analysis processing complete.")
        
        # Print summary to console
        print("\nAnalysis Summary:")
        print(f"Total breaks analyzed: {len(results_df_for_analysis_output)}")
        
        high_breaks = len(results_df_for_analysis_output[results_df_for_analysis_output['break_type'] == 'high'])
        low_breaks = len(results_df_for_analysis_output[results_df_for_analysis_output['break_type'] == 'low'])
            
        print(f"High breaks: {high_breaks}")
        print(f"Low breaks: {low_breaks}")
            
        # Get outcome counts
        continuation_count = len(results_df_for_analysis_output[results_df_for_analysis_output['outcome'] == 'continuation'])
        reversal_count = len(results_df_for_analysis_output[results_df_for_analysis_output['outcome'] == 'reversal'])
        cont_then_rev = len(results_df_for_analysis_output[results_df_for_analysis_output['outcome'] == 'continuation_then_reversal'])
        rev_then_cont = len(results_df_for_analysis_output[results_df_for_analysis_output['outcome'] == 'reversal_then_continuation'])
        sideways_count = len(results_df_for_analysis_output[results_df_for_analysis_output['outcome'] == 'sideways'])
            
        print(f"Continuation outcomes: {continuation_count} ({continuation_count/len(results_df_for_analysis_output)*100:.1f}%)")
        print(f"Reversal outcomes: {reversal_count} ({reversal_count/len(results_df_for_analysis_output)*100:.1f}%)")
        print(f"Continuation then reversal: {cont_then_rev} ({cont_then_rev/len(results_df_for_analysis_output)*100:.1f}%)")
        print(f"Reversal then continuation: {rev_then_cont} ({rev_then_cont/len(results_df_for_analysis_output)*100:.1f}%)")
        print(f"Sideways: {sideways_count} ({sideways_count/len(results_df_for_analysis_output)*100:.1f}%)")
            
        print(f"\nFull analysis report saved to: {output_path}")
            
        if csv_path_final:
            print(f"CSV results saved to: {csv_path_final}")
            
        if args.charts and charts_dir_final:
            print(f"Charts saved to: {charts_dir_final}")

    elif (args.optimize and args.analyze) or args.analyze or (not args.optimize and not args.analyze):
        # This case means an analysis was attempted (either post-optimization, standalone, or default) but yielded no results.
        logger.info("Analysis ran but produced no break data to summarize or output.")
    else:
        # This case means only --optimize was run, without --analyze, so no analysis results to summarize here.
        # The optimization report has already been generated.
        logger.info("Optimization finished. No standard analysis was run in this session.")

if __name__ == "__main__":
    main() 