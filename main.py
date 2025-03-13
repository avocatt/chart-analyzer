"""
Main script for Bitcoin Mean Reversion Statistical Analyzer.

This script serves as the entry point for running the analyzer from the command line.
"""

import argparse
import os
import sys
import logging
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import analyzer
from src.analyzer import BreakAnalyzer
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
    
    return parser.parse_args()

def main():
    """Main function to run the analyzer."""
    # Parse command line arguments
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
    
    # Process output path
    output_path = args.output.replace('{timestamp}', timestamp)
    
    # Initialize analyzer
    analyzer = BreakAnalyzer(
        minute_data_path=args.minute_data,
        daily_data_path=args.daily_data,
        config_path=args.config
    )
    
    # Prepare analysis parameters
    analysis_params = {}
    
    if args.confirmation_threshold is not None:
        analysis_params['confirmation_threshold'] = args.confirmation_threshold
    
    if args.continuation_threshold is not None:
        analysis_params['continuation_threshold'] = args.continuation_threshold
    
    if args.reversal_threshold is not None:
        analysis_params['reversal_threshold'] = args.reversal_threshold
    
    if args.window_hours is not None:
        analysis_params['window_hours'] = args.window_hours
    
    # Run analysis
    logger.info("Running analysis...")
    results = analyzer.analyze_key_level_breaks(
        start_date=args.start_date,
        end_date=args.end_date,
        **analysis_params
    )
    
    # Generate and save report
    logger.info(f"Generating report and saving to {output_path}")
    report = analyzer.generate_report(results, output_path=output_path)
    
    # Save results to CSV if requested
    csv_path = None
    if args.csv_output:
        csv_path = args.csv_output.replace('{timestamp}', timestamp)
        logger.info(f"Saving results to CSV: {csv_path}")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results.to_csv(csv_path, index=False)
    
    # Generate visualization charts if requested
    if args.charts and not results.empty:
        charts_dir = args.charts_dir.replace('{timestamp}', timestamp)
        logger.info(f"Generating charts in {charts_dir}")
        charts = generate_report_charts(results, charts_dir)
        logger.info(f"Generated {len(charts)} charts")
    
    logger.info("Analysis complete")
    
    # Print summary to console
    print("\nAnalysis Summary:")
    print(f"Total breaks analyzed: {len(results)}")
    
    if not results.empty:
        high_breaks = len(results[results['break_type'] == 'high'])
        low_breaks = len(results[results['break_type'] == 'low'])
        
        print(f"High breaks: {high_breaks}")
        print(f"Low breaks: {low_breaks}")
        
        # Get outcome counts
        continuation_count = len(results[results['outcome'] == 'continuation'])
        reversal_count = len(results[results['outcome'] == 'reversal'])
        
        cont_then_rev = len(results[results['outcome'] == 'continuation_then_reversal'])
        rev_then_cont = len(results[results['outcome'] == 'reversal_then_continuation'])
        
        sideways_count = len(results[results['outcome'] == 'sideways'])
        
        print(f"Continuation outcomes: {continuation_count} ({continuation_count/len(results)*100:.1f}%)")
        print(f"Reversal outcomes: {reversal_count} ({reversal_count/len(results)*100:.1f}%)")
        print(f"Continuation then reversal: {cont_then_rev} ({cont_then_rev/len(results)*100:.1f}%)")
        print(f"Reversal then continuation: {rev_then_cont} ({rev_then_cont/len(results)*100:.1f}%)")
        print(f"Sideways: {sideways_count} ({sideways_count/len(results)*100:.1f}%)")
        
        print(f"\nFull report saved to: {output_path}")
        
        if csv_path:
            print(f"CSV results saved to: {csv_path}")
        
        if args.charts:
            print(f"Charts saved to: {charts_dir}")

if __name__ == "__main__":
    main() 