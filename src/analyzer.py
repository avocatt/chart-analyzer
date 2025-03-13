"""
Core analysis module for Bitcoin Mean Reversion Statistical Analyzer.

This module contains the BreakAnalyzer class which implements the key level break
detection and analysis functionality.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yaml
import os
from pathlib import Path

# Import local modules
from src.data_loader import load_data, prepare_previous_day_levels, get_day_minute_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BreakAnalyzer:
    """
    Analyzer for detecting and analyzing price breaks of previous day's high/low levels.
    
    This class implements the core functionality for the Bitcoin Mean Reversion
    Statistical Analyzer, including break detection and outcome classification.
    """
    
    def __init__(self, minute_data_path=None, daily_data_path=None, config_path=None):
        """
        Initialize the BreakAnalyzer with data paths and configuration.
        
        Args:
            minute_data_path: Path to 1-minute OHLCV data CSV file
            daily_data_path: Path to daily OHLCV data CSV file
            config_path: Path to configuration YAML file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set data paths from config if not provided
        if minute_data_path is None and 'data' in self.config:
            minute_data_path = self.config['data'].get('minute_data')
        if daily_data_path is None and 'data' in self.config:
            daily_data_path = self.config['data'].get('daily_data')
            
        self.minute_data_path = minute_data_path
        self.daily_data_path = daily_data_path
        
        # Initialize data attributes
        self.minute_data = None
        self.daily_data = None
        self.prev_day_levels = None
        
        # Load data if paths are provided
        if minute_data_path and daily_data_path:
            self.load_data()
    
    def _load_config(self, config_path=None):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file
            
        Returns:
            dict: Configuration dictionary
        """
        # Default config
        default_config = {
            'break_detection': {
                'confirmation_threshold': 0.1,
                'max_hours_from_day_start': 24
            },
            'analysis': {
                'continuation_threshold': 1.0,
                'reversal_threshold': 0.5,
                'window_hours': 24,
                'use_utc': True
            }
        }
        
        # If config path is provided, load from file
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default configuration")
                return default_config
        
        # Check for config in default location
        default_path = os.path.join('config', 'analyzer_config.yaml')
        if os.path.exists(default_path):
            try:
                with open(default_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {default_path}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config from {default_path}: {e}")
        
        logger.info("Using default configuration")
        return default_config
    
    def load_data(self):
        """
        Load data from specified paths.
        
        Returns:
            self: For method chaining
        
        Raises:
            ValueError: If data paths are not set
        """
        if not self.minute_data_path or not self.daily_data_path:
            raise ValueError("Data paths must be set before loading data")
        
        # Load data
        self.minute_data, self.daily_data = load_data(
            self.minute_data_path,
            self.daily_data_path
        )
        
        # Prepare previous day levels
        self.prev_day_levels = prepare_previous_day_levels(self.daily_data)
        
        return self
    
    def find_level_break(self, day_data, level, break_type, confirmation_threshold):
        """
        Find the first occurrence of price breaking above/below the given level.
        
        Args:
            day_data: Minute data for a specific day
            level: Price level to watch (previous day's high or low)
            break_type: 'high' for breaks above, 'low' for breaks below
            confirmation_threshold: Minimum percentage beyond level to confirm break
            
        Returns:
            dict: Dictionary with break information or None if no break
        """
        if day_data.empty:
            return None
            
        confirmation_amount = level * confirmation_threshold / 100
        
        if break_type == 'high':
            # Find candles where high is above level + confirmation
            break_candles = day_data[day_data['high'] > level + confirmation_amount]
        else:  # low break
            # Find candles where low is below level - confirmation
            break_candles = day_data[day_data['low'] < level - confirmation_amount]
        
        if len(break_candles) == 0:
            return None
        
        # Get the first break candle
        first_break = break_candles.iloc[0]
        
        return {
            'datetime': first_break.name,
            'break_type': break_type,
            'level': level,
            'break_price': first_break['close'],
            'candle_body': abs(first_break['close'] - first_break['open'])
        }
    
    def analyze_post_break(self, minute_data, break_data, continuation_threshold, 
                          reversal_threshold, window_hours):
        """
        Analyze price action after a break.
        
        Args:
            minute_data: Full minute data DataFrame
            break_data: Dictionary with break information
            continuation_threshold: Minimum % move in break direction to count as continuation
            reversal_threshold: Minimum % move against break direction to count as reversal
            window_hours: Hours to analyze after break
            
        Returns:
            dict: Dictionary with analysis results
        """
        break_time = break_data['datetime']
        break_price = break_data['break_price']
        break_type = break_data['break_type']
        
        # Get data for analysis window
        end_time = break_time + timedelta(hours=window_hours)
        window_data = minute_data[(minute_data.index > break_time) & (minute_data.index <= end_time)]
        
        if len(window_data) == 0:
            return {**break_data, 'outcome': 'insufficient_data'}
        
        # Calculate max moves in both directions
        if break_type == 'high':
            # For high breaks, continuation is up, reversal is down
            max_continuation_pct = ((window_data['high'].max() - break_price) / break_price) * 100
            max_reversal_pct = ((break_price - window_data['low'].min()) / break_price) * 100
        else:
            # For low breaks, continuation is down, reversal is up
            max_continuation_pct = ((break_price - window_data['low'].min()) / break_price) * 100
            max_reversal_pct = ((window_data['high'].max() - break_price) / break_price) * 100
        
        # Find which happened first (continuation or reversal)
        if break_type == 'high':
            cont_threshold = break_price * (1 + continuation_threshold/100)
            rev_threshold = break_price * (1 - reversal_threshold/100)
            
            cont_data = window_data[window_data['high'] >= cont_threshold]
            rev_data = window_data[window_data['low'] <= rev_threshold]
        else:
            cont_threshold = break_price * (1 - continuation_threshold/100)
            rev_threshold = break_price * (1 + reversal_threshold/100)
            
            cont_data = window_data[window_data['low'] <= cont_threshold]
            rev_data = window_data[window_data['high'] >= rev_threshold]
        
        # Determine which came first (continuation or reversal)
        cont_time = cont_data.index.min() if not cont_data.empty else None
        rev_time = rev_data.index.min() if not rev_data.empty else None
        
        if cont_time is not None and rev_time is not None:
            first_to_occur = 'continuation' if cont_time < rev_time else 'reversal'
            first_time = min(cont_time, rev_time)
        elif cont_time is not None:
            first_to_occur = 'continuation'
            first_time = cont_time
        elif rev_time is not None:
            first_to_occur = 'reversal'
            first_time = rev_time
        else:
            first_to_occur = 'neither'
            first_time = None
        
        # Calculate time to first threshold hit
        if first_time:
            time_to_first = (first_time - break_time).total_seconds() / 60  # in minutes
        else:
            time_to_first = None
        
        # Determine final outcome category
        if max_continuation_pct >= continuation_threshold and max_reversal_pct >= reversal_threshold:
            if first_to_occur == 'continuation':
                outcome = 'continuation_then_reversal'
            else:
                outcome = 'reversal_then_continuation'
        elif max_continuation_pct >= continuation_threshold:
            outcome = 'continuation'
        elif max_reversal_pct >= reversal_threshold:
            outcome = 'reversal'
        else:
            outcome = 'sideways'
        
        return {
            **break_data,
            'max_continuation_pct': max_continuation_pct,
            'max_reversal_pct': max_reversal_pct,
            'first_to_occur': first_to_occur,
            'time_to_first_threshold': time_to_first,
            'outcome': outcome
        }
    
    def analyze_key_level_breaks(self, start_date=None, end_date=None, **kwargs):
        """
        Analyze what happens after price breaks previous day's high/low levels.
        
        Args:
            start_date: Start date for analysis (None for all available)
            end_date: End date for analysis (None for all available)
            **kwargs: Override configuration parameters
            
        Returns:
            DataFrame: Results of the analysis
        """
        if self.minute_data is None or self.daily_data is None:
            self.load_data()
        
        # Get configuration parameters
        break_detection_config = self.config.get('break_detection', {})
        analysis_config = self.config.get('analysis', {})
        
        # Allow parameter overrides
        confirmation_threshold = kwargs.get(
            'confirmation_threshold', 
            break_detection_config.get('confirmation_threshold', 0.1)
        )
        
        continuation_threshold = kwargs.get(
            'continuation_threshold', 
            analysis_config.get('continuation_threshold', 1.0)
        )
        
        reversal_threshold = kwargs.get(
            'reversal_threshold', 
            analysis_config.get('reversal_threshold', 0.5)
        )
        
        window_hours = kwargs.get(
            'window_hours', 
            analysis_config.get('window_hours', 24)
        )
        
        logger.info(f"Starting key level break analysis with parameters: "
                   f"confirmation_threshold={confirmation_threshold}, "
                   f"continuation_threshold={continuation_threshold}, "
                   f"reversal_threshold={reversal_threshold}, "
                   f"window_hours={window_hours}")
        
        results = []
        
        # Get unique days in our minute data
        days = sorted(set(self.minute_data.index.date))
        
        # Filter days by date range if provided
        if start_date:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date).date()
            days = [day for day in days if day >= start_date]
        
        if end_date:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date).date()
            days = [day for day in days if day <= end_date]
        
        for current_day in days:
            # Skip if we don't have previous day levels
            if current_day not in self.prev_day_levels:
                continue
                
            prev_high = self.prev_day_levels[current_day]['high']
            prev_low = self.prev_day_levels[current_day]['low']
            
            # Get minute data for current day
            day_data = get_day_minute_data(self.minute_data, current_day)
            
            # Look for high breaks
            high_break_data = self.find_level_break(
                day_data, prev_high, 'high', confirmation_threshold
            )
            
            if high_break_data:
                # Analyze what happens after the break
                high_break_results = self.analyze_post_break(
                    self.minute_data, high_break_data, 
                    continuation_threshold, reversal_threshold, window_hours
                )
                
                # Add date information
                high_break_results['day'] = current_day
                high_break_results['day_of_week'] = high_break_results['datetime'].dayofweek
                high_break_results['hour_of_day'] = high_break_results['datetime'].hour
                
                results.append(high_break_results)
                logger.debug(f"Found high break on {current_day}, outcome: {high_break_results['outcome']}")
            
            # Look for low breaks
            low_break_data = self.find_level_break(
                day_data, prev_low, 'low', confirmation_threshold
            )
            
            if low_break_data:
                # Analyze what happens after the break
                low_break_results = self.analyze_post_break(
                    self.minute_data, low_break_data,
                    continuation_threshold, reversal_threshold, window_hours
                )
                
                # Add date information
                low_break_results['day'] = current_day
                low_break_results['day_of_week'] = low_break_results['datetime'].dayofweek
                low_break_results['hour_of_day'] = low_break_results['datetime'].hour
                
                results.append(low_break_results)
                logger.debug(f"Found low break on {current_day}, outcome: {low_break_results['outcome']}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        logger.info(f"Analysis complete. Found {len(results_df)} breaks.")
        
        return results_df
    
    def summarize_break_statistics(self, results_df):
        """
        Summarize statistics of break analysis results.
        
        Args:
            results_df: DataFrame with analysis results
            
        Returns:
            dict: Summary statistics
        """
        if results_df.empty:
            logger.warning("No results to summarize")
            return {}
        
        # Group by break type
        grouped = results_df.groupby('break_type')
        
        summary = {}
        
        for name, group in grouped:
            total_breaks = len(group)
            
            # Count outcomes
            continuation_count = sum(group['outcome'].isin(['continuation', 'continuation_then_reversal']))
            reversal_count = sum(group['outcome'].isin(['reversal', 'reversal_then_continuation']))
            sideways_count = sum(group['outcome'] == 'sideways')
            
            # Calculate percentages
            continuation_pct = (continuation_count / total_breaks) * 100
            reversal_pct = (reversal_count / total_breaks) * 100
            sideways_pct = (sideways_count / total_breaks) * 100
            
            # Average maximum moves
            avg_continuation = group['max_continuation_pct'].mean()
            avg_reversal = group['max_reversal_pct'].mean()
            
            # First to occur stats
            first_continuation = sum(group['first_to_occur'] == 'continuation')
            first_reversal = sum(group['first_to_occur'] == 'reversal')
            first_continuation_pct = (first_continuation / total_breaks) * 100
            first_reversal_pct = (first_reversal / total_breaks) * 100
            
            # Time analysis
            avg_time_to_threshold = group['time_to_first_threshold'].mean()
            
            # Day of week analysis
            day_of_week_counts = group['day_of_week'].value_counts().to_dict()
            
            # Hour of day analysis
            hour_of_day_counts = group['hour_of_day'].value_counts().to_dict()
            
            summary[name] = {
                'total_breaks': total_breaks,
                'continuation_pct': continuation_pct,
                'reversal_pct': reversal_pct,
                'sideways_pct': sideways_pct,
                'avg_continuation': avg_continuation,
                'avg_reversal': avg_reversal,
                'first_continuation_pct': first_continuation_pct,
                'first_reversal_pct': first_reversal_pct,
                'avg_time_to_threshold': avg_time_to_threshold,
                'day_of_week_counts': day_of_week_counts,
                'hour_of_day_counts': hour_of_day_counts
            }
        
        return summary
    
    def generate_report(self, results_df, output_path=None):
        """
        Generate a report with analysis results.
        
        Args:
            results_df: DataFrame with analysis results
            output_path: Path to save the report (None for no saving)
            
        Returns:
            str: Report text
        """
        if results_df.empty:
            return "No results to report"
        
        # Get summary statistics
        summary = self.summarize_break_statistics(results_df)
        
        # Build report text
        report = []
        report.append("# Bitcoin Mean Reversion Statistical Analysis Report")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("## Analysis Parameters")
        report.append(f"- Confirmation Threshold: {self.config['break_detection']['confirmation_threshold']}%")
        report.append(f"- Continuation Threshold: {self.config['analysis']['continuation_threshold']}%")
        report.append(f"- Reversal Threshold: {self.config['analysis']['reversal_threshold']}%")
        report.append(f"- Analysis Window: {self.config['analysis']['window_hours']} hours\n")
        
        report.append("## Overall Statistics")
        report.append(f"- Total Breaks Analyzed: {len(results_df)}")
        report.append(f"- Date Range: {results_df['day'].min()} to {results_df['day'].max()}")
        report.append(f"- High Breaks: {len(results_df[results_df['break_type'] == 'high'])}")
        report.append(f"- Low Breaks: {len(results_df[results_df['break_type'] == 'low'])}\n")
        
        # Add break type specific statistics
        for break_type, stats in summary.items():
            report.append(f"## {break_type.capitalize()} Break Statistics")
            report.append(f"- Total {break_type.capitalize()} Breaks: {stats['total_breaks']}")
            report.append(f"- Continuation %: {stats['continuation_pct']:.1f}%")
            report.append(f"- Reversal %: {stats['reversal_pct']:.1f}%")
            report.append(f"- Sideways %: {stats['sideways_pct']:.1f}%")
            report.append(f"- Average Continuation Size: {stats['avg_continuation']:.2f}%")
            report.append(f"- Average Reversal Size: {stats['avg_reversal']:.2f}%")
            report.append(f"- Continuation Happened First: {stats['first_continuation_pct']:.1f}%")
            report.append(f"- Reversal Happened First: {stats['first_reversal_pct']:.1f}%")
            report.append(f"- Average Time to First Threshold: {stats['avg_time_to_threshold']:.1f} minutes\n")
        
        report_text = "\n".join(report)
        
        # Save report if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text

if __name__ == "__main__":
    # Simple test to verify the module works
    try:
        analyzer = BreakAnalyzer(
            minute_data_path="data/BTCUSDT_1m.csv",
            daily_data_path="data/BTCUSDT_1d.csv"
        )
        
        # Run analysis
        print("Running analysis...")
        results = analyzer.analyze_key_level_breaks()
        
        # Generate and print report
        print("\nAnalysis Report:")
        report = analyzer.generate_report(results)
        print(report)
        
    except Exception as e:
        print(f"Error: {e}") 