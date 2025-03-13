"""
Data loading and preprocessing module for Bitcoin Mean Reversion Statistical Analyzer.

This module handles loading data from CSV files, preprocessing, and preparing it
for analysis by the analyzer module.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(minute_data_path, daily_data_path):
    """
    Load minute and daily data from CSV files.
    
    Args:
        minute_data_path: Path to 1-minute OHLCV data CSV file
        daily_data_path: Path to daily OHLCV data CSV file
        
    Returns:
        tuple: (minute_data, daily_data) pandas DataFrames
    
    Raises:
        FileNotFoundError: If either file doesn't exist
        ValueError: If data format is invalid
    """
    logger.info(f"Loading data from {minute_data_path} and {daily_data_path}")
    
    try:
        # Load data
        minute_data = pd.read_csv(minute_data_path)
        daily_data = pd.read_csv(daily_data_path)
        
        # Convert datetime columns
        minute_data['datetime'] = pd.to_datetime(minute_data['datetime'])
        daily_data['datetime'] = pd.to_datetime(daily_data['datetime'])
        
        # Set datetime as index
        minute_data.set_index('datetime', inplace=True)
        daily_data.set_index('datetime', inplace=True)
        
        # Verify data has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for column in required_columns:
            if column not in minute_data.columns:
                raise ValueError(f"Minute data missing required column: {column}")
            if column not in daily_data.columns:
                raise ValueError(f"Daily data missing required column: {column}")
        
        logger.info(f"Successfully loaded {len(minute_data)} minute bars and {len(daily_data)} daily bars")
        
        return minute_data, daily_data
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid data format: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise

def prepare_previous_day_levels(daily_data):
    """
    Prepare a dictionary of previous day's high and low levels for each date.
    
    Args:
        daily_data: DataFrame with daily OHLCV data
        
    Returns:
        dict: Dictionary with date as key and dict of {'high': value, 'low': value} as value
    """
    logger.info("Preparing previous day levels dictionary")
    
    # Create dictionary to store previous day's levels
    prev_day_levels = {}
    
    # Get list of dates in the daily data
    dates = daily_data.index.date
    
    # For each date, store the previous day's high and low
    for i in range(1, len(dates)):
        current_date = dates[i]
        prev_date = dates[i-1]
        
        prev_day_levels[current_date] = {
            'high': daily_data.loc[daily_data.index.date == prev_date, 'high'].iloc[0],
            'low': daily_data.loc[daily_data.index.date == prev_date, 'low'].iloc[0]
        }
    
    logger.info(f"Created previous day levels for {len(prev_day_levels)} days")
    return prev_day_levels

def filter_minute_data_by_date_range(minute_data, start_date=None, end_date=None):
    """
    Filter minute data to specified date range.
    
    Args:
        minute_data: DataFrame with minute OHLCV data
        start_date: Start date (inclusive), as datetime or string 'YYYY-MM-DD'
        end_date: End date (inclusive), as datetime or string 'YYYY-MM-DD'
        
    Returns:
        DataFrame: Filtered minute data
    """
    logger.info(f"Filtering minute data from {start_date} to {end_date}")
    
    filtered_data = minute_data.copy()
    
    if start_date:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        filtered_data = filtered_data[filtered_data.index >= start_date]
    
    if end_date:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            # Make end_date inclusive by setting it to end of day
            end_date = end_date + timedelta(days=1) - timedelta(microseconds=1)
        filtered_data = filtered_data[filtered_data.index <= end_date]
    
    logger.info(f"Filtered data contains {len(filtered_data)} minute bars")
    return filtered_data

def get_day_minute_data(minute_data, date):
    """
    Get all minute data for a specific date.
    
    Args:
        minute_data: DataFrame with minute OHLCV data
        date: The date to get data for, as datetime.date or string 'YYYY-MM-DD'
        
    Returns:
        DataFrame: Minute data for the specified date
    """
    if isinstance(date, str):
        date = pd.to_datetime(date).date()
    
    return minute_data[minute_data.index.date == date]

def check_data_quality(minute_data, daily_data):
    """
    Perform data quality checks and return a report.
    
    Args:
        minute_data: DataFrame with minute OHLCV data
        daily_data: DataFrame with daily OHLCV data
        
    Returns:
        dict: Dictionary with data quality metrics
    """
    logger.info("Performing data quality checks")
    
    quality_report = {
        'minute_data': {
            'total_bars': len(minute_data),
            'date_range': f"{minute_data.index.min()} to {minute_data.index.max()}",
            'missing_values': minute_data.isna().sum().to_dict(),
            'duplicate_timestamps': minute_data.index.duplicated().sum(),
        },
        'daily_data': {
            'total_bars': len(daily_data),
            'date_range': f"{daily_data.index.min()} to {daily_data.index.max()}",
            'missing_values': daily_data.isna().sum().to_dict(),
            'duplicate_timestamps': daily_data.index.duplicated().sum(),
        }
    }
    
    # Check for missing days in daily data
    all_days = pd.date_range(start=daily_data.index.min(), end=daily_data.index.max(), freq='D')
    missing_days = [day for day in all_days if day not in daily_data.index]
    quality_report['daily_data']['missing_days'] = len(missing_days)
    
    # Check for large gaps in minute data
    minute_data_sorted = minute_data.sort_index()
    time_diffs = minute_data_sorted.index[1:] - minute_data_sorted.index[:-1]
    large_gaps = time_diffs[time_diffs > timedelta(minutes=5)]
    quality_report['minute_data']['gaps_larger_than_5min'] = len(large_gaps)
    
    logger.info("Data quality check completed")
    return quality_report

if __name__ == "__main__":
    # Simple test to verify the module works
    try:
        minute_data, daily_data = load_data(
            "data/BTCUSDT_1m.csv",
            "data/BTCUSDT_1d.csv"
        )
        
        print("Data loaded successfully!")
        print(f"Minute data: {len(minute_data)} rows")
        print(f"Daily data: {len(daily_data)} rows")
        
        # Print sample of data
        print("\nSample of minute data:")
        print(minute_data.head())
        
        print("\nSample of daily data:")
        print(daily_data.head())
        
        # Get previous day levels
        prev_day_levels = prepare_previous_day_levels(daily_data)
        print("\nSample of previous day levels:")
        for date, levels in list(prev_day_levels.items())[:3]:
            print(f"{date}: High={levels['high']}, Low={levels['low']}")
        
        # Run data quality check
        quality_report = check_data_quality(minute_data, daily_data)
        print("\nData quality report:")
        print(quality_report)
        
    except Exception as e:
        print(f"Error: {e}") 