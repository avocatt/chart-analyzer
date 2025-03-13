"""
Unit tests for data_loader.py module.

This module contains tests for all data processing functions in the 
Bitcoin Mean Reversion Statistical Analyzer.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import tempfile
from unittest.mock import patch, mock_open

# Add src directory to path to import data_loader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_loader import (
    load_data,
    prepare_previous_day_levels,
    filter_minute_data_by_date_range,
    get_day_minute_data,
    check_data_quality
)


class TestDataLoader(unittest.TestCase):
    """Test case for data_loader.py module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample minute data
        self.minute_data = pd.DataFrame({
            'open': [10000.0, 10005.0, 10010.0, 10015.0, 10020.0, 10025.0],
            'high': [10010.0, 10015.0, 10020.0, 10025.0, 10030.0, 10035.0],
            'low': [9990.0, 9995.0, 10000.0, 10005.0, 10010.0, 10015.0],
            'close': [10005.0, 10010.0, 10015.0, 10020.0, 10025.0, 10030.0],
            'volume': [1.5, 2.0, 1.8, 2.2, 1.9, 2.1]
        }, index=pd.DatetimeIndex([
            '2023-01-01 00:00:00', '2023-01-01 00:01:00',
            '2023-01-02 00:00:00', '2023-01-02 00:01:00',
            '2023-01-03 00:00:00', '2023-01-03 00:01:00'
        ]))

        # Create sample daily data
        self.daily_data = pd.DataFrame({
            'open': [10000.0, 10200.0, 10400.0],
            'high': [10200.0, 10400.0, 10600.0],
            'low': [9800.0, 10000.0, 10200.0],
            'close': [10100.0, 10300.0, 10500.0],
            'volume': [100.0, 120.0, 110.0]
        }, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03']))

    def test_load_data(self):
        """Test load_data function with mocked CSV files."""
        # Mock CSV content
        minute_csv = """datetime,open,high,low,close,volume
2023-01-01 00:00:00,10000.0,10010.0,9990.0,10005.0,1.5
2023-01-01 00:01:00,10005.0,10015.0,9995.0,10010.0,2.0"""
        
        daily_csv = """datetime,open,high,low,close,volume
2023-01-01,10000.0,10200.0,9800.0,10100.0,100.0"""

        # Mock open function to return our test data
        with patch('builtins.open', mock_open(read_data=minute_csv)), \
             patch('pandas.read_csv', side_effect=[
                 pd.read_csv(pd.io.common.StringIO(minute_csv)),
                 pd.read_csv(pd.io.common.StringIO(daily_csv))
             ]):
            
            minute_data, daily_data = load_data('fake_minute.csv', 'fake_daily.csv')
            
            # Check that data was loaded and processed correctly
            self.assertEqual(len(minute_data), 2)
            self.assertEqual(len(daily_data), 1)
            self.assertTrue('open' in minute_data.columns)
            self.assertTrue('high' in daily_data.columns)
            self.assertEqual(minute_data.index.name, 'datetime')
            self.assertEqual(daily_data.index.name, 'datetime')
    
    def test_load_data_file_not_found(self):
        """Test load_data function with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_data('non_existent_minute.csv', 'non_existent_daily.csv')
    
    def test_prepare_previous_day_levels(self):
        """Test prepare_previous_day_levels function."""
        prev_day_levels = prepare_previous_day_levels(self.daily_data)
        
        # Check that we have entries for all days except the first
        self.assertEqual(len(prev_day_levels), 2)
        
        # Check values for specific dates
        date_20230102 = datetime(2023, 1, 2).date()
        date_20230103 = datetime(2023, 1, 3).date()
        
        self.assertIn(date_20230102, prev_day_levels)
        self.assertIn(date_20230103, prev_day_levels)
        
        # Check that levels match expected values
        self.assertEqual(prev_day_levels[date_20230102]['high'], 10200.0)
        self.assertEqual(prev_day_levels[date_20230102]['low'], 9800.0)
        self.assertEqual(prev_day_levels[date_20230103]['high'], 10400.0)
        self.assertEqual(prev_day_levels[date_20230103]['low'], 10000.0)
    
    def test_filter_minute_data_by_date_range(self):
        """Test filter_minute_data_by_date_range function."""
        # 1. Test with exact datetime objects
        start_date = datetime(2023, 1, 2, 0, 0, 0)  # Exact datetime at 00:00:00
        filtered_data = filter_minute_data_by_date_range(
            self.minute_data, start_date, start_date
        )
        
        # Should only match entries with exactly this datetime (just one in our test data)
        self.assertEqual(len(filtered_data), 1, 
                         f"Expected 1 entry for exact datetime {start_date}, got {len(filtered_data)}")
        
        # 2. Test with string dates (should convert to full day range)
        filtered_data = filter_minute_data_by_date_range(
            self.minute_data, '2023-01-01', '2023-01-02'
        )
        
        # When using string dates, it should include all of start date and all of end date
        self.assertEqual(len(filtered_data), 4, 
                         f"Expected 4 entries from 2023-01-01 to 2023-01-02, got {len(filtered_data)}")
        
        # 3. Test with only start_date
        filtered_data = filter_minute_data_by_date_range(
            self.minute_data, start_date=datetime(2023, 1, 3, 0, 0, 0)
        )
        
        # Should include entries at or after the exact start datetime
        self.assertEqual(len(filtered_data), 2, 
                         f"Expected 2 entries on or after 2023-01-03 00:00:00, got {len(filtered_data)}")
        
        # 4. Test with only end_date
        filtered_data = filter_minute_data_by_date_range(
            self.minute_data, end_date=datetime(2023, 1, 1, 0, 0, 0)
        )
        
        # Should include entries at the exact end datetime (inclusive)
        self.assertEqual(len(filtered_data), 1, 
                         f"Expected 1 entry at 2023-01-01 00:00:00, got {len(filtered_data)}")
        
        # 5. Test with date range spanning all data
        filtered_data = filter_minute_data_by_date_range(
            self.minute_data, '2023-01-01', '2023-01-03'
        )
        
        # Should include all data
        self.assertEqual(len(filtered_data), 6,
                         f"Expected all 6 entries, got {len(filtered_data)}")
    
    def test_get_day_minute_data(self):
        """Test get_day_minute_data function."""
        # Test with datetime.date object
        day_data = get_day_minute_data(self.minute_data, datetime(2023, 1, 2).date())
        
        # Should only include data from Jan 2
        self.assertEqual(len(day_data), 2)
        self.assertTrue(all(d.date() == datetime(2023, 1, 2).date() 
                           for d in day_data.index))
        
        # Test with string date
        day_data = get_day_minute_data(self.minute_data, '2023-01-03')
        
        # Should only include data from Jan 3
        self.assertEqual(len(day_data), 2)
        self.assertTrue(all(d.date() == datetime(2023, 1, 3).date() 
                           for d in day_data.index))
        
        # Test with non-existent date
        day_data = get_day_minute_data(self.minute_data, '2023-01-04')
        
        # Should return empty DataFrame
        self.assertEqual(len(day_data), 0)
    
    def test_check_data_quality(self):
        """Test check_data_quality function."""
        # Create data with quality issues
        minute_data_with_issues = self.minute_data.copy()
        minute_data_with_issues.loc[minute_data_with_issues.index[0], 'close'] = np.nan
        
        daily_data_with_issues = self.daily_data.copy()
        # Add duplicate timestamp
        daily_data_with_issues = pd.concat([
            daily_data_with_issues, 
            pd.DataFrame({
                'open': [10000.0],
                'high': [10200.0],
                'low': [9800.0],
                'close': [10100.0],
                'volume': [100.0]
            }, index=pd.DatetimeIndex(['2023-01-01']))
        ])
        
        # Run quality check
        quality_report = check_data_quality(minute_data_with_issues, daily_data_with_issues)
        
        # Verify report contents
        self.assertEqual(quality_report['minute_data']['total_bars'], 6)
        self.assertEqual(quality_report['daily_data']['total_bars'], 4)
        self.assertEqual(quality_report['minute_data']['missing_values']['close'], 1)
        self.assertEqual(quality_report['daily_data']['duplicate_timestamps'], 1)
        self.assertIn('gaps_larger_than_5min', quality_report['minute_data'])


if __name__ == '__main__':
    unittest.main() 