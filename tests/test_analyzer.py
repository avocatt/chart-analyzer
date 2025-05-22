"""
Unit tests for analyzer.py module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os
import sys

# Add src directory to path to import analyzer and data_loader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from analyzer import BreakAnalyzer
from data_loader import prepare_previous_day_levels, get_day_minute_data

class TestBreakAnalyzer(unittest.TestCase):
    """Test case for BreakAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample daily data:
        # 2023-01-01: H=120, L=80
        # 2023-01-02: H=105, L=95
        self.daily_data_df = pd.DataFrame({
            'open':   [100, 100],
            'high':   [120, 105],
            'low':    [ 80,  95],
            'close':  [110, 100],
            'volume': [1000, 1000]
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02']))

        # Sample minute data for 2023-01-02. Prev Day H=120, L=80
        # Window for analysis: break_time + 24 hours (default)
        self.minute_data_list = []
        # Day 2023-01-02 data
        # High break example: level 120
        self.minute_data_list.extend([
            {'datetime': '2023-01-02 08:00:00', 'open': 118, 'high': 119, 'low': 117, 'close': 118.5, 'volume': 10},
            {'datetime': '2023-01-02 08:01:00', 'open': 118.5, 'high': 120.0, 'low': 118, 'close': 119.5, 'volume': 10}, # Touches 120
            # Break candle (high > 120 + 0.1% of 120 = 120.12)
            {'datetime': '2023-01-02 08:02:00', 'open': 119.5, 'high': 120.5, 'low': 119, 'close': 120.2, 'volume': 10}, # break_price = 120.2
            # Post-break: Continuation (1% of 120.2 = 1.202; target = 121.402)
            {'datetime': '2023-01-02 08:03:00', 'open': 120.2, 'high': 121.5, 'low': 120.1, 'close': 121.45, 'volume': 10}, # Hits continuation
            # Post-break: Reversal (0.5% of 120.2 = 0.601; target = 119.599)
            {'datetime': '2023-01-02 08:04:00', 'open': 121.45, 'high': 121.5, 'low': 119.5, 'close': 119.55, 'volume': 10}, # Hits reversal
        ])
        # Low break example: level 80
        self.minute_data_list.extend([
            {'datetime': '2023-01-02 09:00:00', 'open': 82, 'high': 83, 'low': 81.5, 'close': 82.5, 'volume': 10},
            {'datetime': '2023-01-02 09:01:00', 'open': 82.5, 'high': 82.6, 'low': 80.0, 'close': 80.5, 'volume': 10}, # Touches 80
            # Break candle (low < 80 - 0.1% of 80 = 79.92)
            {'datetime': '2023-01-02 09:02:00', 'open': 80.5, 'high': 80.6, 'low': 79.5, 'close': 79.8, 'volume': 10}, # break_price = 79.8
            # Post-break: Continuation (1% of 79.8 = 0.798; target = 79.002)
            {'datetime': '2023-01-02 09:03:00', 'open': 79.8, 'high': 79.9, 'low': 78.9, 'close': 79.0, 'volume': 10}, # Hits continuation
        ])

        self.minute_data_df = pd.DataFrame(self.minute_data_list)
        self.minute_data_df['datetime'] = pd.to_datetime(self.minute_data_df['datetime'])
        self.minute_data_df.set_index('datetime', inplace=True)
        
        self.analyzer = BreakAnalyzer(config_path=None) # Uses default config from BreakAnalyzer
        # Manually set data to avoid file IO for testing
        self.analyzer.minute_data = self.minute_data_df
        self.analyzer.daily_data = self.daily_data_df
        self.analyzer.prev_day_levels = prepare_previous_day_levels(self.daily_data_df)

        self.default_config = self.analyzer.config # Store for reference

    def test_find_level_break_high_break(self):
        day_data = get_day_minute_data(self.analyzer.minute_data, date(2023, 1, 2))
        prev_high = self.analyzer.prev_day_levels[date(2023, 1, 2)]['high'] # Should be 120
        confirmation_threshold = self.default_config['break_detection']['confirmation_threshold']

        break_info = self.analyzer.find_level_break(day_data, prev_high, 'high', confirmation_threshold)
        self.assertIsNotNone(break_info)
        self.assertEqual(break_info['break_type'], 'high')
        self.assertEqual(break_info['level'], 120)
        self.assertEqual(break_info['datetime'], pd.Timestamp('2023-01-02 08:02:00'))
        self.assertAlmostEqual(break_info['break_price'], 120.2)

    def test_find_level_break_low_break(self):
        day_data = get_day_minute_data(self.analyzer.minute_data, date(2023, 1, 2))
        prev_low = self.analyzer.prev_day_levels[date(2023, 1, 2)]['low'] # Should be 80
        confirmation_threshold = self.default_config['break_detection']['confirmation_threshold']

        break_info = self.analyzer.find_level_break(day_data, prev_low, 'low', confirmation_threshold)
        self.assertIsNotNone(break_info)
        self.assertEqual(break_info['break_type'], 'low')
        self.assertEqual(break_info['level'], 80)
        self.assertEqual(break_info['datetime'], pd.Timestamp('2023-01-02 09:02:00'))
        self.assertAlmostEqual(break_info['break_price'], 79.8)

    def test_find_level_break_no_break(self):
        day_data = get_day_minute_data(self.analyzer.minute_data, date(2023, 1, 2))
        # Test with a level that won't be broken
        no_break_level = 150
        confirmation_threshold = self.default_config['break_detection']['confirmation_threshold']
        break_info = self.analyzer.find_level_break(day_data, no_break_level, 'high', confirmation_threshold)
        self.assertIsNone(break_info)

    def test_analyze_post_break_continuation_then_reversal(self):
        # Using the high break from setUp
        break_data = {
            'datetime': pd.Timestamp('2023-01-02 08:02:00'),
            'break_type': 'high',
            'level': 120,
            'break_price': 120.2 # from previous test
        }
        analysis_config = self.default_config['analysis']
        result = self.analyzer.analyze_post_break(
            self.analyzer.minute_data,
            break_data,
            analysis_config['continuation_threshold'],
            analysis_config['reversal_threshold'],
            analysis_config['window_hours']
        )
        self.assertEqual(result['outcome'], 'continuation_then_reversal')
        self.assertEqual(result['first_to_occur'], 'continuation')
        # Max continuation: (121.5 - 120.2) / 120.2 * 100 = 1.3 / 120.2 * 100 = 1.0815%
        self.assertGreaterEqual(result['max_continuation_pct'], analysis_config['continuation_threshold'])
        # Max reversal: (120.2 - 119.5) / 120.2 * 100 = 0.7 / 120.2 * 100 = 0.5823%
        self.assertGreaterEqual(result['max_reversal_pct'], analysis_config['reversal_threshold'])

    def test_analyze_post_break_continuation_only(self):
         # Using the low break from setUp, assume it only continues
        break_data = {
            'datetime': pd.Timestamp('2023-01-02 09:02:00'),
            'break_type': 'low',
            'level': 80,
            'break_price': 79.8
        }
        analysis_config = self.default_config['analysis']
        
        # Create a small window of data that only shows continuation
        mock_minute_data_continuation_only = pd.DataFrame([
             # Break candle (already happened)
            {'datetime': '2023-01-02 09:02:00', 'open': 80.5, 'high': 80.6, 'low': 79.5, 'close': 79.8, 'volume': 10},
            # Post-break: Continuation (1% of 79.8 = 0.798; target = 79.002)
            {'datetime': '2023-01-02 09:03:00', 'open': 79.8, 'high': 79.9, 'low': 78.9, 'close': 79.0, 'volume': 10}, # Hits continuation
            # More continuation, no reversal
            {'datetime': '2023-01-02 09:04:00', 'open': 79.0, 'high': 79.1, 'low': 78.5, 'close': 78.6, 'volume': 10},
        ])
        mock_minute_data_continuation_only['datetime'] = pd.to_datetime(mock_minute_data_continuation_only['datetime'])
        mock_minute_data_continuation_only.set_index('datetime', inplace=True)

        result = self.analyzer.analyze_post_break(
            mock_minute_data_continuation_only, # Use this restricted data
            break_data,
            analysis_config['continuation_threshold'], # 1.0%
            analysis_config['reversal_threshold'],   # 0.5%
            analysis_config['window_hours']
        )
        # Max continuation: (79.8 - 78.5) / 79.8 * 100 = 1.3 / 79.8 * 100 = 1.629%
        self.assertGreaterEqual(result['max_continuation_pct'], analysis_config['continuation_threshold'])
        # Max reversal: (79.9 - 79.8) / 79.8 * 100 = 0.1 / 79.8 * 100 = 0.125% (assuming high of next bar is highest point for reversal calc)
        self.assertLess(result['max_reversal_pct'], analysis_config['reversal_threshold'])
        self.assertEqual(result['outcome'], 'continuation')
        self.assertEqual(result['first_to_occur'], 'continuation')


    def test_analyze_key_level_breaks_basic_run(self):
        # This test uses the data set up in setUp which has one high and one low break on 2023-01-02
        results_df = self.analyzer.analyze_key_level_breaks(start_date='2023-01-02', end_date='2023-01-02')
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), 2) # Expect one high break and one low break
        self.assertTrue('outcome' in results_df.columns)
        self.assertTrue('break_type' in results_df.columns)
        self.assertTrue('day' in results_df.columns)
        
        high_break_result = results_df[results_df['break_type'] == 'high'].iloc[0]
        self.assertEqual(high_break_result['outcome'], 'continuation_then_reversal')

        low_break_result = results_df[results_df['break_type'] == 'low'].iloc[0]
        # Based on the limited data for low break, it should be continuation
        # (79.8 - 78.9)/79.8 * 100 = 1.12% continuation.
        # (79.9 - 79.8)/79.8 * 100 = 0.125% reversal.
        # So it should be 'continuation'
        self.assertEqual(low_break_result['outcome'], 'continuation')


if __name__ == '__main__':
    unittest.main()