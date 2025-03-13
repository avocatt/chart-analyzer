#!/usr/bin/env python
"""
Test runner for Bitcoin Mean Reversion Statistical Analyzer.

This script discovers and runs all tests in the test directory.
To run tests, execute this script from the project root directory.
"""

import unittest
import sys
import os

if __name__ == '__main__':
    # Add the src directory to the path so tests can import modules
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests')
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return non-zero exit code if any tests failed
    sys.exit(not result.wasSuccessful()) 