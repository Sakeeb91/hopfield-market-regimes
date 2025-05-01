#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run all tests for the Hopfield Network Market Regime Classifier.

Usage:
    python run_tests.py
"""

import unittest
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Discover and run all tests
if __name__ == "__main__":
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test_*.py")
    test_runner = unittest.TextTestRunner(verbosity=2)
    results = test_runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(not results.wasSuccessful()) 