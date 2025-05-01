#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the Market Regime Classifier from the command line.

Usage:
    python run_market_regime_classifier.py --ticker SPY --period 2y --interval 1d --save-model

For more options:
    python run_market_regime_classifier.py --help
"""

import sys
import os
from src.main import main

if __name__ == "__main__":
    main() 