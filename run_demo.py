#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script for the Hopfield Network Market Regime Classifier.

This script runs a comprehensive demonstration of the market regime classification system,
generating visualizations and statistics to showcase the capabilities of the system.

Usage:
    python run_demo.py [--ticker TICKER] [--period PERIOD] [--interval INTERVAL]

Example:
    python run_demo.py --ticker AAPL --period 3y --interval 1d
"""

import os
import sys
import argparse
import logging
import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import local modules
from src.data.data_acquisition import DataAcquisition
from src.models.market_regime_classifier import MarketRegimeClassifier
from src.visualization.visualize import (
    plot_price_with_regimes,
    plot_regime_transitions,
    plot_regime_distribution,
    plot_regime_returns
)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Market Regime Classification Demo')
    
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol to analyze (default: SPY)')
    parser.add_argument('--period', type=str, default='2y', help='Period to download (default: 2y)')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (default: 1d)')
    
    return parser.parse_args()


def run_demo(ticker='SPY', period='2y', interval='1d'):
    """
    Run the market regime classification demo.
    
    Args:
        ticker: Ticker symbol to analyze
        period: Period to download
        interval: Data interval
    """
    print(f"Running market regime analysis demo for {ticker} over {period}...")
    
    # Create timestamp for the run
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(f"results/demo_{ticker}_{timestamp}")
    plots_dir = results_dir / "plots"
    
    # Create directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Helper function to save plots
    def save_plot(fig, filename):
        """Save a plot to the plots directory"""
        filepath = plots_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {filepath}")
        plt.close(fig)
    
    # Initialize data acquisition
    data_acq = DataAcquisition()
    
    # Fetch data
    data = data_acq.fetch_data_yf(ticker=ticker, period=period, interval=interval)
    
    # Clean data
    data = data_acq.clean_data(data)
    
    # Set Date as index if not already
    if 'Date' in data.columns:
        data = data.set_index('Date')
    
    # Save cleaned data
    data.to_csv(results_dir / "data.csv")
    
    # Initialize market regime classifier
    classifier = MarketRegimeClassifier()
    
    # Prepare data for classification
    features_df, encoded_df, patterns, time_indices = classifier.prepare_data(data)
    
    # Define market regimes
    regime_patterns = classifier.define_regimes(encoded_df)
    
    # Train the Hopfield Network
    hopfield_net = classifier.train_network()
    
    # Visualize regime patterns
    regime_fig = classifier.visualize_regime_patterns()
    save_plot(regime_fig, "regime_patterns.png")
    
    # Visualize weight matrix
    weight_fig = classifier.visualize_weight_matrix()
    save_plot(weight_fig, "weight_matrix.png")
    
    # Classify patterns
    classification_df = classifier.classify_time_series(
        patterns=patterns,
        time_indices=time_indices
    )
    
    # Save classification results
    classification_df.to_csv(results_dir / "classification_results.csv")
    
    # Visualize classification results
    classification_fig = classifier.visualize_regime_classification(
        data=data,
        classification_df=classification_df
    )
    save_plot(classification_fig, "regime_classification.png")
    
    # Generate additional visualizations
    
    # Price with regimes
    price_regimes_fig = plot_price_with_regimes(
        data=data,
        classification_df=classification_df
    )
    save_plot(price_regimes_fig, "price_with_regimes.png")
    
    # Regime transitions
    transitions_fig = plot_regime_transitions(
        classification_df=classification_df
    )
    save_plot(transitions_fig, "regime_transitions.png")
    
    # Regime distribution
    distribution_fig = plot_regime_distribution(
        classification_df=classification_df
    )
    save_plot(distribution_fig, "regime_distribution.png")
    
    # Regime returns
    returns_fig = plot_regime_returns(
        data=data,
        classification_df=classification_df
    )
    save_plot(returns_fig, "regime_returns.png")
    
    # Calculate regime statistics
    regime_stats = classification_df.groupby('regime').agg({
        'overlap': ['mean', 'min', 'max', 'std'],
        'iterations': ['mean', 'min', 'max', 'std'],
        'converged': 'mean',
        'regime': 'count'
    })
    
    # Rename count column to frequency
    regime_stats.columns = ['overlap_mean', 'overlap_min', 'overlap_max', 'overlap_std',
                          'iterations_mean', 'iterations_min', 'iterations_max', 'iterations_std',
                          'convergence_rate', 'frequency']
    
    # Save regime statistics
    regime_stats.to_csv(results_dir / "regime_statistics.csv")
    
    print("\nAnalysis completed successfully!")
    print(f"\nResults saved to: {results_dir}")
    print("\nRegime Statistics:")
    print(regime_stats[['frequency', 'convergence_rate']])
    
    print("\nTo visualize the results, check the plots directory in the results folder.")
    
    return {
        "data": data,
        "classification_df": classification_df,
        "regime_stats": regime_stats,
        "results_dir": results_dir,
        "plots_dir": plots_dir
    }


if __name__ == "__main__":
    args = parse_args()
    run_demo(ticker=args.ticker, period=args.period, interval=args.interval) 