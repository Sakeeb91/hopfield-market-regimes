# Notebooks

This directory contains Jupyter notebooks demonstrating the Hopfield Network for Market Regime Analysis.

## Available Notebooks

### hopfield_market_regime_demo.ipynb

A comprehensive demonstration of using Hopfield networks to classify market regimes based on financial data patterns. This notebook covers:

1. Data Acquisition: Fetching historical price data from Yahoo Finance.
2. Feature Engineering: Calculating technical indicators and encoding them into binary patterns.
3. Hopfield Network Training: Defining market regimes and training the Hopfield network.
4. Market Regime Classification: Classifying market data into predefined regimes.
5. Analysis & Visualization: Analyzing and visualizing the classification results.

## Running the Notebooks

To run these notebooks, make sure you have the required dependencies installed:

```bash
pip install -r ../requirements.txt
```

Then start Jupyter:

```bash
jupyter notebook
```

## Generating New Results

You can modify the following parameters in the notebooks to generate different results:

- `ticker`: Stock symbol to analyze (e.g., 'SPY', 'AAPL', 'MSFT')
- `period`: Time period to analyze (e.g., '1y', '2y', '5y')
- `interval`: Data interval (e.g., '1d', '1wk')
- `lookback_window`: Number of periods to include in each pattern

You can also modify the regime definitions in `src/features/feature_engineering.py` to create your own market regime classifications. 