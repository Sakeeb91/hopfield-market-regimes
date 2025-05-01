# Hopfield Network for Market Regime Analysis

## Project Overview

This project implements a system that uses Hopfield networks to classify market regimes based on patterns in financial data. The core concept treats market regimes as stable patterns (attractors) in the Hopfield network's state space, enabling content-addressable memory to recognize current market conditions.

## Architecture

The project follows a modular architecture with the following components:

```
├── data/               # Data storage
├── src/                # Source code
│   ├── data/           # Data acquisition and processing
│   ├── models/         # Hopfield network implementation
│   ├── features/       # Feature engineering
│   └── visualization/  # Visualization utilities
├── notebooks/          # Jupyter notebooks for exploration
├── tests/              # Unit tests
└── results/            # Results and plots
    └── plots/          # Visualizations
```

### Core Components

1. **Hopfield Network**: Implementation of a Hopfield network with binary state neurons. Features include:
   - Asynchronous and synchronous update rules
   - Energy function calculation
   - Pattern storage with Hebbian learning
   - Pattern recognition through energy minimization

2. **Feature Engineering**: Technical indicators and pattern encoding:
   - Calculation of standard financial indicators (RSI, MACD, volatility, etc.)
   - Binary encoding of indicators into pattern vectors
   - Market regime definition based on indicator patterns

3. **Market Regime Classifier**: Ties everything together:
   - Prepares data by calculating indicators and encoding patterns
   - Trains the Hopfield network on defined market regimes
   - Classifies market data into regimes based on pattern recognition
   - Visualizes results in various formats

4. **Utilities**: Supporting functionality:
   - Data acquisition from Yahoo Finance
   - Data cleaning and preprocessing
   - Visualization of patterns, regimes, and statistics

## Theoretical Background

### Hopfield Networks

Hopfield networks are a form of recurrent artificial neural network that can serve as content-addressable memory systems. They consist of:

- A fully connected network of neurons (no self-connections)
- Binary or bipolar state neurons (-1, 1)
- Symmetric weights between neurons (w_ij = w_ji)
- An energy function that decreases as the network evolves

The network works by:
1. Storing patterns using Hebbian learning (correlation between neurons)
2. Processing a probe pattern by letting the network dynamics evolve
3. Converging to a stable state that represents the closest stored pattern

### Market Regimes

Market regimes are distinct states that financial markets can exhibit, such as:
- Bullish trend (upward price movement with low volatility)
- Bearish trend (downward price movement with low volatility)
- High volatility breakout (upward movement with high volatility)
- High volatility breakdown (downward movement with high volatility)
- Low volatility consolidation (sideways movement with low volatility)

These regimes are encoded as binary patterns based on the states of various technical indicators.

## Implementation Details

### Pattern Encoding

Technical indicators are calculated from price data and then encoded into binary patterns:
1. An indicator value above its median (or specified threshold) is encoded as 1
2. A value below is encoded as -1

This creates a binary vector that represents the current market state.

### Training

1. Define canonical market regime patterns (e.g., what a "Bullish Trend" looks like in terms of indicators)
2. Train the Hopfield network by storing these patterns with Hebbian learning
3. The weight matrix encodes the relationships between different indicators in each regime

### Classification

1. Encode the current market state as a binary pattern
2. Use this pattern as a probe for the Hopfield network
3. Let the network dynamics evolve until convergence
4. The final state corresponds to the closest stored market regime

## Results and Validation

The system produces:
- Regime classifications over time
- Transition probabilities between regimes
- Statistics on returns within each regime
- Visualizations of regimes overlaid on price charts

Validation is performed through:
- Unit tests for core components
- Synthetic data tests to verify regime identification
- Qualitative assessment of regime alignment with chart patterns
- Statistical analysis of regime characteristics

## Usage Example

```python
from src.data.data_acquisition import DataAcquisition
from src.models.market_regime_classifier import MarketRegimeClassifier

# Fetch data
data_acq = DataAcquisition()
data = data_acq.fetch_data_yf(ticker='SPY', period='2y')

# Initialize and train classifier
classifier = MarketRegimeClassifier()
features_df, encoded_df, patterns, time_indices = classifier.prepare_data(data)
regime_patterns = classifier.define_regimes(encoded_df)
hopfield_net = classifier.train_network()

# Classify market regimes
classification_df = classifier.classify_time_series(patterns, time_indices)

# Visualize results
fig = classifier.visualize_regime_classification(data, classification_df)
```

## Future Enhancements

Potential improvements to the system include:
- Exploring modern variants of Hopfield networks (Dense Associative Memory, Continuous Hopfield Networks)
- Incorporating more sophisticated feature engineering
- Adding reinforcement learning to optimize regime definitions
- Developing trading strategies based on regime identification
- Creating a real-time monitoring system for regime changes 