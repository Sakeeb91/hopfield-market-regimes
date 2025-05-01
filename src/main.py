import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from src.data.data_acquisition import DataAcquisition
from src.features.feature_engineering import FeatureEngineering
from src.models.hopfield_network import HopfieldNetwork
from src.models.market_regime_classifier import MarketRegimeClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Market Regime Classification with Hopfield Networks')
    
    # Data acquisition arguments
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol to analyze')
    parser.add_argument('--start-date', type=str, help='Start date (format: YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (format: YYYY-MM-DD)')
    parser.add_argument('--period', type=str, default='2y', help='Period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (1d, 5d, 1wk, 1mo, 3mo)')
    
    # Feature engineering arguments
    parser.add_argument('--lookback-window', type=int, default=1, help='Number of periods to include in pattern')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to store results')
    parser.add_argument('--save-model', action='store_true', help='Save the trained model')
    parser.add_argument('--load-model', type=str, help='Path to load a trained model')
    
    # Processing arguments
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode with a smaller dataset')
    
    return parser.parse_args()


def setup_output_directory(args):
    """Setup output directory for results"""
    # Create timestamp for the run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create run-specific directory
    run_dir = os.path.join(args.output_dir, f"{args.ticker}_{timestamp}")
    plots_dir = os.path.join(run_dir, 'plots')
    
    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    logger.info(f"Results will be saved to {run_dir}")
    
    return run_dir, plots_dir


def save_plot(fig, filename, plots_dir, dpi=300):
    """Save a plot to file"""
    filepath = os.path.join(plots_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    logger.info(f"Plot saved to {filepath}")
    plt.close(fig)


def create_plot_description(filename, title, description):
    """Create a description for a plot"""
    return {
        'filename': filename,
        'title': title,
        'description': description
    }


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()
    
    # Setup output directory
    run_dir, plots_dir = setup_output_directory(args)
    
    # Save run configuration
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Initialize data acquisition
    data_acq = DataAcquisition(data_dir='data')
    
    # Fetch data
    if args.test_mode:
        logger.info("Running in test mode with a smaller dataset")
        if args.start_date is None:
            args.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if args.end_date is None:
            args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Fetching data for {args.ticker}")
    if args.start_date and args.end_date:
        data = data_acq.fetch_data_yf(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            save_data=True
        )
    else:
        data = data_acq.fetch_data_yf(
            ticker=args.ticker,
            period=args.period,
            interval=args.interval,
            save_data=True
        )
    
    # Check if data was retrieved successfully
    if data.empty:
        logger.error(f"Failed to retrieve data for {args.ticker}")
        return
    
    # Save raw data
    data.to_csv(os.path.join(run_dir, 'raw_data.csv'), index=False)
    
    # Clean data
    data = data_acq.clean_data(data)
    
    # Set Date as index if not already
    if 'Date' in data.columns:
        data = data.set_index('Date')
    
    # Save cleaned data
    data.to_csv(os.path.join(run_dir, 'cleaned_data.csv'))
    
    # Initialize market regime classifier
    classifier = MarketRegimeClassifier()
    
    # If loading a pre-trained model
    if args.load_model:
        logger.info(f"Loading model from {args.load_model}")
        classifier.load_model(args.load_model)
        
        # Prepare data for classification
        features_df, encoded_df, patterns, time_indices = classifier.prepare_data(
            data=data,
            lookback_window=args.lookback_window
        )
    else:
        # Prepare data for training and classification
        features_df, encoded_df, patterns, time_indices = classifier.prepare_data(
            data=data,
            lookback_window=args.lookback_window
        )
        
        # Define market regimes
        regime_patterns = classifier.define_regimes(encoded_df)
        
        # Train the Hopfield Network
        hopfield_net = classifier.train_network()
        
        # Save model if requested
        if args.save_model:
            model_path = os.path.join(run_dir, 'hopfield_market_classifier.pkl')
            classifier.save_model(model_path)
    
    # Visualize regime patterns
    regime_fig = classifier.visualize_regime_patterns()
    save_plot(regime_fig, 'regime_patterns.png', plots_dir)
    
    # Create plot description
    plot_descriptions = []
    plot_descriptions.append(create_plot_description(
        'regime_patterns.png',
        'Market Regime Patterns',
        'Visual representation of the defined market regime patterns. Each bar represents a feature value (-1 or 1) for the respective regime.'
    ))
    
    # Visualize weight matrix
    weight_fig = classifier.visualize_weight_matrix()
    save_plot(weight_fig, 'weight_matrix.png', plots_dir)
    plot_descriptions.append(create_plot_description(
        'weight_matrix.png',
        'Hopfield Network Weight Matrix',
        'Visualization of the Hopfield Network weight matrix. The weights represent the strength of connections between neurons.'
    ))
    
    # Visualize encoded features
    encoded_fig = classifier.feature_eng.visualize_encoded_features(encoded_df)
    save_plot(encoded_fig, 'encoded_features.png', plots_dir)
    plot_descriptions.append(create_plot_description(
        'encoded_features.png',
        'Binary Encoded Features',
        'Time series of binary encoded features. Each feature is encoded as either -1 or 1 based on quantile thresholds.'
    ))
    
    # Classify patterns
    classification_df = classifier.classify_time_series(
        patterns=patterns,
        time_indices=time_indices,
        max_iterations=100,
        async_updates=True,
        convergence_threshold=3
    )
    
    # Save classification results
    classification_df.to_csv(os.path.join(run_dir, 'classification_results.csv'))
    
    # Visualize classification results
    classification_fig = classifier.visualize_regime_classification(
        data=data,
        classification_df=classification_df
    )
    save_plot(classification_fig, 'regime_classification.png', plots_dir)
    plot_descriptions.append(create_plot_description(
        'regime_classification.png',
        'Market Regime Classification',
        'Visualization of the market regime classification results. The top plot shows the price data with regime overlays, and the bottom plot shows the classified regimes over time.'
    ))
    
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
    regime_stats.to_csv(os.path.join(run_dir, 'regime_statistics.csv'))
    
    # Create a README for the results
    readme_content = f"""# Market Regime Classification Results

## Overview
- **Ticker**: {args.ticker}
- **Period**: {args.period if not (args.start_date and args.end_date) else f"{args.start_date} to {args.end_date}"}
- **Interval**: {args.interval}
- **Lookback Window**: {args.lookback_window}
- **Run Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Results Summary
The analysis classified the market data into the following regimes:

{regime_stats[['frequency', 'convergence_rate']].to_string()}

## Plots
The following plots were generated during the analysis:

"""
    
    # Add plot descriptions to README
    for plot in plot_descriptions:
        readme_content += f"""### {plot['title']}
![{plot['title']}](plots/{plot['filename']})

{plot['description']}

"""
    
    # Write README file
    with open(os.path.join(run_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Analysis completed. Results saved to {run_dir}")


if __name__ == "__main__":
    main() 