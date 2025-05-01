import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import logging
import os
from datetime import datetime

# Import local modules
from src.models.hopfield_network import HopfieldNetwork
from src.features.feature_engineering import FeatureEngineering

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketRegimeClassifier:
    """
    Class for classifying market regimes using a Hopfield Network.
    """
    
    def __init__(self):
        """
        Initialize the market regime classifier.
        """
        self.feature_eng = FeatureEngineering()
        self.hopfield_network = None
        self.regime_patterns = []
        self.regime_labels = []
        self.feature_names = []
    
    def prepare_data(self, 
                    data: pd.DataFrame, 
                    feature_names: List[str] = None,
                    lookback_window: int = 1,
                    encoding_methods: Dict[str, Dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[np.ndarray], List]:
        """
        Prepare data for regime classification.
        
        Args:
            data: DataFrame with OHLCV data
            feature_names: List of features to use for classification
            lookback_window: Number of periods to include in pattern
            encoding_methods: Dictionary of encoding methods for features
            
        Returns:
            Tuple of (features_df, encoded_df, patterns, time_indices)
        """
        # Calculate all technical indicators
        indicators_df = self.feature_eng.calculate_indicators(data)
        
        # Select features for regime identification
        features_df = self.feature_eng.define_regime_features(
            indicators_df, 
            feature_names=feature_names
        )
        
        # Store feature names for later use
        self.feature_names = features_df.columns.tolist()
        
        # Encode features into binary patterns
        encoded_df = self.feature_eng.encode_features(
            features_df, 
            encoding_methods=encoding_methods
        )
        
        # Create pattern vectors
        patterns, time_indices = self.feature_eng.create_pattern_vectors(
            encoded_df,
            window_size=lookback_window
        )
        
        return features_df, encoded_df, patterns, time_indices
    
    def define_regimes(self, 
                      encoded_df: pd.DataFrame,
                      market_regimes: List[Dict] = None) -> List[Tuple[str, np.ndarray]]:
        """
        Define market regime patterns.
        
        Args:
            encoded_df: DataFrame with encoded features
            market_regimes: List of market regime definitions
            
        Returns:
            List of regime patterns
        """
        # Define market regimes
        regime_patterns = self.feature_eng.define_market_regimes(
            market_regimes=market_regimes,
            feature_names=self.feature_names
        )
        
        # Store regime patterns and labels
        self.regime_patterns = [pattern for _, pattern in regime_patterns]
        self.regime_labels = [name for name, _ in regime_patterns]
        
        return regime_patterns
    
    def train_network(self, num_neurons: Optional[int] = None) -> HopfieldNetwork:
        """
        Train the Hopfield Network with the defined market regimes.
        
        Args:
            num_neurons: Number of neurons in the network (if None, will be inferred from patterns)
            
        Returns:
            Trained Hopfield Network
        """
        if not self.regime_patterns:
            raise ValueError("No regime patterns defined. Call define_regimes() first.")
        
        # Determine network size if not specified
        if num_neurons is None:
            num_neurons = len(self.regime_patterns[0])
            
        # Initialize Hopfield Network
        self.hopfield_network = HopfieldNetwork(num_neurons)
        
        # Store each regime pattern in the network
        for i, pattern in enumerate(self.regime_patterns):
            label = self.regime_labels[i] if i < len(self.regime_labels) else f"Regime {i}"
            self.hopfield_network.store_pattern(pattern, label=label)
            
        logger.info(f"Trained Hopfield Network with {len(self.regime_patterns)} regime patterns")
        
        return self.hopfield_network
    
    def classify_pattern(self, 
                       pattern: np.ndarray, 
                       max_iterations: int = 100, 
                       async_updates: bool = True,
                       convergence_threshold: int = 3) -> Dict:
        """
        Classify a pattern into one of the defined market regimes.
        
        Args:
            pattern: Binary pattern to classify
            max_iterations: Maximum iterations for the Hopfield Network
            async_updates: Whether to use asynchronous updates
            convergence_threshold: Number of iterations without change to consider converged
            
        Returns:
            Classification result
        """
        if self.hopfield_network is None:
            raise ValueError("Hopfield Network not trained. Call train_network() first.")
            
        # Classify the pattern
        result = self.hopfield_network.classify_pattern(
            pattern,
            max_iterations=max_iterations,
            async_updates=async_updates,
            convergence_threshold=convergence_threshold
        )
        
        return result
    
    def classify_time_series(self, 
                          patterns: List[np.ndarray], 
                          time_indices: List,
                          **kwargs) -> pd.DataFrame:
        """
        Classify a time series of patterns.
        
        Args:
            patterns: List of patterns to classify
            time_indices: List of corresponding time indices
            **kwargs: Additional arguments for classify_pattern
            
        Returns:
            DataFrame with classification results
        """
        if self.hopfield_network is None:
            raise ValueError("Hopfield Network not trained. Call train_network() first.")
            
        results = []
        
        for i, pattern in enumerate(patterns):
            result = self.classify_pattern(pattern, **kwargs)
            
            # Create a row with classification results
            row = {
                'time_index': time_indices[i],
                'regime': result['label'],
                'overlap': result['overlap'],
                'iterations': result['iterations'],
                'energy': result['final_energy'],
                'converged': result['iterations'] < kwargs.get('max_iterations', 100)
            }
            
            results.append(row)
            
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Set time_index as index if they are datetime
        if isinstance(results_df['time_index'].iloc[0], (pd.Timestamp, datetime)):
            results_df.set_index('time_index', inplace=True)
            
        logger.info(f"Classified {len(patterns)} patterns")
        
        return results_df
    
    def visualize_weight_matrix(self, figsize=(10, 8)):
        """
        Visualize the Hopfield Network weight matrix.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.hopfield_network is None:
            raise ValueError("Hopfield Network not trained. Call train_network() first.")
            
        return self.hopfield_network.visualize_weights(figsize=figsize)
    
    def visualize_regime_patterns(self, figsize=(12, 8)):
        """
        Visualize the defined market regime patterns.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.regime_patterns or not self.regime_labels or not self.feature_names:
            raise ValueError("No regime patterns defined. Call define_regimes() first.")
            
        # Create list of (name, pattern) tuples
        regime_patterns = list(zip(self.regime_labels, self.regime_patterns))
        
        # Extract feature names from encoded column names
        display_feature_names = [name.replace('_encoded', '') for name in self.feature_names]
            
        return self.feature_eng.visualize_regime_patterns(
            regime_patterns=regime_patterns,
            feature_names=display_feature_names,
            figsize=figsize
        )
    
    def visualize_regime_classification(self, 
                                      data: pd.DataFrame, 
                                      classification_df: pd.DataFrame,
                                      price_col: str = 'Close',
                                      figsize=(15, 10)):
        """
        Visualize the market regime classification overlaid on price data.
        
        Args:
            data: Original price data
            classification_df: DataFrame with classification results
            price_col: Column to use for price data
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Create figure with price subplot and regime subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Plot price data
        ax1.plot(data.index, data[price_col], color='blue', alpha=0.7)
        ax1.set_ylabel('Price')
        ax1.set_title('Market Regimes Classification')
        ax1.grid(True, alpha=0.3)
        
        # Create color map for regimes
        unique_regimes = classification_df['regime'].unique()
        color_map = dict(zip(unique_regimes, sns.color_palette("husl", len(unique_regimes))))
        
        # Plot regime classification
        for regime in unique_regimes:
            # Filter data for this regime
            regime_data = classification_df[classification_df['regime'] == regime]
            
            # Create spans for consecutive occurrences of the same regime
            spans = []
            current_span = None
            
            for idx, row in regime_data.iterrows():
                if current_span is None:
                    current_span = [idx, idx]
                elif (idx - current_span[1]) <= pd.Timedelta(days=1):  # Adjust this threshold as needed
                    current_span[1] = idx
                else:
                    spans.append(current_span)
                    current_span = [idx, idx]
            
            if current_span is not None:
                spans.append(current_span)
            
            # Add spans to the plot
            for span in spans:
                ax1.axvspan(span[0], span[1], alpha=0.3, color=color_map[regime])
                
            # Plot regime on the lower subplot
            ax2.scatter(regime_data.index, [regime] * len(regime_data), 
                      marker='s', s=100, color=color_map[regime], label=regime)
        
        # Configure the regime subplot
        ax2.set_yticks([])
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.01, 1))
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Regime')
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filepath: str = 'models/hopfield_market_classifier.pkl') -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'hopfield_network': self.hopfield_network,
            'regime_patterns': self.regime_patterns,
            'regime_labels': self.regime_labels,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'models/hopfield_market_classifier.pkl') -> None:
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            self.hopfield_network = model_data['hopfield_network']
            self.regime_patterns = model_data['regime_patterns']
            self.regime_labels = model_data['regime_labels']
            self.feature_names = model_data['feature_names']
            
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            raise 