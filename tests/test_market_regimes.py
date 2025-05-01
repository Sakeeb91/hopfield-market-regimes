import unittest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.feature_engineering import FeatureEngineering
from src.models.market_regime_classifier import MarketRegimeClassifier


class TestMarketRegimes(unittest.TestCase):
    """
    Tests for market regime definition and classification.
    This test uses synthetic data to test the market regime classifier.
    """
    
    def setUp(self):
        """Set up synthetic market data for testing."""
        # Create a date range for synthetic data
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Initialize empty DataFrame
        self.market_data = pd.DataFrame(index=dates)
        
        # Generate synthetic OHLCV data
        n = len(dates)
        seed = 42
        np.random.seed(seed)
        
        # Start with a base price
        base_price = 100
        
        # Generate price series with different segments (regimes)
        price_series = []
        
        # Regime 1: Bullish trend (first 50 days)
        days_1 = 50
        trend_1 = np.linspace(0, 15, days_1)  # Linear uptrend
        noise_1 = np.random.normal(0, 1, days_1)  # Low volatility
        regime_1 = base_price + trend_1 + noise_1
        price_series.extend(regime_1)
        
        # Regime 2: High volatility breakdown (next 40 days)
        days_2 = 40
        trend_2 = np.linspace(0, -20, days_2)  # Downtrend
        noise_2 = np.random.normal(0, 3, days_2)  # High volatility
        regime_2 = regime_1[-1] + trend_2 + noise_2
        price_series.extend(regime_2)
        
        # Regime 3: Low volatility consolidation (next 60 days)
        days_3 = 60
        trend_3 = np.zeros(days_3)  # Flat
        noise_3 = np.random.normal(0, 0.8, days_3)  # Very low volatility
        regime_3 = regime_2[-1] + trend_3 + noise_3
        price_series.extend(regime_3)
        
        # Regime 4: Bullish trend again (remaining days)
        days_4 = n - days_1 - days_2 - days_3
        trend_4 = np.linspace(0, 25, days_4)  # Stronger uptrend
        noise_4 = np.random.normal(0, 1.2, days_4)  # Low volatility
        regime_4 = regime_3[-1] + trend_4 + noise_4
        price_series.extend(regime_4)
        
        # Generate OHLCV data based on the price series
        self.market_data['Close'] = price_series
        
        # Generate synthetic Open, High, Low based on Close
        for i in range(n):
            daily_volatility = 1.5 if days_1 <= i < days_1 + days_2 else 0.8
            self.market_data.loc[self.market_data.index[i], 'Open'] = self.market_data['Close'].iloc[i] * (1 + np.random.normal(0, 0.005))
            self.market_data.loc[self.market_data.index[i], 'High'] = self.market_data['Close'].iloc[i] * (1 + abs(np.random.normal(0, daily_volatility * 0.008)))
            self.market_data.loc[self.market_data.index[i], 'Low'] = self.market_data['Close'].iloc[i] * (1 - abs(np.random.normal(0, daily_volatility * 0.008)))
            
        # Ensure High is always the highest and Low is always the lowest
        self.market_data['High'] = self.market_data[['Open', 'Close', 'High']].max(axis=1)
        self.market_data['Low'] = self.market_data[['Open', 'Close', 'Low']].min(axis=1)
        
        # Generate volume
        vol_base = 1000000
        vol_trend = np.concatenate([
            np.random.normal(1.0, 0.1, days_1),  # Normal volume in regime 1
            np.random.normal(1.5, 0.3, days_2),  # Higher volume in regime 2 (volatile)
            np.random.normal(0.7, 0.1, days_3),  # Lower volume in regime 3 (consolidation)
            np.random.normal(1.2, 0.2, days_4)   # Higher volume in regime 4 (bullish)
        ])
        self.market_data['Volume'] = vol_base * vol_trend
        
        # Store regime boundaries for validation
        self.regime_boundaries = {
            'Bullish Trend': (0, days_1),
            'High Volatility Breakdown': (days_1, days_1 + days_2),
            'Low Volatility Consolidation': (days_1 + days_2, days_1 + days_2 + days_3),
            'Bullish Trend': (days_1 + days_2 + days_3, n)
        }
        
        # Create feature engineering and classifier instances
        self.feature_eng = FeatureEngineering()
        self.classifier = MarketRegimeClassifier()
    
    def test_feature_engineering(self):
        """Test that feature engineering generates the expected indicators."""
        # Calculate all indicators
        indicators_df = self.feature_eng.calculate_indicators(self.market_data)
        
        # Check that key indicators were calculated
        key_indicators = [
            'return_1', 'return_5', 'volatility_5', 'volatility_20', 
            'atr_14', 'macd_line', 'macd_histogram', 'rsi_14'
        ]
        
        for indicator in key_indicators:
            self.assertIn(indicator, indicators_df.columns, f"Indicator {indicator} not found")
        
        # Check that indicators have expected properties
        # RSI should be between 0 and 100
        self.assertTrue((indicators_df['rsi_14'] >= 0).all() and (indicators_df['rsi_14'] <= 100).all())
        
        # Volatility should be non-negative
        self.assertTrue((indicators_df['volatility_5'] >= 0).all())
        self.assertTrue((indicators_df['volatility_20'] >= 0).all())
        
        # ATR should be non-negative
        self.assertTrue((indicators_df['atr_14'] >= 0).all())
    
    def test_binary_encoding(self):
        """Test that binary encoding works correctly."""
        # Create a simple series
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Encode using threshold method
        binary_threshold = self.feature_eng.encode_feature_binary(
            series, method='threshold', threshold=5
        )
        self.assertTrue(np.array_equal(binary_threshold, np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])))
        
        # Encode using quantile method
        binary_quantile = self.feature_eng.encode_feature_binary(
            series, method='quantile'
        )
        # The median (5.5) should be the threshold
        self.assertTrue(np.array_equal(binary_quantile, np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])))
    
    def test_market_regime_definition(self):
        """Test market regime definition."""
        # Calculate indicators
        indicators_df = self.feature_eng.calculate_indicators(self.market_data)
        
        # Define regime features
        features_df = self.feature_eng.define_regime_features(indicators_df)
        
        # Encode features
        encoded_df = self.feature_eng.encode_features(features_df)
        
        # Define regimes
        regime_patterns = self.classifier.define_regimes(encoded_df)
        
        # Check that regimes were defined
        self.assertGreater(len(regime_patterns), 0)
        self.assertGreater(len(self.classifier.regime_patterns), 0)
        self.assertGreater(len(self.classifier.regime_labels), 0)
        
        # Check that each regime label corresponds to a pattern
        self.assertEqual(len(self.classifier.regime_patterns), len(self.classifier.regime_labels))
    
    def test_market_regime_classification(self):
        """Test market regime classification on synthetic data."""
        # Prepare data
        features_df, encoded_df, patterns, time_indices = self.classifier.prepare_data(self.market_data)
        
        # Define regimes
        regime_patterns = self.classifier.define_regimes(encoded_df)
        
        # Train network
        hopfield_net = self.classifier.train_network()
        
        # Classify patterns
        classification_df = self.classifier.classify_time_series(
            patterns=patterns,
            time_indices=time_indices,
            max_iterations=100,
            async_updates=True,
            convergence_threshold=3
        )
        
        # Check that classifications were generated
        self.assertGreater(len(classification_df), 0)
        self.assertIn('regime', classification_df.columns)
        
        # Validate regime statistics
        regime_stats = classification_df.groupby('regime').size()
        
        # Check that all defined regimes appear in the classification
        for regime in self.classifier.regime_labels:
            self.assertIn(regime, classification_df['regime'].values)
        
        # Simple check for pattern in regimes over time
        # This is a rough check since the boundaries aren't going to be exact
        # Given how technical indicators lag and the probabilistic nature of the Hopfield network
        
        # Function to get the dominant regime in a time slice
        def get_dominant_regime(start_idx, end_idx):
            slice_df = classification_df.iloc[start_idx:end_idx]
            return slice_df['regime'].mode()[0]
        
        # Check if the dominant regimes near the synthetic boundaries match expectations
        # Allow for some buffer since indicators lag
        buffer = 20  # Business days
        
        # We expect a bullish trend at the beginning
        early_dominant = get_dominant_regime(buffer, self.regime_boundaries['Bullish Trend'][1] - buffer)
        self.assertIn('Bullish', early_dominant, f"Expected 'Bullish' in early regime, got {early_dominant}")
        
        # We expect high volatility breakdown in the second regime
        middle_start = self.regime_boundaries['High Volatility Breakdown'][0] + buffer
        middle_end = self.regime_boundaries['High Volatility Breakdown'][1] - buffer
        middle_dominant = get_dominant_regime(middle_start, middle_end)
        self.assertIn('Volatility', middle_dominant, f"Expected 'Volatility' in middle regime, got {middle_dominant}")
        
        # We expect low volatility consolidation in the third regime
        late_start = self.regime_boundaries['Low Volatility Consolidation'][0] + buffer
        late_end = self.regime_boundaries['Low Volatility Consolidation'][1] - buffer
        late_dominant = get_dominant_regime(late_start, late_end)
        self.assertIn('Volatility', late_dominant, f"Expected 'Volatility' in late regime, got {late_dominant}")
        
    def test_save_load_model(self):
        """Test saving and loading the model."""
        import tempfile
        
        # Prepare data and train model
        features_df, encoded_df, patterns, time_indices = self.classifier.prepare_data(self.market_data)
        regime_patterns = self.classifier.define_regimes(encoded_df)
        hopfield_net = self.classifier.train_network()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp:
            model_path = temp.name
        
        try:
            # Save the model
            self.classifier.save_model(model_path)
            
            # Create a new classifier
            new_classifier = MarketRegimeClassifier()
            
            # Load the model
            new_classifier.load_model(model_path)
            
            # Check that the model was loaded correctly
            self.assertEqual(len(new_classifier.regime_patterns), len(self.classifier.regime_patterns))
            self.assertEqual(len(new_classifier.regime_labels), len(self.classifier.regime_labels))
            
            # Test classification with loaded model
            _, new_encoded_df, new_patterns, new_time_indices = new_classifier.prepare_data(self.market_data)
            
            new_classification_df = new_classifier.classify_time_series(
                patterns=new_patterns,
                time_indices=new_time_indices
            )
            
            self.assertGreater(len(new_classification_df), 0)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)


if __name__ == '__main__':
    unittest.main() 