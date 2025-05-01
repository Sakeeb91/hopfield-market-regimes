import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Class for calculating and encoding technical indicators for market regime analysis.
    """
    
    def __init__(self):
        """
        Initialize the feature engineering module.
        """
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Define quantile thresholds for binary encoding
        self.quantile_thresholds = {
            'binary': 0.5,  # Middle point for binary encoding
            'ternary': [0.33, 0.67],  # Tertiles for ternary encoding
            'quantile': [0.2, 0.4, 0.6, 0.8]  # Quintiles for 5-level encoding
        }
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the dataframe has the required columns.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            logger.error(f"DataFrame is missing required columns: {missing_cols}")
            return False
        return True
    
    def calculate_returns(self, data: pd.DataFrame, periods: List[int] = [1, 5, 20]) -> pd.DataFrame:
        """
        Calculate returns over specified periods.
        
        Args:
            data: DataFrame with price data
            periods: List of periods to calculate returns for
            
        Returns:
            DataFrame with returns columns added
        """
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        for period in periods:
            col_name = f"return_{period}"
            df[col_name] = df['Close'].pct_change(period)
            logger.info(f"Calculated {period}-period returns")
            
        return df
    
    def calculate_volatility(self, data: pd.DataFrame, windows: List[int] = [5, 20]) -> pd.DataFrame:
        """
        Calculate volatility (standard deviation of returns) over specified windows.
        
        Args:
            data: DataFrame with price data
            windows: List of window sizes to calculate volatility for
            
        Returns:
            DataFrame with volatility columns added
        """
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Calculate daily returns if not already present
        if 'return_1' not in df.columns:
            df['return_1'] = df['Close'].pct_change()
        
        for window in windows:
            col_name = f"volatility_{window}"
            df[col_name] = df['return_1'].rolling(window=window).std()
            logger.info(f"Calculated {window}-day volatility")
            
        return df
    
    def calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: DataFrame with price data
            window: Window size for ATR calculation
            
        Returns:
            DataFrame with ATR column added
        """
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Calculate True Range
        df['high_low'] = df['High'] - df['Low']
        df['high_close'] = np.abs(df['High'] - df['Close'].shift())
        df['low_close'] = np.abs(df['Low'] - df['Close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Calculate ATR
        df[f'atr_{window}'] = df['tr'].rolling(window=window).mean()
        
        # Drop intermediate columns
        df = df.drop(['high_low', 'high_close', 'low_close', 'tr'], axis=1)
        
        logger.info(f"Calculated {window}-period ATR")
        
        return df
    
    def calculate_moving_averages(self, data: pd.DataFrame, windows: List[int] = [5, 20, 50, 200]) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages (SMA) for specified windows.
        
        Args:
            data: DataFrame with price data
            windows: List of window sizes for SMA calculation
            
        Returns:
            DataFrame with SMA columns added
        """
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        for window in windows:
            col_name = f"sma_{window}"
            df[col_name] = df['Close'].rolling(window=window).mean()
            logger.info(f"Calculated {window}-period SMA")
            
        return df
    
    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            data: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal EMA period
            
        Returns:
            DataFrame with MACD columns added
        """
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Calculate MACD components
        df['ema_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
        df['ema_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
        df['macd_line'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd_line'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        # Drop intermediate EMA columns
        df = df.drop(['ema_fast', 'ema_slow'], axis=1)
        
        logger.info(f"Calculated MACD ({fast},{slow},{signal})")
        
        return df
    
    def calculate_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: DataFrame with price data
            window: Window size for RSI calculation
            
        Returns:
            DataFrame with RSI column added
        """
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Calculate daily price changes
        delta = df['Close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        logger.info(f"Calculated {window}-period RSI")
        
        return df
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with price data
            window: Window size for moving average
            num_std: Number of standard deviations for bands
            
        Returns:
            DataFrame with Bollinger Band columns added
        """
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Calculate middle band (SMA)
        df[f'bb_middle_{window}'] = df['Close'].rolling(window=window).mean()
        
        # Calculate standard deviation
        std = df['Close'].rolling(window=window).std()
        
        # Calculate upper and lower bands
        df[f'bb_upper_{window}'] = df[f'bb_middle_{window}'] + (std * num_std)
        df[f'bb_lower_{window}'] = df[f'bb_middle_{window}'] - (std * num_std)
        
        # Calculate bandwidth and %B
        df[f'bb_bandwidth_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
        df[f'bb_percent_{window}'] = (df['Close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
        
        logger.info(f"Calculated {window}-period Bollinger Bands")
        
        return df
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        if not self.validate_data(data):
            return data
        
        logger.info("Calculating all technical indicators...")
        
        # Chain all indicator calculations
        df = data.copy()
        df = self.calculate_returns(df)
        df = self.calculate_volatility(df)
        df = self.calculate_atr(df)
        df = self.calculate_moving_averages(df)
        df = self.calculate_macd(df)
        df = self.calculate_rsi(df)
        df = self.calculate_bollinger_bands(df)
        
        logger.info("Completed calculation of all indicators")
        
        return df
    
    def define_regime_features(self, 
                             data: pd.DataFrame, 
                             feature_names: List[str] = None,
                             lookback_window: int = 5) -> pd.DataFrame:
        """
        Define features to be used for regime identification.
        
        Args:
            data: DataFrame with indicators
            feature_names: List of column names to use as features
            lookback_window: Number of periods to look back for pattern recognition
            
        Returns:
            DataFrame with features selected for given lookback window
        """
        if feature_names is None:
            # Default set of features for regime identification
            feature_names = [
                'return_1', 'return_5', 
                'volatility_5', 'volatility_20', 
                'atr_14',
                'macd_line', 'macd_histogram',
                'rsi_14', 
                'bb_bandwidth_20', 'bb_percent_20'
            ]
        
        # Ensure all feature names exist in the DataFrame
        missing_features = set(feature_names) - set(data.columns)
        if missing_features:
            logger.warning(f"Some specified features are missing: {missing_features}")
            feature_names = list(set(feature_names) - missing_features)
        
        if not feature_names:
            logger.error("No valid features specified")
            return pd.DataFrame()
        
        # Create a view of the data with only the specified features
        features_df = data[feature_names].copy()
        
        # Drop NaN values (from lagged indicators)
        features_df = features_df.dropna()
        
        logger.info(f"Selected {len(feature_names)} features for regime identification")
        
        return features_df
    
    def encode_feature_binary(self, 
                           feature_series: pd.Series, 
                           method: str = 'threshold',
                           threshold: float = 0,
                           window: int = None,
                           rolling_threshold: str = 'median') -> np.ndarray:
        """
        Encode a feature series into binary pattern (-1, 1).
        
        Args:
            feature_series: Series to encode
            method: Encoding method ('threshold', 'quantile', 'moving_avg')
            threshold: Fixed threshold for 'threshold' method
            window: Window size for 'moving_avg' method
            rolling_threshold: Statistic to use as threshold for 'moving_avg' ('mean', 'median')
            
        Returns:
            Binary encoded series (-1, 1)
        """
        series = feature_series.copy()
        
        if method == 'threshold':
            # Simple threshold at specified value
            return np.where(series > threshold, 1, -1)
            
        elif method == 'quantile':
            # Use series median as threshold
            threshold = series.quantile(self.quantile_thresholds['binary'])
            return np.where(series > threshold, 1, -1)
            
        elif method == 'moving_avg':
            if window is None:
                window = min(20, len(series) // 4)  # Default window is 20 or 1/4 of series length
                
            if rolling_threshold == 'mean':
                rolling_stat = series.rolling(window=window).mean()
            elif rolling_threshold == 'median':
                rolling_stat = series.rolling(window=window).median()
            else:
                logger.warning(f"Unrecognized rolling threshold: {rolling_threshold}, using median")
                rolling_stat = series.rolling(window=window).median()
                
            # Forward fill NaNs from the initial window
            rolling_stat = rolling_stat.fillna(method='ffill')
            
            # Compare to rolling statistic
            return np.where(series > rolling_stat, 1, -1)
        
        else:
            logger.warning(f"Unrecognized encoding method: {method}, using quantile")
            threshold = series.quantile(self.quantile_thresholds['binary'])
            return np.where(series > threshold, 1, -1)
    
    def encode_features(self, 
                      features_df: pd.DataFrame, 
                      encoding_methods: Dict[str, Dict] = None) -> pd.DataFrame:
        """
        Encode multiple features into binary patterns.
        
        Args:
            features_df: DataFrame with features
            encoding_methods: Dictionary mapping feature names to encoding parameters
            
        Returns:
            DataFrame with encoded features
        """
        if encoding_methods is None:
            # Default to quantile encoding for all features
            encoding_methods = {col: {'method': 'quantile'} for col in features_df.columns}
        
        encoded_df = pd.DataFrame(index=features_df.index)
        
        for col in features_df.columns:
            # Get encoding parameters for this feature (or use default)
            params = encoding_methods.get(col, {'method': 'quantile'})
            
            # Encode the feature
            encoded_series = self.encode_feature_binary(
                features_df[col],
                method=params.get('method', 'quantile'),
                threshold=params.get('threshold', 0),
                window=params.get('window', None),
                rolling_threshold=params.get('rolling_threshold', 'median')
            )
            
            # Add to encoded DataFrame
            encoded_df[f"{col}_encoded"] = encoded_series
            
        logger.info(f"Encoded {len(features_df.columns)} features into binary patterns")
        
        return encoded_df
    
    def create_pattern_vectors(self, 
                             encoded_df: pd.DataFrame, 
                             window_size: int = 1) -> Tuple[List[np.ndarray], List[int]]:
        """
        Create pattern vectors from encoded features.
        
        Args:
            encoded_df: DataFrame with encoded features
            window_size: Number of consecutive time periods to include in each pattern
            
        Returns:
            Tuple of (list of pattern vectors, list of corresponding time indices)
        """
        if window_size < 1:
            logger.warning(f"Invalid window size: {window_size}, setting to 1")
            window_size = 1
            
        num_features = encoded_df.shape[1]
        pattern_length = num_features * window_size
        
        patterns = []
        time_indices = []
        
        # For each possible window position
        for i in range(len(encoded_df) - window_size + 1):
            # Extract window of encoded features
            window = encoded_df.iloc[i:i+window_size]
            
            # Flatten the window into a 1D pattern vector
            pattern = window.values.flatten()
            
            patterns.append(pattern)
            time_indices.append(encoded_df.index[i + window_size - 1])  # Use the last index in the window
            
        logger.info(f"Created {len(patterns)} pattern vectors of length {pattern_length}")
        
        return patterns, time_indices
    
    def define_market_regimes(self, 
                            market_regimes: List[Dict] = None,
                            feature_names: List[str] = None) -> List[Tuple[str, np.ndarray]]:
        """
        Define canonical market regime patterns.
        
        Args:
            market_regimes: List of dictionaries defining market regimes
            feature_names: List of feature names that will be encoded
            
        Returns:
            List of tuples (regime_name, pattern_vector)
        """
        if feature_names is None and hasattr(self, 'feature_names'):
            feature_names = self.feature_names
        elif feature_names is None:
            # Default set of features if none provided
            feature_names = [
                'return_1', 'return_5', 
                'volatility_5', 'volatility_20', 
                'atr_14',
                'macd_line', 'macd_histogram',
                'rsi_14', 
                'bb_bandwidth_20', 'bb_percent_20'
            ]
        
        if market_regimes is None:
            # Define common market regimes
            # Example: each regime is defined by expected states of certain indicators
            market_regimes = [
                {
                    'name': 'Bullish Trend',
                    'features': {
                        'return_1': 1,       # Positive returns
                        'return_5': 1,       # Positive medium-term returns
                        'volatility_5': -1,  # Low short-term volatility
                        'volatility_20': -1, # Low medium-term volatility
                        'macd_line': 1,      # MACD line above zero
                        'macd_histogram': 1, # MACD histogram positive (bullish momentum)
                        'rsi_14': 1,         # RSI high (but not extremely high)
                        'bb_percent_20': 1   # Price near upper band
                    }
                },
                {
                    'name': 'Bearish Trend',
                    'features': {
                        'return_1': -1,      # Negative returns
                        'return_5': -1,      # Negative medium-term returns
                        'volatility_5': -1,  # Can be low volatility
                        'volatility_20': -1, # Low medium-term volatility
                        'macd_line': -1,     # MACD line below zero
                        'macd_histogram': -1,# MACD histogram negative (bearish momentum)
                        'rsi_14': -1,        # RSI low (but not extremely low)
                        'bb_percent_20': -1  # Price near lower band
                    }
                },
                {
                    'name': 'High Volatility Breakout',
                    'features': {
                        'return_1': 1,       # Positive returns
                        'return_5': 1,       # Positive medium-term returns
                        'volatility_5': 1,   # High short-term volatility
                        'volatility_20': 1,  # High medium-term volatility
                        'atr_14': 1,         # High ATR
                        'macd_histogram': 1, # MACD histogram positive (bullish momentum)
                        'bb_bandwidth_20': 1 # Expanded Bollinger Bands
                    }
                },
                {
                    'name': 'High Volatility Breakdown',
                    'features': {
                        'return_1': -1,      # Negative returns
                        'return_5': -1,      # Negative medium-term returns
                        'volatility_5': 1,   # High short-term volatility
                        'volatility_20': 1,  # High medium-term volatility
                        'atr_14': 1,         # High ATR
                        'macd_histogram': -1,# MACD histogram negative (bearish momentum)
                        'bb_bandwidth_20': 1 # Expanded Bollinger Bands
                    }
                },
                {
                    'name': 'Low Volatility Consolidation',
                    'features': {
                        'return_1': -1,      # Small returns (using -1 for simplicity)
                        'return_5': -1,      # Small medium-term returns
                        'volatility_5': -1,  # Low short-term volatility
                        'volatility_20': -1, # Low medium-term volatility
                        'atr_14': -1,        # Low ATR
                        'macd_line': -1,     # MACD line around zero
                        'bb_bandwidth_20': -1# Contracted Bollinger Bands
                    }
                }
            ]
        
        regime_patterns = []
        
        # For each regime definition
        encoded_feature_names = [f"{feat}_encoded" for feat in feature_names]
        
        for regime in market_regimes:
            # Get the feature values for this regime
            features = regime['features']
            
            # Create pattern vector
            pattern = np.array([features.get(feat.replace('_encoded', ''), 0) for feat in encoded_feature_names])
            
            regime_patterns.append((regime['name'], pattern))
            
        logger.info(f"Defined {len(regime_patterns)} market regimes")
        
        return regime_patterns
    
    def visualize_encoded_features(self, encoded_df: pd.DataFrame, figsize=(15, 10)):
        """
        Visualize encoded features over time.
        
        Args:
            encoded_df: DataFrame with encoded features
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)
        
        # Plot each encoded feature
        for i, col in enumerate(encoded_df.columns):
            plt.subplot(len(encoded_df.columns), 1, i+1)
            plt.plot(encoded_df.index, encoded_df[col], 'o-')
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.yticks([-1, 1])
            plt.ylabel(col)
            
            if i < len(encoded_df.columns) - 1:
                plt.xticks([])
            
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_pattern_similarities(self, patterns: List[np.ndarray], figsize=(10, 8)):
        """
        Visualize the similarity matrix between patterns.
        
        Args:
            patterns: List of pattern vectors
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Calculate similarity matrix (dot product)
        n_patterns = len(patterns)
        sim_matrix = np.zeros((n_patterns, n_patterns))
        
        for i in range(n_patterns):
            for j in range(n_patterns):
                # Normalize dot product by pattern length
                sim_matrix[i, j] = np.dot(patterns[i], patterns[j]) / len(patterns[i])
        
        # Visualize the similarity matrix
        plt.figure(figsize=figsize)
        sns.heatmap(sim_matrix, cmap='coolwarm', center=0)
        plt.title('Pattern Similarity Matrix')
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_regime_patterns(self, 
                                regime_patterns: List[Tuple[str, np.ndarray]], 
                                feature_names: List[str],
                                figsize=(12, 8)):
        """
        Visualize defined market regime patterns.
        
        Args:
            regime_patterns: List of tuples (regime_name, pattern_vector)
            feature_names: Names of features in the pattern vectors
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_regimes = len(regime_patterns)
        n_features = len(feature_names)
        
        plt.figure(figsize=figsize)
        
        for i, (name, pattern) in enumerate(regime_patterns):
            plt.subplot(n_regimes, 1, i+1)
            
            # Ensure pattern length matches number of features
            if len(pattern) != n_features:
                logger.warning(f"Pattern length ({len(pattern)}) doesn't match number of features ({n_features})")
                pattern = pattern[:n_features]  # Truncate if needed
            
            plt.bar(feature_names, pattern)
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.ylim(-1.2, 1.2)
            plt.title(name)
            plt.xticks(rotation=45, ha='right')
            
        plt.tight_layout()
        
        return plt.gcf() 