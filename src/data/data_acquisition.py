import yfinance as yf
import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataAcquisition:
    """
    Class for acquiring financial data from various sources.
    Currently supports Yahoo Finance via yfinance.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data acquisition module.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created data directory: {data_dir}")
    
    def fetch_data_yf(self, 
                    ticker: str, 
                    start_date: Optional[Union[str, datetime]] = None,
                    end_date: Optional[Union[str, datetime]] = None,
                    period: str = "1y", 
                    interval: str = "1d",
                    auto_adjust: bool = True,
                    save_data: bool = True) -> pd.DataFrame:
        """
        Fetch historical data for a given ticker from Yahoo Finance.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date for data (format: 'YYYY-MM-DD')
            end_date: End date for data (format: 'YYYY-MM-DD')
            period: Period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            auto_adjust: Adjust OHLC automatically
            save_data: Whether to save the data to a CSV file
            
        Returns:
            DataFrame with historical data
        """
        try:
            logger.info(f"Fetching data for {ticker} from Yahoo Finance...")
            
            # If start_date and end_date are provided, ignore period
            if start_date is not None and end_date is not None:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    progress=False
                )
            else:
                data = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    progress=False
                )
            
            # Check if data is empty
            if data.empty:
                logger.warning(f"No data found for {ticker}.")
                return pd.DataFrame()
            
            # Reset index to have Date as a column
            data = data.reset_index()
            
            # Add ticker column for multi-ticker datasets
            data['Ticker'] = ticker
            
            logger.info(f"Retrieved {len(data)} rows of data for {ticker}.")
            
            # Save data if requested
            if save_data:
                self._save_data(data, ticker, interval)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_tickers(self, 
                             tickers: List[str], 
                             **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            **kwargs: Additional arguments for fetch_data_yf
            
        Returns:
            Dictionary of DataFrames with ticker as key
        """
        results = {}
        
        for ticker in tickers:
            data = self.fetch_data_yf(ticker, **kwargs)
            if not data.empty:
                results[ticker] = data
        
        return results
    
    def _save_data(self, data: pd.DataFrame, ticker: str, interval: str) -> None:
        """
        Save data to CSV file.
        
        Args:
            data: DataFrame to save
            ticker: Ticker symbol
            interval: Data interval for filename
        """
        # Create filename
        today = datetime.now().strftime('%Y%m%d')
        filename = f"{ticker}_{interval}_{today}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            data.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {str(e)}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            data = pd.read_csv(filepath)
            
            # Convert date column to datetime
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            
            logger.info(f"Loaded {len(data)} rows from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {str(e)}")
            return pd.DataFrame()
    
    def find_latest_data_file(self, ticker: str, interval: str = "1d") -> Optional[str]:
        """
        Find the latest data file for a given ticker and interval.
        
        Args:
            ticker: Ticker symbol
            interval: Data interval
            
        Returns:
            Path to the latest file or None if not found
        """
        pattern = f"{ticker}_{interval}_"
        matching_files = [f for f in os.listdir(self.data_dir) if f.startswith(pattern) and f.endswith('.csv')]
        
        if not matching_files:
            logger.warning(f"No data files found for {ticker} with interval {interval}")
            return None
        
        # Sort by date (filename format should be ticker_interval_YYYYMMDD.csv)
        matching_files.sort(reverse=True)
        latest_file = os.path.join(self.data_dir, matching_files[0])
        
        logger.info(f"Latest data file for {ticker}: {latest_file}")
        return latest_file
    
    def clean_data(self, data: pd.DataFrame, handle_missing: str = 'ffill') -> pd.DataFrame:
        """
        Clean the data by handling missing values and ensuring proper types.
        
        Args:
            data: DataFrame to clean
            handle_missing: Method to handle missing values ('ffill', 'bfill', 'drop', or 'interpolate')
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        clean_data = data.copy()
        
        # Convert date column to datetime if it exists and isn't already
        if 'Date' in clean_data.columns and not pd.api.types.is_datetime64_any_dtype(clean_data['Date']):
            clean_data['Date'] = pd.to_datetime(clean_data['Date'])
        
        # Handle multi-level columns (sometimes returned by yfinance)
        if isinstance(clean_data.columns, pd.MultiIndex):
            # Get the ticker name from the second level
            ticker = clean_data.columns.get_level_values(1)[0]
            # Reset the columns to the first level
            clean_data.columns = clean_data.columns.get_level_values(0)
            # Add the ticker column
            clean_data['Ticker'] = ticker
        
        # Standardize column names (Yahoo Finance usually returns uppercase column names)
        column_mapping = {
            'OPEN': 'Open',
            'HIGH': 'High',
            'LOW': 'Low',
            'CLOSE': 'Close',
            'VOLUME': 'Volume',
            'ADJ CLOSE': 'Adj Close',
            'Adj_Close': 'Adj Close'
        }
        
        # Apply column mapping for string column names
        clean_data.rename(columns=lambda col: column_mapping.get(col.upper(), col) if isinstance(col, str) else col, inplace=True)
        
        # Check for missing values
        missing_count = clean_data.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values in the data")
            
            # Handle missing values based on method
            if handle_missing == 'ffill':
                clean_data = clean_data.ffill()
            elif handle_missing == 'bfill':
                clean_data = clean_data.bfill()
            elif handle_missing == 'drop':
                clean_data = clean_data.dropna()
            elif handle_missing == 'interpolate':
                clean_data = clean_data.interpolate(method='linear')
            else:
                logger.warning(f"Unrecognized method '{handle_missing}' for handling missing values")
        
        # Check for and handle outliers
        # This is a simple example - replace with more sophisticated outlier detection if needed
        numeric_cols = clean_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Identify outliers using IQR method
            Q1 = clean_data[col].quantile(0.25)
            Q3 = clean_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Count outliers
            outliers = ((clean_data[col] < lower_bound) | (clean_data[col] > upper_bound)).sum()
            if outliers > 0:
                logger.info(f"Found {outliers} outliers in column '{col}'")
                
                # Cap outliers at boundaries
                clean_data[col] = clean_data[col].clip(lower=lower_bound, upper=upper_bound)
        
        return clean_data 