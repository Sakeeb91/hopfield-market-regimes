import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_price_with_regimes(data: pd.DataFrame,
                          classification_df: pd.DataFrame,
                          price_col: str = 'Close',
                          volume_col: str = 'Volume',
                          title: str = None,
                          figsize: Tuple[int, int] = (16, 12)):
    """
    Plot price data with market regime overlays.
    
    Args:
        data: DataFrame with price data
        classification_df: DataFrame with regime classification
        price_col: Column name for price data
        volume_col: Column name for volume data
        title: Plot title
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure
    """
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, 
                                       gridspec_kw={'height_ratios': [3, 1, 1]}, 
                                       sharex=True)
    
    # Plot price data on top subplot
    ax1.plot(data.index, data[price_col], color='black', lw=1.5, label=price_col)
    ax1.set_ylabel('Price', fontsize=12)
    
    if title:
        ax1.set_title(title, fontsize=16)
    else:
        ax1.set_title(f'Price with Market Regimes', fontsize=16)
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Create color map for regimes
    unique_regimes = classification_df['regime'].unique()
    color_map = dict(zip(unique_regimes, sns.color_palette("husl", len(unique_regimes))))
    
    # Color regions by regime
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
            ax1.axvspan(span[0], span[1], alpha=0.3, color=color_map[regime], label=regime if span==spans[0] else "")
    
    # Add volume on middle subplot
    if volume_col in data.columns:
        ax2.bar(data.index, data[volume_col], color='darkgray', alpha=0.7)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.set_visible(False)
    
    # Plot regime on bottom subplot
    # Get all dates from the classification
    all_dates = classification_df.index.tolist()
    all_regimes = classification_df['regime'].tolist()
    
    # Create a categorical y-axis for the regimes
    regime_categories = sorted(unique_regimes)
    regime_indices = [regime_categories.index(r) for r in all_regimes]
    
    # Plot regimes as a scatter plot
    for i, regime in enumerate(regime_categories):
        mask = [r == regime for r in all_regimes]
        ax3.scatter(
            [all_dates[j] for j, m in enumerate(mask) if m], 
            [i] * sum(mask), 
            marker='s', s=100, 
            color=color_map[regime], 
            label=regime
        )
    
    # Set up the regime axis
    ax3.set_yticks(range(len(regime_categories)))
    ax3.set_yticklabels(regime_categories)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Regime', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_regime_transitions(classification_df: pd.DataFrame,
                           figsize: Tuple[int, int] = (12, 10)):
    """
    Plot the regime transitions matrix as a heatmap.
    
    Args:
        classification_df: DataFrame with regime classification
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure
    """
    # Get unique regimes
    regimes = classification_df['regime'].unique()
    n_regimes = len(regimes)
    
    # Initialize transition matrix
    transition_matrix = pd.DataFrame(0, index=regimes, columns=regimes)
    
    # Count transitions
    for i in range(1, len(classification_df)):
        prev_regime = classification_df['regime'].iloc[i-1]
        curr_regime = classification_df['regime'].iloc[i]
        transition_matrix.loc[prev_regime, curr_regime] += 1
    
    # Convert to probabilities (row-wise)
    for regime in regimes:
        row_sum = transition_matrix.loc[regime].sum()
        if row_sum > 0:
            transition_matrix.loc[regime] = transition_matrix.loc[regime] / row_sum
    
    # Visualize the transition matrix
    plt.figure(figsize=figsize)
    sns.heatmap(transition_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
    plt.title('Market Regime Transition Probabilities', fontsize=16)
    plt.tight_layout()
    
    return plt.gcf()


def plot_regime_distribution(classification_df: pd.DataFrame,
                           figsize: Tuple[int, int] = (10, 8)):
    """
    Plot the distribution of market regimes as a pie chart.
    
    Args:
        classification_df: DataFrame with regime classification
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure
    """
    # Count occurrences of each regime
    regime_counts = classification_df['regime'].value_counts()
    
    # Calculate percentages
    regime_pcts = regime_counts / len(classification_df) * 100
    
    # Create labels with percentages
    labels = [f"{regime} ({pct:.1f}%)" for regime, pct in zip(regime_counts.index, regime_pcts)]
    
    # Create pie chart
    plt.figure(figsize=figsize)
    plt.pie(regime_counts, labels=None, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.title('Distribution of Market Regimes', fontsize=16)
    
    # Add legend
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    return plt.gcf()


def plot_regime_returns(data: pd.DataFrame, 
                       classification_df: pd.DataFrame,
                       return_col: str = 'return_1',
                       figsize: Tuple[int, int] = (14, 8)):
    """
    Plot the distribution of returns for each market regime.
    
    Args:
        data: DataFrame with returns data
        classification_df: DataFrame with regime classification
        return_col: Column name for returns data
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure
    """
    # Check if return column exists
    if return_col not in data.columns:
        # Try to calculate returns if 'Close' exists
        if 'Close' in data.columns:
            data[return_col] = data['Close'].pct_change()
        else:
            logger.error(f"Return column '{return_col}' not found and cannot be calculated")
            return None
    
    # Merge returns with classification
    merged_data = pd.DataFrame()
    merged_data['return'] = data[return_col]
    merged_data['date'] = data.index
    
    merged_data = merged_data.set_index('date')
    
    # Align indices
    common_index = merged_data.index.intersection(classification_df.index)
    merged_data = merged_data.loc[common_index]
    regime_data = classification_df.loc[common_index]
    
    merged_data['regime'] = regime_data['regime']
    
    # Prepare data for boxplot
    regime_returns = []
    regime_labels = []
    
    for regime in merged_data['regime'].unique():
        returns = merged_data.loc[merged_data['regime'] == regime, 'return'].dropna() * 100  # Convert to percentage
        regime_returns.append(returns)
        regime_labels.append(f"{regime} (n={len(returns)})")
    
    # Create boxplot
    plt.figure(figsize=figsize)
    plt.boxplot(regime_returns, labels=regime_labels, showmeans=True)
    plt.title('Distribution of Returns by Market Regime', fontsize=16)
    plt.ylabel('Return (%)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add mean labels
    means = [returns.mean() for returns in regime_returns]
    pos = range(1, len(regime_returns) + 1)
    for i, mean_val in enumerate(means):
        plt.annotate(f'Mean: {mean_val:.2f}%', 
                   xy=(pos[i], mean_val), 
                   xytext=(pos[i] + 0.1, mean_val + 0.5),
                   fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    return plt.gcf()


def plot_energy_landscape(energy_values: List[float], 
                        regime_label: str = None,
                        figsize: Tuple[int, int] = (10, 6)):
    """
    Plot the energy landscape during network dynamics.
    
    Args:
        energy_values: List of energy values from network dynamics
        regime_label: Label of the regime for the title
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    plt.plot(energy_values, 'o-', linewidth=2)
    plt.grid(True, alpha=0.3)
    
    if regime_label:
        plt.title(f'Energy Landscape for {regime_label}', fontsize=16)
    else:
        plt.title('Energy Landscape During Network Dynamics', fontsize=16)
        
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Energy', fontsize=14)
    plt.tight_layout()
    
    return plt.gcf()


def save_plots(plots: Dict[str, plt.Figure], 
             output_dir: str, 
             dpi: int = 300):
    """
    Save multiple plots to files.
    
    Args:
        plots: Dictionary mapping filenames to matplotlib figures
        output_dir: Directory to save the plots
        dpi: Resolution of the saved plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each plot
    for filename, fig in plots.items():
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {filepath}")
        plt.close(fig) 