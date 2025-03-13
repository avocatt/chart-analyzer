"""
Chart generation module for Bitcoin Mean Reversion Statistical Analyzer.

This module contains functions for generating visualizations of the analysis results,
including candlestick charts, distribution charts, and heatmaps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_outcome_distribution_chart(results_df, output_path=None):
    """
    Create a chart showing the distribution of outcomes for high and low breaks.
    
    Args:
        results_df: DataFrame with analysis results
        output_path: Path to save the chart (None for no saving)
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if results_df.empty:
        logger.warning("No results for outcome distribution chart")
        return None
    
    logger.info("Creating outcome distribution chart")
    
    # Count outcomes by break type
    outcome_counts = pd.crosstab(results_df['break_type'], results_df['outcome'])
    
    # Convert to percentages
    outcome_pcts = outcome_counts.div(outcome_counts.sum(axis=1), axis=0) * 100
    
    # Create chart
    fig, ax = plt.subplots(figsize=(12, 8))
    outcome_pcts.plot(kind='bar', ax=ax)
    
    ax.set_xlabel('Break Type')
    ax.set_ylabel('Percentage')
    ax.set_title('Outcome Distribution by Break Type')
    ax.legend(title='Outcome')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%')
    
    plt.tight_layout()
    
    # Save chart if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Saved outcome distribution chart to {output_path}")
    
    return fig

def create_time_to_threshold_boxplot(results_df, output_path=None):
    """
    Create a boxplot showing the time to first threshold hit by break type and outcome.
    
    Args:
        results_df: DataFrame with analysis results
        output_path: Path to save the chart (None for no saving)
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if results_df.empty or 'time_to_first_threshold' not in results_df.columns:
        logger.warning("No results for time to threshold boxplot")
        return None
    
    logger.info("Creating time to threshold boxplot")
    
    # Filter out rows with missing time_to_first_threshold
    filtered_df = results_df.dropna(subset=['time_to_first_threshold'])
    
    if filtered_df.empty:
        logger.warning("No valid time to threshold data")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create boxplot with Seaborn
    sns.boxplot(
        x='break_type',
        y='time_to_first_threshold',
        hue='first_to_occur',
        data=filtered_df,
        ax=ax
    )
    
    ax.set_xlabel('Break Type')
    ax.set_ylabel('Time to First Threshold (minutes)')
    ax.set_title('Time to First Threshold by Break Type and Outcome')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add individual data points
    sns.stripplot(
        x='break_type',
        y='time_to_first_threshold',
        hue='first_to_occur',
        data=filtered_df,
        dodge=True,
        alpha=0.5,
        ax=ax,
        legend=False
    )
    
    plt.tight_layout()
    
    # Save chart if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Saved time to threshold boxplot to {output_path}")
    
    return fig

def create_time_of_day_heatmap(results_df, output_path=None):
    """
    Create a heatmap showing break frequency and outcome by hour of day.
    
    Args:
        results_df: DataFrame with analysis results
        output_path: Path to save the chart (None for no saving)
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if results_df.empty or 'hour_of_day' not in results_df.columns:
        logger.warning("No results for time of day heatmap")
        return None
    
    logger.info("Creating time of day heatmap")
    
    # Create figure with 2 subplots (one for each break type)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    break_types = ['high', 'low']
    
    for i, break_type in enumerate(break_types):
        # Filter results for this break type
        filtered_df = results_df[results_df['break_type'] == break_type]
        
        if filtered_df.empty:
            axes[i].text(0.5, 0.5, f"No {break_type} break data", 
                         ha='center', va='center', fontsize=12)
            continue
        
        # Create counts by hour and outcome
        hour_outcome_counts = pd.crosstab(
            filtered_df['hour_of_day'],
            filtered_df['outcome']
        )
        
        # Fill missing hours with zeros
        all_hours = pd.Series(range(24))
        hour_outcome_counts = hour_outcome_counts.reindex(all_hours, fill_value=0)
        
        # Create heatmap
        sns.heatmap(
            hour_outcome_counts,
            cmap='YlGnBu',
            annot=True,
            fmt='d',
            ax=axes[i]
        )
        
        axes[i].set_title(f'{break_type.capitalize()} Breaks by Hour and Outcome')
        axes[i].set_xlabel('Outcome')
        axes[i].set_ylabel('Hour of Day (UTC)')
    
    plt.tight_layout()
    
    # Save chart if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Saved time of day heatmap to {output_path}")
    
    return fig

def create_day_of_week_chart(results_df, output_path=None):
    """
    Create a chart showing break frequency and outcome by day of week.
    
    Args:
        results_df: DataFrame with analysis results
        output_path: Path to save the chart (None for no saving)
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if results_df.empty or 'day_of_week' not in results_df.columns:
        logger.warning("No results for day of week chart")
        return None
    
    logger.info("Creating day of week chart")
    
    # Map day numbers to names
    day_names = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }
    
    # Add day name column
    results_df = results_df.copy()
    results_df['day_name'] = results_df['day_of_week'].map(day_names)
    
    # Create counts by day and break type
    day_counts = pd.crosstab(
        results_df['day_name'],
        results_df['break_type']
    )
    
    # Ensure proper ordering of days
    day_counts = day_counts.reindex([day_names[i] for i in range(7)])
    
    # Create chart
    fig, ax = plt.subplots(figsize=(12, 8))
    day_counts.plot(kind='bar', ax=ax)
    
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Number of Breaks')
    ax.set_title('Break Frequency by Day of Week')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    
    plt.tight_layout()
    
    # Save chart if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Saved day of week chart to {output_path}")
    
    return fig

def create_max_move_histogram(results_df, output_path=None):
    """
    Create histograms showing the distribution of maximum continuation and reversal moves.
    
    Args:
        results_df: DataFrame with analysis results
        output_path: Path to save the chart (None for no saving)
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if (results_df.empty or 
        'max_continuation_pct' not in results_df.columns or
        'max_reversal_pct' not in results_df.columns):
        logger.warning("No results for max move histogram")
        return None
    
    logger.info("Creating max move histogram")
    
    # Create figure with 2 rows (continuation/reversal) and 2 columns (high/low breaks)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    break_types = ['high', 'low']
    move_types = ['max_continuation_pct', 'max_reversal_pct']
    titles = [
        ['Max Continuation % (High Breaks)', 'Max Continuation % (Low Breaks)'],
        ['Max Reversal % (High Breaks)', 'Max Reversal % (Low Breaks)']
    ]
    
    for i, move_type in enumerate(move_types):
        for j, break_type in enumerate(break_types):
            # Filter results
            filtered_df = results_df[results_df['break_type'] == break_type]
            
            if filtered_df.empty:
                axes[i, j].text(0.5, 0.5, f"No {break_type} break data", 
                             ha='center', va='center', fontsize=12)
                continue
            
            # Create histogram
            sns.histplot(
                filtered_df[move_type],
                kde=True,
                ax=axes[i, j]
            )
            
            # Add vertical line at mean
            mean_value = filtered_df[move_type].mean()
            axes[i, j].axvline(mean_value, color='r', linestyle='--', 
                            label=f'Mean: {mean_value:.2f}%')
            
            # Add vertical line at median
            median_value = filtered_df[move_type].median()
            axes[i, j].axvline(median_value, color='g', linestyle='-.',
                            label=f'Median: {median_value:.2f}%')
            
            axes[i, j].set_title(titles[i][j])
            axes[i, j].set_xlabel('Percentage Move')
            axes[i, j].set_ylabel('Frequency')
            axes[i, j].legend()
            axes[i, j].grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save chart if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Saved max move histogram to {output_path}")
    
    return fig

def generate_report_charts(results_df, output_dir='reports/charts'):
    """
    Generate a set of charts for the analysis report.
    
    Args:
        results_df: DataFrame with analysis results
        output_dir: Directory to save charts
        
    Returns:
        list: List of generated chart filenames
    """
    if results_df.empty:
        logger.warning("No results for report charts")
        return []
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate charts
    charts = []
    
    # Outcome distribution chart
    outcome_chart_path = os.path.join(output_dir, f'outcome_distribution_{timestamp}.png')
    create_outcome_distribution_chart(results_df, outcome_chart_path)
    charts.append(outcome_chart_path)
    
    # Time to threshold boxplot
    time_chart_path = os.path.join(output_dir, f'time_to_threshold_{timestamp}.png')
    create_time_to_threshold_boxplot(results_df, time_chart_path)
    charts.append(time_chart_path)
    
    # Time of day heatmap
    time_of_day_path = os.path.join(output_dir, f'time_of_day_{timestamp}.png')
    create_time_of_day_heatmap(results_df, time_of_day_path)
    charts.append(time_of_day_path)
    
    # Day of week chart
    day_of_week_path = os.path.join(output_dir, f'day_of_week_{timestamp}.png')
    create_day_of_week_chart(results_df, day_of_week_path)
    charts.append(day_of_week_path)
    
    # Max move histogram
    max_move_path = os.path.join(output_dir, f'max_move_{timestamp}.png')
    create_max_move_histogram(results_df, max_move_path)
    charts.append(max_move_path)
    
    logger.info(f"Generated {len(charts)} charts in {output_dir}")
    
    return charts

if __name__ == "__main__":
    # Simple test with sample data
    try:
        # Create sample data
        sample_data = {
            'datetime': pd.date_range(start='2024-01-01', periods=50, freq='H'),
            'break_type': ['high'] * 25 + ['low'] * 25,
            'outcome': (['continuation'] * 10 + ['reversal'] * 10 + 
                       ['sideways'] * 5 + ['continuation'] * 10 + 
                       ['reversal'] * 10 + ['sideways'] * 5),
            'max_continuation_pct': np.random.uniform(0.5, 3.0, 50),
            'max_reversal_pct': np.random.uniform(0.2, 2.0, 50),
            'first_to_occur': (['continuation'] * 15 + ['reversal'] * 10 + 
                              ['neither'] * 5 + ['continuation'] * 10 + 
                              ['reversal'] * 15),
            'time_to_first_threshold': np.random.uniform(5, 300, 50),
            'day_of_week': np.random.randint(0, 7, 50),
            'hour_of_day': np.random.randint(0, 24, 50)
        }
        
        sample_df = pd.DataFrame(sample_data)
        
        # Generate test charts
        print("Generating test charts...")
        charts = generate_report_charts(sample_df, 'reports/test_charts')
        
        print(f"Generated {len(charts)} test charts:")
        for chart in charts:
            print(f"- {chart}")
        
    except Exception as e:
        print(f"Error: {e}") 