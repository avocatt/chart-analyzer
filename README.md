# Bitcoin Mean Reversion Statistical Analyzer

A Python-based statistical analysis tool for studying Bitcoin price behavior after breaks of previous day's high and low levels.

## Project Description

This project analyzes Bitcoin price data to determine whether there are statistically significant patterns in how price behaves after breaking previous day's high and low levels. The core hypothesis being tested is whether price tends to reverse (mean reversion) or continue after these key level breaks, and what the optimal take-profit and stop-loss parameters would be for a trading strategy based on these patterns.

## Features

- Precise break detection with configurable parameters
- Comprehensive statistical analysis of post-break price movements
- Time-of-day and day-of-week analysis
- Market context analysis (volatility, volume)
- Sequential pattern recognition
- Visualization of key findings and metrics
- Strategy parameter optimization

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/btc-mean-reversion-analyzer.git
cd btc-mean-reversion-analyzer
```

2. Create and activate a virtual environment:
```
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Basic Analysis

```python
from src.analyzer import BreakAnalyzer

# Initialize analyzer with your data
analyzer = BreakAnalyzer(
    minute_data_path='data/BTCUSDT_1m.csv',
    daily_data_path='data/BTCUSDT_1d.csv'
)

# Run analysis with default parameters
results = analyzer.analyze_key_level_breaks()

# Generate summary report
analyzer.generate_report(results, output_path='reports/analysis_report.html')
```

### Parameter Optimization

```python
from src.optimizer import ParameterOptimizer

optimizer = ParameterOptimizer(analyzer)

# Run grid search for optimal parameters
optimal_params = optimizer.grid_search(
    continuation_thresholds=[0.5, 1.0, 1.5, 2.0],
    reversal_thresholds=[0.25, 0.5, 0.75, 1.0],
    break_confirmations=[0.05, 0.1, 0.2],
    window_hours=[4, 8, 12, 24]
)

# Generate optimization report
optimizer.generate_report(optimal_params, output_path='reports/optimization_report.html')
```

## Project Structure

```
/btc-mean-reversion-analyzer
├── data/                 # Data directory
│   ├── BTCUSDT_1m.csv    # 1-minute BTC/USDT price data
│   └── BTCUSDT_1d.csv    # Daily BTC/USDT price data
├── src/                  # Source code
│   ├── analyzer.py       # Core analysis logic
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── optimizer.py      # Parameter optimization
│   └── visualization.py  # Chart generation
├── tests/                # Unit tests
├── config/               # Configuration files
├── reports/              # Generated reports
├── notebooks/            # Jupyter notebooks for exploration
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Configuration

Analyzer parameters can be configured in `config/analyzer_config.yaml`:

```yaml
break_detection:
  confirmation_threshold: 0.1  # % beyond level to confirm break
  
analysis:
  continuation_threshold: 1.0  # % move in break direction to count as continuation
  reversal_threshold: 0.5      # % move against break direction to count as reversal
  window_hours: 24             # Hours to analyze after break
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 