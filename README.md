# Data Analysis Tool

A comprehensive Python-based data analysis tool for processing and visualizing large datasets using Pandas and Matplotlib.

![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.0+-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Data Loading**: Support for multiple file formats (CSV, Excel, JSON, Parquet, Pickle)
- **Data Exploration**: Quick overview of dataset statistics, missing values, and column types
- **Data Cleaning**: Handle missing values and duplicates with various methods
- **Analysis**: Statistical analysis of columns and correlation calculations
- **Visualization**: Create histograms, scatter plots, correlation heatmaps, and boxplots
- **Data Export**: Save processed data in multiple formats
- **Performance Logging**: Track operation times and resource usage

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/data-analysis-tool.git
cd data-analysis-tool
pip install -r requirements.txt
```

### Requirements

- Python 3.6+
- pandas
- matplotlib
- numpy
- argparse

## Usage

### As a Python Module

```python
from data_analysis_tool import DataAnalysisTool

# Initialize with a file
tool = DataAnalysisTool(file_path="your_data.csv")

# Or with existing data
import pandas as pd
data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
tool = DataAnalysisTool(data=data)

# Explore the dataset
tool.explore_data()

# Clean the data
tool.clean_data(drop_duplicates=True, fill_method='mean')

# Analyze a specific column
analysis = tool.analyze_column("A")

# Create visualizations
fig = tool.plot_histogram("A", bins=20, title="Distribution of A")
fig.savefig("histogram.png")

# Generate correlation matrix and plot
tool.plot_correlation_heatmap()

# Save processed data
tool.save_data("processed_data.csv")
```

### Command Line Interface

```bash
python data_analysis_tool.py --file data.csv --explore --clean --analyze column_name --correlate --save processed.csv
```

Command line options:
- `--file` or `-f`: Path to the data file
- `--explore` or `-e`: Explore the data
- `--clean` or `-c`: Clean the data (remove duplicates and fill missing values)
- `--analyze` or `-a`: Analyze a specific column
- `--correlate` or `-r`: Show correlation matrix
- `--save` or `-s`: Save processed data to file

## API Reference

### DataAnalysisTool Class

#### Initialization

```python
DataAnalysisTool(data=None, file_path=None)
```

- `data`: Pre-loaded pandas DataFrame (optional)
- `file_path`: Path to data file (optional)

#### Methods

##### Data Loading

```python
load_data(file_path)
```
Load data from a file (CSV, Excel, JSON, Parquet, Pickle).

##### Data Exploration

```python
explore_data(sample_size=5)
```
Get a comprehensive overview of the data.

##### Data Cleaning

```python
clean_data(drop_na=False, drop_duplicates=False, fill_method=None)
```
Clean the data by handling missing values and duplicates.

- `drop_na`: Whether to drop rows with missing values
- `drop_duplicates`: Whether to drop duplicate rows
- `fill_method`: Method to fill missing values ('mean', 'median', 'mode', or a value)

##### Analysis

```python
analyze_column(column_name)
```
Perform detailed analysis on a specific column.

```python
correlate(columns=None, method='pearson')
```
Calculate correlation between numeric columns.

- `columns`: List of columns to correlate (all numeric columns if None)
- `method`: Correlation method ('pearson', 'kendall', 'spearman')

##### Visualization

```python
plot_histogram(column_name, bins=10, figsize=(10, 6), color='skyblue', title=None, xlabel=None, ylabel='Frequency', save_path=None)
```
Plot a histogram for a numeric column.

```python
plot_scatter(x_column, y_column, color='blue', alpha=0.6, figsize=(10, 6), title=None, xlabel=None, ylabel=None, save_path=None)
```
Create a scatter plot between two columns.

```python
plot_correlation_heatmap(columns=None, method='pearson', cmap='coolwarm', figsize=(12, 10), annot=True, save_path=None)
```
Plot a correlation heatmap.

```python
plot_boxplot(columns=None, figsize=(12, 8), vert=True, title='Boxplot Comparison', save_path=None)
```
Create boxplots for numeric columns.

##### Data Management

```python
save_data(file_path, index=False)
```
Save the current data to a file.

```python
reset_data()
```
Reset data to the original loaded state.

```python
filter_data(conditions)
```
Filter data using boolean conditions.

- `conditions`: String with pandas query syntax (e.g., "column > 5 and column2 == 'value'")

## Examples

### Basic Data Exploration

```python
from data_analysis_tool import DataAnalysisTool

# Load data
tool = DataAnalysisTool(file_path="sales_data.csv")

# Get an overview
tool.explore_data()

# Analyze the 'Revenue' column
tool.analyze_column('Revenue')

# Plot revenue distribution
tool.plot_histogram('Revenue', bins=20, title='Revenue Distribution')
```

### Data Cleaning and Processing

```python
from data_analysis_tool import DataAnalysisTool

# Load data
tool = DataAnalysisTool(file_path="customer_data.csv")

# Clean data - remove duplicates and fill missing values with mean
tool.clean_data(drop_duplicates=True, fill_method='mean')

# Filter data
filtered_data = tool.filter_data("Age > 30 and Income > 50000")

# Save processed data
tool.save_data("processed_customers.csv")
```

### Correlation Analysis and Visualization

```python
from data_analysis_tool import DataAnalysisTool

# Load data
tool = DataAnalysisTool(file_path="housing_data.csv")

# Get correlation matrix
correlation = tool.correlate()
print(correlation)

# Plot correlation heatmap
tool.plot_correlation_heatmap(cmap='viridis', save_path="correlation.png")

# Create a scatter plot
tool.plot_scatter('Square_Feet', 'Price', save_path="price_vs_size.png")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)
