import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import argparse
from datetime import datetime

class DataAnalysisTool:
    """
    A comprehensive tool for data analysis and visualization.
    
    This class provides methods for loading, exploring, cleaning,
    analyzing, and visualizing data using pandas and matplotlib.
    """
    
    def __init__(self, data=None, file_path=None):
        """
        Initialize the data analysis tool.
        
        Args:
            data (pd.DataFrame, optional): Pre-loaded data.
            file_path (str, optional): Path to data file.
        """
        self.data = None
        self.original_data = None
        self.summary_stats = None
        self.performance_log = []
        
        if data is not None:
            self.data = data
            self.original_data = data.copy()
        elif file_path is not None:
            self.load_data(file_path)
    
    def load_data(self, file_path):
        """
        Load data from a file.
        
        Args:
            file_path (str): Path to the data file.
            
        Returns:
            pd.DataFrame: The loaded data.
        """
        start_time = time.time()
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        try:
            if ext == '.csv':
                self.data = pd.read_csv(file_path)
            elif ext == '.xlsx' or ext == '.xls':
                self.data = pd.read_excel(file_path)
            elif ext == '.json':
                self.data = pd.read_json(file_path)
            elif ext == '.parquet':
                self.data = pd.read_parquet(file_path)
            elif ext == '.pickle' or ext == '.pkl':
                self.data = pd.read_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
            self.original_data = self.data.copy()
            
            load_time = time.time() - start_time
            self.performance_log.append({
                'operation': 'load_data',
                'file_path': file_path,
                'rows': len(self.data),
                'columns': len(self.data.columns),
                'time_seconds': load_time,
                'timestamp': datetime.now()
            })
            
            print(f"Data loaded successfully from {file_path}")
            print(f"Shape: {self.data.shape} | Time: {load_time:.2f} seconds")
            
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def explore_data(self, sample_size=5):
        """
        Get a comprehensive overview of the data.
        
        Args:
            sample_size (int): Number of rows to display.
            
        Returns:
            dict: Dictionary containing exploration results.
        """
        if self.data is None:
            print("No data loaded. Use load_data() method first.")
            return None
        
        exploration = {
            'head': self.data.head(sample_size),
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes,
            'missing_values': self.data.isnull().sum(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data)) * 100,
            'duplicates': self.data.duplicated().sum()
        }
        
        # Generate summary statistics
        self.summary_stats = self.data.describe(include='all').T
        exploration['summary_stats'] = self.summary_stats
        
        # Print basic information
        print(f"Dataset Overview:")
        print(f"Dimensions: {exploration['shape'][0]} rows, {exploration['shape'][1]} columns")
        print(f"Memory usage: {self.data.memory_usage().sum() / 1024**2:.2f} MB")
        print(f"Duplicate rows: {exploration['duplicates']}")
        print("\nColumn Types:")
        for col, dtype in zip(self.data.columns, self.data.dtypes):
            print(f"  - {col}: {dtype}")
        
        missing_cols = exploration['missing_values'][exploration['missing_values'] > 0]
        if not missing_cols.empty:
            print("\nMissing Values:")
            for col, count in missing_cols.items():
                percentage = (count / len(self.data)) * 100
                print(f"  - {col}: {count} missing ({percentage:.2f}%)")
        
        return exploration
    
    def clean_data(self, drop_na=False, drop_duplicates=False, fill_method=None):
        """
        Clean the data by handling missing values and duplicates.
        
        Args:
            drop_na (bool): Whether to drop rows with missing values.
            drop_duplicates (bool): Whether to drop duplicate rows.
            fill_method (str, optional): Method to fill missing values 
                ('mean', 'median', 'mode', or value).
                
        Returns:
            pd.DataFrame: The cleaned data.
        """
        if self.data is None:
            print("No data loaded. Use load_data() method first.")
            return None
        
        start_time = time.time()
        
        # Create a copy to avoid modifying original
        cleaned_data = self.data.copy()
        original_shape = cleaned_data.shape
        
        # Handle duplicates
        if drop_duplicates:
            cleaned_data = cleaned_data.drop_duplicates()
        
        # Handle missing values
        if fill_method:
            if fill_method == 'mean':
                for col in cleaned_data.select_dtypes(include=[np.number]).columns:
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mean())
            elif fill_method == 'median':
                for col in cleaned_data.select_dtypes(include=[np.number]).columns:
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
            elif fill_method == 'mode':
                for col in cleaned_data.columns:
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mode()[0])
            else:
                # Assume fill_method is a value to fill with
                cleaned_data = cleaned_data.fillna(fill_method)
        
        # Drop rows with any remaining NA values
        if drop_na:
            cleaned_data = cleaned_data.dropna()
        
        # Update the current data
        self.data = cleaned_data
        
        clean_time = time.time() - start_time
        
        # Log performance
        self.performance_log.append({
            'operation': 'clean_data',
            'original_shape': original_shape,
            'new_shape': cleaned_data.shape,
            'rows_removed': original_shape[0] - cleaned_data.shape[0],
            'time_seconds': clean_time,
            'timestamp': datetime.now()
        })
        
        print(f"Data cleaned: {original_shape[0]} → {cleaned_data.shape[0]} rows ({original_shape[0] - cleaned_data.shape[0]} removed)")
        print(f"Cleaning time: {clean_time:.2f} seconds")
        
        return cleaned_data
    
    def analyze_column(self, column_name):
        """
        Perform detailed analysis on a specific column.
        
        Args:
            column_name (str): The name of the column to analyze.
            
        Returns:
            dict: Analysis results for the column.
        """
        if self.data is None:
            print("No data loaded. Use load_data() method first.")
            return None
        
        if column_name not in self.data.columns:
            print(f"Column '{column_name}' not found in data.")
            return None
        
        column_data = self.data[column_name]
        column_type = column_data.dtype
        
        analysis = {
            'name': column_name,
            'dtype': column_type,
            'missing_count': column_data.isnull().sum(),
            'missing_percentage': (column_data.isnull().sum() / len(column_data)) * 100,
            'unique_values': column_data.nunique()
        }
        
        # For numerical columns
        if np.issubdtype(column_type, np.number):
            analysis.update({
                'min': column_data.min(),
                'max': column_data.max(),
                'mean': column_data.mean(),
                'median': column_data.median(),
                'std': column_data.std(),
                'skewness': column_data.skew(),
                'kurtosis': column_data.kurt(),
                'quantiles': column_data.quantile([0.25, 0.5, 0.75]).to_dict()
            })
        
        # For categorical/object columns
        else:
            value_counts = column_data.value_counts()
            analysis.update({
                'most_common': value_counts.head(5).to_dict() if not value_counts.empty else {},
                'least_common': value_counts.tail(5).to_dict() if not value_counts.empty else {}
            })
        
        return analysis
    
    def correlate(self, columns=None, method='pearson'):
        """
        Calculate correlation between numeric columns.
        
        Args:
            columns (list, optional): List of columns to correlate.
            method (str): Correlation method ('pearson', 'kendall', 'spearman').
            
        Returns:
            pd.DataFrame: Correlation matrix.
        """
        if self.data is None:
            print("No data loaded. Use load_data() method first.")
            return None
        
        # Select only numeric columns if not specified
        if columns is None:
            numeric_data = self.data.select_dtypes(include=[np.number])
        else:
            numeric_data = self.data[columns].select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            print("No numeric columns available for correlation.")
            return None
        
        correlation = numeric_data.corr(method=method)
        return correlation
    
    def plot_histogram(self, column_name, bins=10, figsize=(10, 6), color='skyblue', 
                       title=None, xlabel=None, ylabel='Frequency', save_path=None):
        """
        Plot a histogram for a numeric column.
        
        Args:
            column_name (str): The column to plot.
            bins (int): Number of bins.
            figsize (tuple): Figure size.
            color (str): Bar color.
            title (str, optional): Plot title.
            xlabel (str, optional): X-axis label.
            ylabel (str): Y-axis label.
            save_path (str, optional): Path to save the plot.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        if self.data is None:
            print("No data loaded. Use load_data() method first.")
            return None
        
        if column_name not in self.data.columns:
            print(f"Column '{column_name}' not found in data.")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        self.data[column_name].hist(bins=bins, color=color, ax=ax)
        
        # Set labels and title
        ax.set_xlabel(xlabel if xlabel else column_name)
        ax.set_ylabel(ylabel)
        ax.set_title(title if title else f'Histogram of {column_name}')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def plot_scatter(self, x_column, y_column, color='blue', alpha=0.6, figsize=(10, 6),
                    title=None, xlabel=None, ylabel=None, save_path=None):
        """
        Create a scatter plot between two columns.
        
        Args:
            x_column (str): Column for x-axis.
            y_column (str): Column for y-axis.
            color (str): Point color.
            alpha (float): Transparency.
            figsize (tuple): Figure size.
            title (str, optional): Plot title.
            xlabel (str, optional): X-axis label.
            ylabel (str, optional): Y-axis label.
            save_path (str, optional): Path to save the plot.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        if self.data is None:
            print("No data loaded. Use load_data() method first.")
            return None
        
        if x_column not in self.data.columns or y_column not in self.data.columns:
            print(f"One or both columns not found in data.")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot scatter
        ax.scatter(self.data[x_column], self.data[y_column], color=color, alpha=alpha)
        
        # Add regression line
        if self.data[x_column].dtype.kind in 'fcb' and self.data[y_column].dtype.kind in 'fcb':
            try:
                z = np.polyfit(self.data[x_column], self.data[y_column], 1)
                p = np.poly1d(z)
                ax.plot(self.data[x_column], p(self.data[x_column]), "r--", alpha=0.8)
            except:
                pass  # Skip regression line if error occurs
        
        # Set labels and title
        ax.set_xlabel(xlabel if xlabel else x_column)
        ax.set_ylabel(ylabel if ylabel else y_column)
        ax.set_title(title if title else f'{y_column} vs {x_column}')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, columns=None, method='pearson', cmap='coolwarm', 
                                figsize=(12, 10), annot=True, save_path=None):
        """
        Plot a correlation heatmap.
        
        Args:
            columns (list, optional): List of columns to include.
            method (str): Correlation method.
            cmap (str): Colormap.
            figsize (tuple): Figure size.
            annot (bool): Whether to annotate cells.
            save_path (str, optional): Path to save the plot.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        correlation = self.correlate(columns, method)
        if correlation is None:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        heatmap = ax.imshow(correlation, cmap=cmap)
        
        # Add colorbar
        cbar = plt.colorbar(heatmap)
        
        # Set ticks and labels
        tick_labels = correlation.columns
        ax.set_xticks(np.arange(len(tick_labels)))
        ax.set_yticks(np.arange(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_yticklabels(tick_labels)
        
        # Add annotations
        if annot:
            for i in range(len(tick_labels)):
                for j in range(len(tick_labels)):
                    text = ax.text(j, i, f"{correlation.iloc[i, j]:.2f}",
                                ha="center", va="center", color="w" if abs(correlation.iloc[i, j]) > 0.5 else "black")
        
        # Add title
        plt.title(f'{method.capitalize()} Correlation Heatmap')
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def plot_boxplot(self, columns=None, figsize=(12, 8), vert=True, 
                    title='Boxplot Comparison', save_path=None):
        """
        Create boxplots for numeric columns.
        
        Args:
            columns (list, optional): Columns to plot.
            figsize (tuple): Figure size.
            vert (bool): Vertical boxplots if True.
            title (str): Plot title.
            save_path (str, optional): Path to save the plot.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        if self.data is None:
            print("No data loaded. Use load_data() method first.")
            return None
        
        # Select numeric columns
        if columns is None:
            numeric_data = self.data.select_dtypes(include=[np.number])
        else:
            numeric_data = self.data[columns].select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            print("No numeric columns available for boxplot.")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot boxplot
        numeric_data.boxplot(ax=ax, vert=vert)
        
        # Adjust labels and title
        if vert:
            plt.xticks(rotation=45)
        else:
            plt.yticks(rotation=0)
        
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def save_data(self, file_path, index=False):
        """
        Save the current data to a file.
        
        Args:
            file_path (str): Path where to save the data.
            index (bool): Whether to save the index.
            
        Returns:
            bool: Success status.
        """
        if self.data is None:
            print("No data loaded. Use load_data() method first.")
            return False
        
        start_time = time.time()
        
        try:
            # Get file extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext == '.csv':
                self.data.to_csv(file_path, index=index)
            elif ext == '.xlsx':
                self.data.to_excel(file_path, index=index)
            elif ext == '.json':
                self.data.to_json(file_path, orient='records')
            elif ext == '.parquet':
                self.data.to_parquet(file_path, index=index)
            elif ext == '.pickle' or ext == '.pkl':
                self.data.to_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
            save_time = time.time() - start_time
            
            self.performance_log.append({
                'operation': 'save_data',
                'file_path': file_path,
                'rows': len(self.data),
                'columns': len(self.data.columns),
                'time_seconds': save_time,
                'timestamp': datetime.now()
            })
            
            print(f"Data saved successfully to {file_path}")
            print(f"Save time: {save_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return False
    
    def reset_data(self):
        """Reset data to the original loaded state."""
        if self.original_data is not None:
            self.data = self.original_data.copy()
            print("Data has been reset to original state.")
        else:
            print("No original data available.")
    
    def filter_data(self, conditions):
        """
        Filter data using boolean conditions.
        
        Args:
            conditions (str): String with pandas query syntax.
            
        Returns:
            pd.DataFrame: Filtered data.
        """
        if self.data is None:
            print("No data loaded. Use load_data() method first.")
            return None
        
        try:
            filtered_data = self.data.query(conditions)
            print(f"Data filtered: {len(self.data)} → {len(filtered_data)} rows")
            return filtered_data
        except Exception as e:
            print(f"Error filtering data: {str(e)}")
            return None


def main():
    """Command line interface for the data analysis tool."""
    parser = argparse.ArgumentParser(description='Data Analysis Tool')
    parser.add_argument('--file', '-f', type=str, help='Path to the data file')
    parser.add_argument('--explore', '-e', action='store_true', help='Explore the data')
    parser.add_argument('--clean', '-c', action='store_true', help='Clean the data')
    parser.add_argument('--analyze', '-a', type=str, help='Analyze specific column')
    parser.add_argument('--correlate', '-r', action='store_true', help='Show correlation matrix')
    parser.add_argument('--save', '-s', type=str, help='Save processed data to file')
    
    args = parser.parse_args()
    
    # Initialize the tool
    tool = DataAnalysisTool()
    
    # Load data if file provided
    if args.file:
        tool.load_data(args.file)
    else:
        print("No data file provided. Use --file option.")
        return
    
    # Explore data if requested
    if args.explore:
        tool.explore_data()
    
    # Clean data if requested
    if args.clean:
        tool.clean_data(drop_duplicates=True, fill_method='mean')
    
    # Analyze specific column if requested
    if args.analyze:
        analysis = tool.analyze_column(args.analyze)
        if analysis:
            print(f"\nColumn Analysis for '{args.analyze}':")
            for key, value in analysis.items():
                print(f"  {key}: {value}")
    
    # Show correlation if requested
    if args.correlate:
        correlation = tool.correlate()
        if correlation is not None:
            print("\nCorrelation Matrix:")
            print(correlation)
    
    # Save processed data if requested
    if args.save:
        tool.save_data(args.save)


if __name__ == "__main__":
    main()
