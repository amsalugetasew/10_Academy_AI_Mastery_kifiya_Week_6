import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
class Plot:
    def __init__(self, data):
        """
        Initialize the EDA class with a DataFrame.
        
        :param data: pd.DataFrame, the dataset to analyze.
        """
        self.data = data
    
    def distribution_numerical_features(self):
        """
        Visualize the distribution of numerical features.
        """
        print("\nDistribution of Numerical Features:")
        numerical_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        for feature in numerical_features:
            plt.figure(figsize=(8, 5))
            sns.histplot(self.data[feature], kde=True, bins=30, color="blue")
            plt.title(f"Distribution of {feature}")
            plt.show()
    def distribution_categorical_features(self, top_n=10):
        """
        Analyze and visualize the distribution of categorical features.
        Only displays the top `n` most frequent categories for each feature.
        """
        print("\nDistribution of Categorical Features:")
        categorical_features = self.data.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            plt.figure(figsize=(8, 5))
            
            # Select the top `n` categories
            top_categories = self.data[feature].value_counts().nlargest(top_n)
            
            # Create the bar plot
            sns.barplot(
                y=top_categories.index, 
                x=top_categories.values, 
                palette="viridis", 
                hue=None  # Explicitly set hue to None for future compatibility
            )
            plt.title(f"Top {top_n} Categories of {feature}")
            plt.xlabel("Count")
            plt.ylabel(feature)
            plt.tight_layout()
            plt.show()





    def correlation_analysis(self):
        """
        Analyze the correlation between numerical features.
        """
        print("\nCorrelation Analysis:")
        numerical_features = self.data.select_dtypes(include=['int64', 'float64'])
        correlation_matrix = numerical_features.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="RdYlGn", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()

    def detect_outliers(self):
        """
        Detect outliers using box plots and calculate their values for numerical features.
        """
        print("\nOutlier Detection:")
        numerical_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        for feature in numerical_features:
            plt.figure(figsize=(8, 5))
            sns.boxplot(
                x=self.data[feature],
                palette=None  # Explicitly set palette to None since hue is not used
            )
            plt.title(f"Box Plot of {feature}")
            plt.xlabel(feature)
            plt.show()
            
            # Calculate IQR for outlier detection
            Q1 = self.data[feature].quantile(0.25)
            Q3 = self.data[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.data[(self.data[feature] < lower_bound) | (self.data[feature] > upper_bound)]
            
            print(f"Feature: {feature}")
            print(f"Outliers Detected:\n{outliers[feature].values}")
