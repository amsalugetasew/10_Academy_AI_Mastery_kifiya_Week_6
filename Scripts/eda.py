class EDA:
    def __init__(self, data):
        """
        Initialize the EDA class with a DataFrame.
        
        :param data: pd.DataFrame, the dataset to analyze.
        """
        self.data = data

    def overview(self):
        """
        Print an overview of the dataset, including rows, columns, and data types.
        """
        print("Dataset Overview:")
        print(f"Number of rows: {self.data.shape[0]}")
        print(f"Number of columns: {self.data.shape[1]}")
        print("\nData Types:\n", self.data.dtypes)

    def summary_statistics(self):
        """
        Display summary statistics for numerical features.
        """
        print("\nSummary Statistics:")
        return self.data.describe()


    def identify_missing_values(self, df):
        """
        Identify missing values in the dataset.
        """
        self.data = df
        print("\nMissing Values:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])

    



