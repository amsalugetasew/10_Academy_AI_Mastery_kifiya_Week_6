import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import chi2_contingency

class CreditScoring:
    def __init__(self, df):
        """
        Initialize the CreditScoring class with the given DataFrame.
        :param df: Input DataFrame containing transaction data.
        """
        self.df = df
        self.rfms_scores = None
        self.binned_data = None

    def calculate_rfms_score(self, df):
        """
        Calculate RFMS scores for each transaction without aggregating.
        - Recency: Use the inverse of TransactionStartTime
        - Frequency: Count the number of transactions per CustomerId
        - Monetary: Amount per transaction
        - Size: Count the number of unique subscriptions per CustomerId
        """
        self.df = df
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        today = self.df['TransactionStartTime'].max() + pd.Timedelta(days=1)

        # Calculate Recency for each transaction
        self.df['Recency'] = self.df['TransactionStartTime'].apply(lambda x: (today - x).days)

        # Calculate Frequency for each transaction
        self.df['Frequency'] = self.df.groupby('CustomerId')['TransactionId'].transform('count')

        # Monetary remains the Amount for each transaction
        self.df['Monetary'] = self.df['Amount']

        # Size: Number of unique subscriptions per CustomerId
        self.df['No_Subscription'] = self.df.groupby('CustomerId')['SubscriptionId'].transform('nunique')

        # Number of accounts per CustomerId
        self.df['No_Account'] = self.df.groupby('CustomerId')['AccountId'].transform('nunique')

        # Calculate RFMS score for each transaction
        self.df['RFMS_Score'] = (
            self.df['Recency'] * 0.25 +
            self.df['Frequency'] * 0.25 +
            self.df['Monetary'] * 0.25 +
            self.df['No_Subscription'] * 0.25
        )

        # Sorting the transactions by RFMS score in descending order
        self.rfms_scores = self.df.sort_values(by='RFMS_Score', ascending=False)
        return self.rfms_scores


    def classify_good_bad(self,df, threshold):
        """
        Classify users into 'good' and 'bad' categories based on RFMS score.
        :param threshold: RFMS score threshold for classification.
        """
        self.rfms_scores = df
        if self.rfms_scores is None:
            raise ValueError("RFMS scores are not calculated. Please run calculate_rfms_score() first.")
        
        print(self.rfms_scores['RFMS_Score'].min(),self.rfms_scores['RFMS_Score'].mean(), self.rfms_scores['RFMS_Score'].max())
        self.rfms_scores['Risk'] = np.where(self.rfms_scores['RFMS_Score'] >= threshold, 'good', 'bad')
        return self.rfms_scores

    def perform_woe_binning(self, df, variable, target='Risk'):
        """
        Perform Weight of Evidence (WoE) binning per transaction.
        :param df: DataFrame containing transaction-level data.
        :param variable: The variable to bin using WoE.
        :param target: The target column indicating 'good' or 'bad' outcomes (default: 'Risk').
        """
        self.df = df.copy()  # Work with a copy to avoid modifying the original DataFrame

        if target not in self.df.columns:
            raise ValueError(f"'{target}' column not found in the dataset. Please ensure classification is done.")

        # Initialize WoE binning result DataFrame
        woe_df = pd.DataFrame(columns=['Bin', 'Good', 'Bad', 'Total', 'Good%_Bin', 'Bad%_Bin', 'WoE'])

        # Bin the variable into quantile bins
        binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        self.df[f'{variable}_bins'] = binner.fit_transform(self.df[[variable]])

        # Add per-transaction WoE calculation
        bin_woe_map = {}
        for bin_id in self.df[f'{variable}_bins'].unique():
            bin_data = self.df[self.df[f'{variable}_bins'] == bin_id]
            good = bin_data[target].value_counts().get('good', 0)
            bad = bin_data[target].value_counts().get('bad', 0)
            total = good + bad

            # Calculate percentages and WoE
            good_pct_bin = good / total if total > 0 else 0
            bad_pct_bin = bad / total if total > 0 else 0

            # Avoid division by zero
            if good_pct_bin > 0 and bad_pct_bin > 0:
                woe = np.log(good_pct_bin / bad_pct_bin)
            else:
                woe = 0

            # Store the results for the bin
            bin_woe_map[bin_id] = woe
            new_row = pd.DataFrame([{'Bin': bin_id, 'Good': good, 'Bad': bad, 'Total': total,
                                    'Good%_Bin': good_pct_bin, 'Bad%_Bin': bad_pct_bin, 'WoE': woe}])
            woe_df = pd.concat([woe_df, new_row], ignore_index=True)

        # Map the WoE back to each transaction
        self.df[f'{variable}_WoE'] = self.df[f'{variable}_bins'].map(bin_woe_map)

        self.binned_data = woe_df  # Save the bin-level summary
        return self.df, self.binned_data


    