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

    def calculate_rfms_score(self):
        """
        Calculate RFMS scores for each user.
        - Recency: Use the inverse of TransactionStartTime
        - Frequency: Count the number of transactions per CustomerId
        - Monetary: Sum of Amount per CustomerId
        - Size: Count the number of unique subscriptions per CustomerId
        """
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        today = self.df['TransactionStartTime'].max() + pd.Timedelta(days=1)

        recency = self.df.groupby('CustomerId')['TransactionStartTime'].max().apply(lambda x: (today - x).days)
        frequency = self.df.groupby('CustomerId')['TransactionId'].count()
        monetary = self.df.groupby('CustomerId')['Amount'].sum()
        size = self.df.groupby('CustomerId')['SubscriptionId'].nunique()
        no_account = self.df.groupby('CustomerId')['AccountId'].nunique()

        rfms = pd.DataFrame({'Recency': recency, 'Frequency': frequency, 'Monetary': monetary, 'No_Subscription': size,'No_Account':no_account})
        rfms['RFMS_Score'] = rfms['Recency'] * 0.25 + rfms['Frequency'] * 0.25 + rfms['Monetary'] * 0.25 + rfms['No_Subscription'] * 0.25
        self.rfms_scores = rfms.sort_values(by='RFMS_Score', ascending=False)
        return self.rfms_scores

    def classify_good_bad(self, threshold):
        """
        Classify users into 'good' and 'bad' categories based on RFMS score.
        :param threshold: RFMS score threshold for classification.
        """
        if self.rfms_scores is None:
            raise ValueError("RFMS scores are not calculated. Please run calculate_rfms_score() first.")
        
        self.rfms_scores['Risk'] = np.where(self.rfms_scores['RFMS_Score'] >= threshold, 'good', 'bad')
        return self.rfms_scores

    def perform_woe_binning(self, variable, target='Risk'):
        """
        Perform Weight of Evidence (WoE) binning.
        :param variable: The variable to bin using WoE.
        :param target: The target column (default: 'Risk').
        """
        if target not in self.rfms_scores.columns:
            raise ValueError(f"'{target}' column not found in RFMS scores. Please ensure classification is done.")

        # Merge RFMS scores back to the main DataFrame for binning
        self.df = self.df.merge(self.rfms_scores[['Risk']], left_on='CustomerId', right_index=True, how='left')
        
        # Initialize WoE binning
        woe_df = pd.DataFrame(columns=['Bin', 'Good', 'Bad', 'Total', 'Good%_Bin', 'Bad%_Bin', 'WoE'])

        binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        self.df[f'{variable}_bins'] = binner.fit_transform(self.df[[variable]])

        # Calculate WoE for each bin
        for bin_id in self.df[f'{variable}_bins'].unique():
            bin_data = self.df[self.df[f'{variable}_bins'] == bin_id]
            good = bin_data[target].value_counts().get('good', 0)
            bad = bin_data[target].value_counts().get('bad', 0)
            total = good + bad

            good_pct_bin = good / total if total > 0 else 0
            bad_pct_bin = bad / total if total > 0 else 0

            # Avoid division by zero in WoE calculation
            if good_pct_bin > 0 and bad_pct_bin > 0:
                woe = np.log(good_pct_bin / bad_pct_bin)
            else:
                woe = 0

            new_row = pd.DataFrame([{'Bin': bin_id, 'Good': good, 'Bad': bad, 'Total': total,
                                    'Good%_Bin': good_pct_bin, 'Bad%_Bin': bad_pct_bin, 'WoE': woe}])
            woe_df = pd.concat([woe_df, new_row], ignore_index=True)
        
        self.binned_data = woe_df
        return self.binned_data


    def plot_rfms_distribution(self):
        """Visualize the RFMS distribution for 'good' and 'bad' users."""
        if self.rfms_scores is None:
            raise ValueError("RFMS scores are not calculated. Please run calculate_rfms_score() first.")

        plt.figure(figsize=(10, 6))
        for risk in ['good', 'bad']:
            subset = self.rfms_scores[self.rfms_scores['Risk'] == risk]
            plt.hist(subset['RFMS_Score'], bins=10, alpha=0.7, label=risk)
        
        plt.title('RFMS Score Distribution')
        plt.xlabel('RFMS Score')
        plt.ylabel('Count')
        plt.legend()
        plt.show()
