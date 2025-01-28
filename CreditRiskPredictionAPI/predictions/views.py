from sklearn.preprocessing import StandardScaler
from django.shortcuts import render
from .forms import SalesDataForm
import pickle
from datetime import date, timedelta
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

def predict_sales(request):
    """Predicts sales based on user input and loaded model.

    Args:
        request: A Django HTTP request object.

    Returns:
        A Django HTTP response object with the rendered template
        and prediction (if successful) or errors (if any).
    """

    

    model = None
    model_dir = os.path.join(os.path.dirname(__file__), '../../CreditRiskPredictionAPI/Model')
    model_path = os.path.join(model_dir, 'random_forest_model.pkl')

    # Check if the model directory and file exist
    if not os.path.exists(model_dir):
        print(f"Model directory does not exist: {model_dir}")
    elif not os.path.exists(model_path):
        print(f"Model file does not exist: {model_path}")
    else:
        model = joblib.load(model_path)
        print(f"Loaded model type: {type(model)}")  # Check the type of the model


    prediction = None

    if request.method == "POST":
        form = SalesDataForm(request.POST)
        if form.is_valid():
            # Extract cleaned data from the form
            cleaned_data = form.cleaned_data

            # Prepare the input data for prediction as a NumPy array
            features = np.array([[
                # data = form.cleaned_data
                # data['TransactionId'] = f"TransactionId_{data['TransactionId']}"
                # data['AccountId'] = f"AccountId_{data['AccountId']}"
                # data['SubscriptionId'] = f"SubscriptionId_{data['SubscriptionId']}",

                cleaned_data["TransactionId"],         
                cleaned_data["SubscriptionId"],
                cleaned_data["AccountId"],
                cleaned_data["CustomerId"],
                cleaned_data["ProductId"], 
                cleaned_data["ProviderId"], 
                cleaned_data["ProductCategory"],
                cleaned_data["ChannelId"],
                cleaned_data["Amount"],
                cleaned_data["TransactionStartTime"],
                cleaned_data["PricingStrategy"],
                cleaned_data["FraudResult"]
            ]])

            # Convert features into a DataFrame (if necessary for your model)
            features_df = pd.DataFrame(features, columns=[
                'TransactionId', 'SubscriptionId', 'AccountId','CustomerId',
                'ProductId','ProviderId', 'ProductCategory','ChannelId', 'Amount','TransactionStartTime',
                'PricingStrategy', 'FraudResult'
            ])
            features_df['TransactionStartTime'] = pd.to_datetime(features_df['TransactionStartTime'])
            # t_id = f"TransactionId_{features_df['TransactionId'][0]}"
            # features_df[t_id] = features_df['TransactionId'][0]

            # a_id = f"SubscriptionId_{features_df['SubscriptionId'][0]}"
            # features_df[a_id] = features_df['SubscriptionId'][0]


            # s_id = f"AccountId_{features_df['AccountId'][0]}"
            # features_df[s_id] = features_df['AccountId'][0]
            
            # c_id = f"CustomerId_{features_df['CustomerId'][0]}"
            # features_df[c_id] = features_df['CustomerId'][0]

            # p_id = f"ProductId_{features_df['ProductId'][0]}"
            # features_df[p_id] = features_df['ProductId'][0]
            
            Product_category = features_df['ProductCategory'].iloc[0] 
            if Product_category == 'data_bundles':    
                features_df['ProductCategory_data_bundles'] = 1
                features_df['ProductCategory_financial_services'] = 0
                features_df['ProductCategory_movies'] = 0
                features_df['ProductCategory_other'] = 0
                features_df['ProductCategory_ticket'] = 0
                features_df['ProductCategory_tv'] = 0
                features_df['ProductCategory_utility_bill'] = 0
                features_df['ProductCategory_transport'] = 0
            elif Product_category == 'financial_services':
                features_df['ProductCategory_data_bundles'] = 0
                features_df['ProductCategory_financial_services'] = 1
                features_df['ProductCategory_movies'] = 0
                features_df['ProductCategory_other'] = 0
                features_df['ProductCategory_ticket'] = 0
                features_df['ProductCategory_tv'] = 0
                features_df['ProductCategory_utility_bill'] = 0
                features_df['ProductCategory_transport'] = 0
            elif Product_category == 'movies':
                features_df['ProductCategory_data_bundles'] = 0
                features_df['ProductCategory_financial_services'] = 0
                features_df['ProductCategory_movies'] = 1
                features_df['ProductCategory_other'] = 0
                features_df['ProductCategory_ticket'] = 0
                features_df['ProductCategory_tv'] = 0
                features_df['ProductCategory_utility_bill'] = 0
                features_df['ProductCategory_transport'] = 0
            elif Product_category == 'other':
                features_df['ProductCategory_data_bundles'] = 0
                features_df['ProductCategory_financial_services'] = 0
                features_df['ProductCategory_movies'] = 0
                features_df['ProductCategory_other'] = 1
                features_df['ProductCategory_ticket'] = 0
                features_df['ProductCategory_tv'] = 0
                features_df['ProductCategory_utility_bill'] = 0
                features_df['ProductCategory_transport'] = 0
            elif Product_category == 'ticket':
                features_df['ProductCategory_data_bundles'] = 0
                features_df['ProductCategory_financial_services'] = 0
                features_df['ProductCategory_movies'] = 0
                features_df['ProductCategory_other'] = 0
                features_df['ProductCategory_ticket'] = 1
                features_df['ProductCategory_tv'] = 0
                features_df['ProductCategory_utility_bill'] = 0
                features_df['ProductCategory_transport'] = 0
            elif Product_category == 'tv':
                features_df['ProductCategory_data_bundles'] = 0
                features_df['ProductCategory_financial_services'] = 0
                features_df['ProductCategory_movies'] = 0
                features_df['ProductCategory_other'] = 0
                features_df['ProductCategory_ticket'] = 0
                features_df['ProductCategory_tv'] = 1
                features_df['ProductCategory_utility_bill'] = 0
                features_df['ProductCategory_transport'] = 0
            elif Product_category == 'utility_bill':
                features_df['ProductCategory_data_bundles'] = 0
                features_df['ProductCategory_financial_services'] = 0
                features_df['ProductCategory_movies'] = 0
                features_df['ProductCategory_other'] = 0
                features_df['ProductCategory_ticket'] = 0
                features_df['ProductCategory_tv'] = 0
                features_df['ProductCategory_utility_bill'] = 1
                features_df['ProductCategory_transport'] = 0
            else:
                features_df['ProductCategory_data_bundles'] = 0
                features_df['ProductCategory_financial_services'] = 0
                features_df['ProductCategory_movies'] = 0
                features_df['ProductCategory_other'] = 0
                features_df['ProductCategory_ticket'] = 0
                features_df['ProductCategory_tv'] = 0
                features_df['ProductCategory_utility_bill'] = 0
                features_df['ProductCategory_transport'] = 1
            # features_df['ProviderId'] = f"ProductId_{features_df['ProductId'][0]}"
            Product_id = features_df['ProductId'].iloc[0]
            
            if Product_id == 2:
                features_df['ProductId_ProductId_10']  = 0
                features_df['ProductId_ProductId_11'] = 0
                features_df['ProductId_ProductId_12'] = 0
                features_df['ProductId_ProductId_13'] = 0
                features_df['ProductId_ProductId_14'] = 0
                features_df['ProductId_ProductId_15'] = 0
                features_df['ProductId_ProductId_16'] = 0
                features_df['ProductId_ProductId_19'] = 0
                features_df['ProductId_ProductId_2'] = 1
                features_df['ProductId_ProductId_20'] = 0
                features_df['ProductId_ProductId_21']  = 0
                features_df['ProductId_ProductId_22'] = 0
                features_df['ProductId_ProductId_23'] = 0
                features_df['ProductId_ProductId_24'] = 0
                features_df['ProductId_ProductId_27'] = 0
                features_df['ProductId_ProductId_3'] = 0
                features_df['ProductId_ProductId_4'] = 0
                features_df['ProductId_ProductId_5'] = 0
                features_df['ProductId_ProductId_6'] = 0
                features_df['ProductId_ProductId_7'] = 0
                features_df['ProductId_ProductId_8'] = 0
                features_df['ProductId_ProductId_9'] = 0
            elif Product_id == 10:
                features_df['ProductId_ProductId_10']  = 1
                features_df['ProductId_ProductId_11'] = 0
                features_df['ProductId_ProductId_12'] = 0
                features_df['ProductId_ProductId_13'] = 0
                features_df['ProductId_ProductId_14'] = 0
                features_df['ProductId_ProductId_15'] = 0
                features_df['ProductId_ProductId_16'] = 0
                features_df['ProductId_ProductId_19'] = 0
                features_df['ProductId_ProductId_2'] = 0
                features_df['ProductId_ProductId_20'] = 0
                features_df['ProductId_ProductId_21']  = 0
                features_df['ProductId_ProductId_22'] = 0
                features_df['ProductId_ProductId_23'] = 0
                features_df['ProductId_ProductId_24'] = 0
                features_df['ProductId_ProductId_27'] = 0
                features_df['ProductId_ProductId_3'] = 0
                features_df['ProductId_ProductId_4'] = 0
                features_df['ProductId_ProductId_5'] = 0
                features_df['ProductId_ProductId_6'] = 0
                features_df['ProductId_ProductId_7'] = 0
                features_df['ProductId_ProductId_8'] = 0
                features_df['ProductId_ProductId_9'] = 0
            elif Product_id == 11:
                features_df['ProductId_ProductId_10']  = 0
                features_df['ProductId_ProductId_11'] = 1
                features_df['ProductId_ProductId_12'] = 0
                features_df['ProductId_ProductId_13'] = 0
                features_df['ProductId_ProductId_14'] = 0
                features_df['ProductId_ProductId_15'] = 0
                features_df['ProductId_ProductId_16'] = 0
                features_df['ProductId_ProductId_19'] = 0
                features_df['ProductId_ProductId_2'] = 0
                features_df['ProductId_ProductId_20'] = 0
                features_df['ProductId_ProductId_21']  = 0
                features_df['ProductId_ProductId_22'] = 0
                features_df['ProductId_ProductId_23'] = 0
                features_df['ProductId_ProductId_24'] = 0
                features_df['ProductId_ProductId_27'] = 0
                features_df['ProductId_ProductId_3'] = 0
                features_df['ProductId_ProductId_4'] = 0
                features_df['ProductId_ProductId_5'] = 0
                features_df['ProductId_ProductId_6'] = 0
                features_df['ProductId_ProductId_7'] = 0
                features_df['ProductId_ProductId_8'] = 0
                features_df['ProductId_ProductId_9'] = 0
            elif Product_id == 12:
                features_df['ProductId_ProductId_10']  = 0
                features_df['ProductId_ProductId_11'] = 0
                features_df['ProductId_ProductId_12'] = 1
                features_df['ProductId_ProductId_13'] = 0
                features_df['ProductId_ProductId_14'] = 0
                features_df['ProductId_ProductId_15'] = 0
                features_df['ProductId_ProductId_16'] = 0
                features_df['ProductId_ProductId_19'] = 0
                features_df['ProductId_ProductId_2'] = 0
                features_df['ProductId_ProductId_20'] = 0
                features_df['ProductId_ProductId_21']  = 0
                features_df['ProductId_ProductId_22'] = 0
                features_df['ProductId_ProductId_23'] = 0
                features_df['ProductId_ProductId_24'] = 0
                features_df['ProductId_ProductId_27'] = 0
                features_df['ProductId_ProductId_3'] = 0
                features_df['ProductId_ProductId_4'] = 0
                features_df['ProductId_ProductId_5'] = 0
                features_df['ProductId_ProductId_6'] = 0
                features_df['ProductId_ProductId_7'] = 0
                features_df['ProductId_ProductId_8'] = 0
                features_df['ProductId_ProductId_9'] = 0
            elif Product_id == 13:
                features_df['ProductId_ProductId_10']  = 0
                features_df['ProductId_ProductId_11'] = 0
                features_df['ProductId_ProductId_12'] = 0
                features_df['ProductId_ProductId_13'] = 1
                features_df['ProductId_ProductId_14'] = 0
                features_df['ProductId_ProductId_15'] = 0
                features_df['ProductId_ProductId_16'] = 0
                features_df['ProductId_ProductId_19'] = 0
                features_df['ProductId_ProductId_2'] = 0
                features_df['ProductId_ProductId_20'] = 0
                features_df['ProductId_ProductId_21']  = 0
                features_df['ProductId_ProductId_22'] = 0
                features_df['ProductId_ProductId_23'] = 0
                features_df['ProductId_ProductId_24'] = 0
                features_df['ProductId_ProductId_27'] = 0
                features_df['ProductId_ProductId_3'] = 0
                features_df['ProductId_ProductId_4'] = 0
                features_df['ProductId_ProductId_5'] = 0
                features_df['ProductId_ProductId_6'] = 0
                features_df['ProductId_ProductId_7'] = 0
                features_df['ProductId_ProductId_8'] = 0
                features_df['ProductId_ProductId_9'] = 0
            elif Product_id == 14:
                features_df['ProductId_ProductId_10']  = 0
                features_df['ProductId_ProductId_11'] = 0
                features_df['ProductId_ProductId_12'] = 0
                features_df['ProductId_ProductId_13'] = 0
                features_df['ProductId_ProductId_14'] = 1
                features_df['ProductId_ProductId_15'] = 0
                features_df['ProductId_ProductId_16'] = 0
                features_df['ProductId_ProductId_19'] = 0
                features_df['ProductId_ProductId_2'] = 0
                features_df['ProductId_ProductId_20'] = 0
                features_df['ProductId_ProductId_21']  = 0
                features_df['ProductId_ProductId_22'] = 0
                features_df['ProductId_ProductId_23'] = 0
                features_df['ProductId_ProductId_24'] = 0
                features_df['ProductId_ProductId_27'] = 0
                features_df['ProductId_ProductId_3'] = 0
                features_df['ProductId_ProductId_4'] = 0
                features_df['ProductId_ProductId_5'] = 0
                features_df['ProductId_ProductId_6'] = 0
                features_df['ProductId_ProductId_7'] = 0
                features_df['ProductId_ProductId_8'] = 0
                features_df['ProductId_ProductId_9'] = 0
            elif Product_id == 15:
                features_df['ProductId_ProductId_10']  = 0
                features_df['ProductId_ProductId_11'] = 0
                features_df['ProductId_ProductId_12'] = 0
                features_df['ProductId_ProductId_13'] = 0
                features_df['ProductId_ProductId_14'] = 0
                features_df['ProductId_ProductId_15'] = 1
                features_df['ProductId_ProductId_16'] = 0
                features_df['ProductId_ProductId_19'] = 0
                features_df['ProductId_ProductId_2'] = 0
                features_df['ProductId_ProductId_20'] = 0
                features_df['ProductId_ProductId_21']  = 0
                features_df['ProductId_ProductId_22'] = 0
                features_df['ProductId_ProductId_23'] = 0
                features_df['ProductId_ProductId_24'] = 0
                features_df['ProductId_ProductId_27'] = 0
                features_df['ProductId_ProductId_3'] = 0
                features_df['ProductId_ProductId_4'] = 0
                features_df['ProductId_ProductId_5'] = 0
                features_df['ProductId_ProductId_6'] = 0
                features_df['ProductId_ProductId_7'] = 0
                features_df['ProductId_ProductId_8'] = 0
                features_df['ProductId_ProductId_9'] = 0
            else:
                features_df['ProductId_ProductId_10']  = 0
                features_df['ProductId_ProductId_11'] = 0
                features_df['ProductId_ProductId_12'] = 0
                features_df['ProductId_ProductId_13'] = 0
                features_df['ProductId_ProductId_14'] = 0
                features_df['ProductId_ProductId_15'] = 0
                features_df['ProductId_ProductId_16'] = 0
                features_df['ProductId_ProductId_19'] = 0
                features_df['ProductId_ProductId_2'] = 0
                features_df['ProductId_ProductId_20'] = 0
                features_df['ProductId_ProductId_21']  = 0
                features_df['ProductId_ProductId_22'] = 0
                features_df['ProductId_ProductId_23'] = 0
                features_df['ProductId_ProductId_24'] = 0
                features_df['ProductId_ProductId_27'] = 0
                features_df['ProductId_ProductId_3'] = 0
                features_df['ProductId_ProductId_4'] = 0
                features_df['ProductId_ProductId_5'] = 0
                features_df['ProductId_ProductId_6'] = 0
                features_df['ProductId_ProductId_7'] = 0
                features_df['ProductId_ProductId_8'] = 0
                features_df['ProductId_ProductId_9'] = 1


            Provider_id = features_df['ProviderId'].iloc[0] 
            if Provider_id == 1:
                features_df['ProviderId_ProviderId_2'] = 1
                features_df['ProviderId_ProviderId_3'] = 0
                features_df['ProviderId_ProviderId_4'] = 0
                features_df['ProviderId_ProviderId_5'] = 0
                features_df['ProviderId_ProviderId_6'] = 0
            elif Provider_id == 3:
                features_df['ProviderId_ProviderId_2'] = 0
                features_df['ProviderId_ProviderId_3'] = 1
                features_df['ProviderId_ProviderId_4'] = 0
                features_df['ProviderId_ProviderId_5'] = 0
                features_df['ProviderId_ProviderId_6'] = 0
            elif Provider_id == 4:
                features_df['ProviderId_ProviderId_2'] = 0
                features_df['ProviderId_ProviderId_3'] = 0
                features_df['ProviderId_ProviderId_4'] = 1
                features_df['ProviderId_ProviderId_5'] = 0
                features_df['ProviderId_ProviderId_6'] = 0
            elif Provider_id == 5:
                features_df['ProviderId_ProviderId_2'] = 0
                features_df['ProviderId_ProviderId_3'] = 0
                features_df['ProviderId_ProviderId_4'] = 0
                features_df['ProviderId_ProviderId_5'] = 1
                features_df['ProviderId_ProviderId_6'] = 0
            else:
                features_df['ProviderId_ProviderId_2'] = 0
                features_df['ProviderId_ProviderId_3'] = 0
                features_df['ProviderId_ProviderId_4'] = 0
                features_df['ProviderId_ProviderId_5'] = 0
                features_df['ProviderId_ProviderId_6'] = 1


            channel_id = features_df['ChannelId'].iloc[0] 
            if channel_id == 2:
                features_df['ChannelId_ChannelId_2'] = 1
                features_df['ChannelId_ChannelId_3'] = 0
                features_df['ChannelId_ChannelId_5'] = 0
            elif channel_id == 3:
                features_df['ChannelId_ChannelId_2'] = 0
                features_df['ChannelId_ChannelId_3'] = 1
                features_df['ChannelId_ChannelId_5'] = 0
            else:
                features_df['ChannelId_ChannelId_2'] = 0
                features_df['ChannelId_ChannelId_3'] = 0
                features_df['ChannelId_ChannelId_5'] = 1
            
            features_df = features_df.drop(columns = ['ChannelId','ProductCategory',
                                                      'ProductId','ProviderId'])
            
            features_df['TotalTransactionAmount'] = features_df.groupby('CustomerId')['Amount'].transform('sum')
            features_df['AvgTransactionAmount'] = features_df.groupby('CustomerId')['Amount'].transform('mean')
            features_df['TransactionCount'] = features_df.groupby('CustomerId')['TransactionId'].transform('count')
            features_df['StdTransactionAmount'] = 0

            features_df['TransactionStartTime'] = pd.to_datetime(features_df['TransactionStartTime'])
            features_df['TransactionHour'] = features_df['TransactionStartTime'].dt.hour
            features_df['TransactionDay'] = features_df['TransactionStartTime'].dt.day
            features_df['TransactionMonth'] = features_df['TransactionStartTime'].dt.month
            features_df['TransactionYear'] = features_df['TransactionStartTime'].dt.year

            # print(features_df.dtypes)

           
            features_df['TransactionStartTime'] = pd.to_datetime(features_df['TransactionStartTime'])
            today = features_df['TransactionStartTime'].max() + pd.Timedelta(days=1)

            # Calculate Recency for each transaction
            features_df['Recency'] = features_df['TransactionStartTime'].apply(lambda x: (today - x).days)

            # Calculate Frequency for each transaction
            features_df['Frequency'] = features_df.groupby('CustomerId')['TransactionId'].transform('count')

            # Monetary remains the Amount for each transaction
            features_df['Monetary'] = features_df['Amount']

            # Size: Number of unique subscriptions per CustomerId
            features_df['No_Subscription'] = features_df.groupby('CustomerId')['SubscriptionId'].transform('nunique')

            # Number of accounts per CustomerId
            features_df['No_Account'] = features_df.groupby('CustomerId')['AccountId'].transform('nunique')

            # Calculate RFMS score for each transaction
            features_df['RFMS_Score'] = (
                features_df['Recency'] * 0.25 +
                features_df['Frequency'] * 0.25 +
                features_df['Monetary'] * 0.25 +
                features_df['No_Subscription'] * 0.25
            )


            

            rfms_scores = features_df
            threshold = 36
            if rfms_scores is None:
                raise ValueError("RFMS scores are not calculated. Please run calculate_rfms_score() first.")
            rfms_scores['Risk'] = np.where(rfms_scores['RFMS_Score'] >= threshold, 'good', 'bad')
            features_df = rfms_scores

            target='Risk'
            variable='Amount'
            if target not in features_df.columns:
                raise ValueError(f"'{target}' column not found in the dataset. Please ensure classification is done.")

            # Initialize WoE binning result DataFrame
            woe_df = pd.DataFrame(columns=['Bin', 'Good', 'Bad', 'Total', 'Good%_Bin', 'Bad%_Bin', 'WoE'])

            # Bin the variable into quantile bins
            binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
            features_df[f'{variable}_bins'] = binner.fit_transform(features_df[[variable]])

            # Add per-transaction WoE calculation
            bin_woe_map = {}
            for bin_id in features_df[f'{variable}_bins'].unique():
                bin_data = features_df[features_df[f'{variable}_bins'] == bin_id]
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
            features_df[f'{variable}_WoE'] = features_df[f'{variable}_bins'].map(bin_woe_map)

            binned_data = woe_df  # Save the bin-level summary

            # # categorical_columns = ['ProviderId', 'ProductId', 'ProductCategory']
            # numerical_columns = ['Amount', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']

            # scaler = StandardScaler()

            # # Apply Standardization to numerical variables
            # features_df[numerical_columns] = scaler.transform(features_df[numerical_columns])
           




            # categorical_columns = ['TransactionId','SubscriptionId','CustomerId','AccountId','ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']

            features_df[['Amount', 'PricingStrategy', 'FraudResult','TotalTransactionAmount','AvgTransactionAmount', 'Monetary', 'RFMS_Score']] = features_df[['Amount', 'PricingStrategy', 'FraudResult','TotalTransactionAmount','AvgTransactionAmount', 'Monetary', 'RFMS_Score']].astype(float)
            # print(features_df.select_dtypes(include='object').columns)
            scaler = StandardScaler()
            numerical_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
            features_df[numerical_cols] = scaler.fit_transform(features_df[numerical_cols])

            features_df = features_df.drop(columns =['Risk'])
            features_df.set_index('TransactionStartTime', inplace=True)
            
            
            # features_df = features_df.drop(columns = ['ChannelId'])
            numerical_columns = features_df.select_dtypes(include=['float64', 'int64']).columns
            method='standardize'
            if method == 'normalize':
                scaler = MinMaxScaler()
                features_df[numerical_columns] = scaler.fit_transform(features_df[numerical_columns])

            elif method == 'standardize':
                scaler = StandardScaler()
                features_df[numerical_columns] = scaler.fit_transform(features_df[numerical_columns])
        # print(features_df)
        # print(features_df.shape)



            columns  = ['TransactionId', 'AccountId', 'SubscriptionId', 'CustomerId', 'Amount',
            'PricingStrategy', 'FraudResult', 'TotalTransactionAmount',
            'AvgTransactionAmount', 'TransactionCount', 'StdTransactionAmount',
            'TransactionHour', 'TransactionDay', 'TransactionMonth',
            'TransactionYear', 'ProviderId_ProviderId_2', 'ProviderId_ProviderId_3',
            'ProviderId_ProviderId_4', 'ProviderId_ProviderId_5',
            'ProviderId_ProviderId_6', 'ProductId_ProductId_10',
            'ProductId_ProductId_11', 'ProductId_ProductId_12',
            'ProductId_ProductId_13', 'ProductId_ProductId_14',
            'ProductId_ProductId_15', 'ProductId_ProductId_16',
            'ProductId_ProductId_19', 'ProductId_ProductId_2',
            'ProductId_ProductId_20', 'ProductId_ProductId_21',
            'ProductId_ProductId_22', 'ProductId_ProductId_23',
            'ProductId_ProductId_24', 'ProductId_ProductId_27',
            'ProductId_ProductId_3', 'ProductId_ProductId_4',
            'ProductId_ProductId_5', 'ProductId_ProductId_6',
            'ProductId_ProductId_7', 'ProductId_ProductId_8',
            'ProductId_ProductId_9', 'ProductCategory_data_bundles',
            'ProductCategory_financial_services', 'ProductCategory_movies',
            'ProductCategory_other', 'ProductCategory_ticket',
            'ProductCategory_transport', 'ProductCategory_tv',
            'ProductCategory_utility_bill', 'ChannelId_ChannelId_2',
            'ChannelId_ChannelId_3', 'ChannelId_ChannelId_5', 'Recency',
            'Frequency', 'Monetary', 'No_Subscription', 'No_Account', 'RFMS_Score',
            'Amount_bins', 'Amount_WoE']
            new_feature = pd.DataFrame()
            for cols in columns:
                new_feature[cols] = features_df[cols].copy()
            print(new_feature.shape)
            # Make the prediction using the prepared features
            prediction = model.predict(new_feature)[0]  # Assuming the model returns a single value
    else:
        form = SalesDataForm()

    # return render(request, "/templates/predictor/predict.html", {"form": form, "prediction": prediction})
    return render(request, "predictor/predict.html", {"form": form, "prediction": prediction})
