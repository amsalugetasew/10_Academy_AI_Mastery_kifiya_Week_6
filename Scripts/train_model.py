import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

class RiskPredictionModel:
    def __init__(self, df):
        self.df = df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.log_reg = None
        self.best_rf = None

    def preprocess_data(self):
        # Ensure TransactionStartTime is set as the index
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        self.df.set_index('TransactionStartTime', inplace=True)

        # Encode Categorical Variables
        label_encoder = LabelEncoder()
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = label_encoder.fit_transform(self.df[col])

        # Feature Scaling
        scaler = StandardScaler()
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])

        # Convert Risk column: greater than 0 -> 1, else -> 0
        self.df['Risk'] = self.df['Risk'].apply(lambda x: 1 if x > 0 else 0)

    def split_data(self):
        X = self.df.drop(['Risk'], axis=1)
        y = self.df['Risk']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    def train_logistic_regression(self):
        self.log_reg = LogisticRegression(random_state=42, max_iter=1000)
        self.log_reg.fit(self.X_train, self.y_train)
        return self.log_reg

    def train_random_forest(self):
        self.rf = RandomForestClassifier(random_state=42)
        self.rf.fit(self.X_train, self.y_train)
        return self.rf
    def grid_search_train_random_forest(self):    
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
        }
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='roc_auc', cv=5)
        grid_search.fit(self.X_train, self.y_train)

        self.best_rf = grid_search.best_estimator_
        return self.best_rf

    def random_forest_randomized_search(self):
        param_distributions = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
        }
        random_search = RandomizedSearchCV(
            estimator=self.best_rf, param_distributions=param_distributions,
            n_iter=50, scoring='roc_auc', cv=5, random_state=42
        )
        random_search.fit(self.X_train, self.y_train)
        self.best_rf_r = random_search.best_estimator_
        return self.best_rf_r
    def evaluate_models(self):
        # Logistic Regression Predictions
        log_reg_preds = self.log_reg.predict(self.X_test)

        # Random Forest Predictions
        rf_preds = self.rf.predict(self.X_test)

        # Random Forest Predictions
        rf_preds_g = self.best_rf.predict(self.X_test)

        # Random Forest Predictions
        rf_preds_r = self.best_rf_r.predict(self.X_test)

        # Logistic Regression Evaluation
        print("Logistic Regression:")
        print("Accuracy:", accuracy_score(self.y_test, log_reg_preds))
        print("Precision:", precision_score(self.y_test, log_reg_preds))
        print("Recall:", recall_score(self.y_test, log_reg_preds))
        print("F1 Score:", f1_score(self.y_test, log_reg_preds))
        print("ROC-AUC:", roc_auc_score(self.y_test, self.log_reg.predict_proba(self.X_test)[:, 1]))

        # Random Forest Evaluation
        print("\nRandom Forest:")
        print("Accuracy:", accuracy_score(self.y_test, rf_preds))
        print("Precision:", precision_score(self.y_test, rf_preds))
        print("Recall:", recall_score(self.y_test, rf_preds))
        print("F1 Score:", f1_score(self.y_test, rf_preds))
        print("ROC-AUC:", roc_auc_score(self.y_test, self.rf.predict_proba(self.X_test)[:, 1]))


        # Grid Search Random Forest Evaluation
        print("\nRandom Forest:")
        print("Accuracy:", accuracy_score(self.y_test, rf_preds_g))
        print("Precision:", precision_score(self.y_test, rf_preds_g))
        print("Recall:", recall_score(self.y_test, rf_preds_g))
        print("F1 Score:", f1_score(self.y_test, rf_preds_g))
        print("ROC-AUC:", roc_auc_score(self.y_test, self.best_rf.predict_proba(self.X_test)[:, 1]))

        # Random Search Random Forest Evaluation
        print("\nRandom Forest:")
        print("Accuracy:", accuracy_score(self.y_test, rf_preds_r))
        print("Precision:", precision_score(self.y_test, rf_preds_r))
        print("Recall:", recall_score(self.y_test, rf_preds_r))
        print("F1 Score:", f1_score(self.y_test, rf_preds_r))
        print("ROC-AUC:", roc_auc_score(self.y_test, self.best_rf_r.predict_proba(self.X_test)[:, 1]))

        
        return log_reg_preds, rf_preds, rf_preds_g, rf_preds_r
    
    def selected_model_train_random_forest(self):
        self.s_rf = RandomForestClassifier(n_estimators=50, max_depth=10,random_state=42)
        self.s_rf.fit(self.X_train, self.y_train)
        return self.s_rf
    def save_model(self):
        # Create the directory if it doesn't exist
        model_dir = '../CreditRiskPredictionAPI/Model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # Save the trained model
        joblib.dump(self.s_rf, os.path.join(model_dir, 'random_forest_model.pkl'))
    def summarize_model(self):
        # Summarize the model's parameters and feature importances
        print("Model Parameters:")
        print(f"Number of Trees: {self.s_rf.n_estimators}")
        print(f"Max Depth: {self.s_rf.max_depth}")
        print("\nFeature Importances:")
        for i, col in enumerate(self.X_train.columns):
            print(f"{col}: {self.s_rf.feature_importances_[i]}")
    def visualize_model(self):
        # Visualize Model Parameters
        print("Visualizing Model Hyperparameters:")
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        # Plotting number of trees and max depth
        ax.bar(["Number of Trees", "Max Depth"], [self.s_rf.n_estimators, self.s_rf.max_depth], color=["skyblue", "orange"])
        ax.set_title("Random Forest Model Hyperparameters")
        ax.set_ylabel("Values")
        plt.show()

        # Visualize Feature Importances
        feature_importances = self.s_rf.feature_importances_
        feature_names = self.X_train.columns

        # Create a DataFrame for easy plotting with Seaborn
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        })

        # Sort by importance
        feature_df = feature_df.sort_values(by='Importance', ascending=False).head(20)

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()
    def evaluation_of_selected_model(self):
        # Evaluate the model using metrics
        rf_preds = self.s_rf.predict(self.X_test)

        print("Random Forest Model Evaluation:")
        print(f"Accuracy: {accuracy_score(self.y_test, rf_preds)}")
        print(f"Precision: {precision_score(self.y_test, rf_preds)}")
        print(f"Recall: {recall_score(self.y_test, rf_preds)}")
        print(f"F1 Score: {f1_score(self.y_test, rf_preds)}")
        print(f"ROC-AUC: {roc_auc_score(self.y_test, self.s_rf.predict_proba(self.X_test)[:, 1])}")

    
    def roc_auc_curve(self):
        # Get predicted probabilities for the ROC-AUC curve
        y_pred_proba = self.s_rf.predict_proba(self.X_test)[:, 1]  # Probabilities for the positive class

        # Compute ROC curve and AUC score
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='skyblue', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

    def predicted_vs_actual_diff(self):
        # Predict values using the trained Random Forest model
        rf_preds = self.s_rf.predict(self.X_test)
        
        # Calculate the difference (residuals) between the predicted and actual values
        diff = rf_preds - self.y_test

        # Plot the predicted vs actual difference (residuals)
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, diff, color='b', alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted - Actual (Residuals)')
        plt.title('Predicted vs Actual Value Difference')
        plt.show()
