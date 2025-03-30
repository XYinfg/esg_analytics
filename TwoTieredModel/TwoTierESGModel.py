from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TwoTierESGModel:
    def __init__(self, cat_features, num_features):
        """
        Initialize the two-tier ESG prediction model.

        Parameters:
        -----------
        cat_features : list
            List of categorical feature names
        num_features : list
            List of numerical feature names
        """
        self.cat_features = cat_features
        self.num_features = num_features

        # Define the preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_features),
                ('cat', categorical_transformer, cat_features)
            ])

        # Tier 1: Model for predicting pillar and disclosure scores
        self.tier1_model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42))
        ])

        # Tier 2: Linear regression model to calculate final ESG score from pillar and disclosure scores
        self.tier2_model = LinearRegression()

        # Names of the intermediate targets (pillar and disclosure scores)
        self.pillar_disclosure_columns = [
            'BESG Environmental Pillar Score',
            'BESG Social Pillar Score',
            'BESG Governance Pillar Score',
            'ESG Disclosure Score',
            'Environmental Disclosure Score',
            'Social Disclosure Score',
            'Governance Disclosure Score'
        ]

        # Final target column
        self.final_target = 'BESG ESG Score'

    def fit(self, X_train, y_train):
        """
        Fit both tiers of the model.

        Parameters:
        -----------
        X_train : DataFrame
            Features for training
        y_train : DataFrame
            Targets for training (should include all pillar, disclosure, and ESG scores)
        """
        # Fit Tier 1 model to predict pillar and disclosure scores
        self.tier1_model.fit(X_train, y_train[self.pillar_disclosure_columns])

        # Get predictions from Tier 1 model
        tier1_predictions = self.tier1_model.predict(X_train)

        # Fit Tier 2 model to predict final ESG score from pillar and disclosure predictions
        self.tier2_model.fit(tier1_predictions, y_train[self.final_target])

        return self

    def predict(self, X):
        """
        Make predictions using both tiers of the model.

        Parameters:
        -----------
        X : DataFrame
            Features for prediction

        Returns:
        --------
        dict with keys:
            'pillar_disclosure_scores': predictions of pillar and disclosure scores
            'esg_score': final ESG score prediction
        """
        # Get pillar and disclosure score predictions from Tier 1
        pillar_disclosure_preds = self.tier1_model.predict(X)

        # Use these predictions to get the final ESG score from Tier 2
        esg_score_preds = self.tier2_model.predict(pillar_disclosure_preds)

        return {
            'pillar_disclosure_scores': pillar_disclosure_preds,
            'esg_score': esg_score_preds
        }

    def evaluate(self, X_test, y_test):
        """
        Evaluate the two-tier model on test data.

        Parameters:
        -----------
        X_test : DataFrame
            Features for testing
        y_test : DataFrame
            Targets for testing

        Returns:
        --------
        DataFrame with performance metrics for each target variable
        """
        # Make predictions
        predictions = self.predict(X_test)

        # Prepare metrics dictionary
        metrics_data = []

        # Evaluate pillar and disclosure score predictions (Tier 1)
        for i, col in enumerate(self.pillar_disclosure_columns):
            true_values = y_test[col].values
            pred_values = predictions['pillar_disclosure_scores'][:, i]

            mse = mean_squared_error(true_values, pred_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_values, pred_values)
            r2 = r2_score(true_values, pred_values)

            metrics_data.append({
                'Target': col,
                'Tier': 'Tier 1',
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R-squared': r2
            })

        # Evaluate final ESG score prediction (Tier 2)
        true_values = y_test[self.final_target].values
        pred_values = predictions['esg_score']

        mse = mean_squared_error(true_values, pred_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, pred_values)
        r2 = r2_score(true_values, pred_values)

        metrics_data.append({
            'Target': self.final_target,
            'Tier': 'Tier 2',
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R-squared': r2
        })

        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics_data)

        return metrics_df

# Example usage:
'''
# Initialize the model
model = TwoTierESGModel(cat_features, num_features)

# Fit the model
model.fit(X_train, y_train)

# Evaluate on test data
metrics = model.evaluate(X_test, y_test)
print(metrics)

# Get predictions for new data
predictions = model.predict(X_new)
'''

def plot_actual_vs_predicted_two_tier(model, X_test, y_test):
    """
    Plot actual vs predicted values for all targets in the two-tier model.

    Parameters:
    -----------
    model : TwoTierESGModel
        The trained two-tier model
    X_test : DataFrame
        Test features
    y_test : DataFrame
        Test targets
    """
    # Get predictions
    predictions = model.predict(X_test)

    # Determine the number of outputs (pillar scores + disclosure scores + final score)
    n_outputs = len(model.pillar_disclosure_columns) + 1

    # Create a grid of subplots
    rows = (n_outputs + 3) // 4  # Ceiling division to determine number of rows
    fig, axes = plt.subplots(rows, 4, figsize=(20, rows * 5))
    axes = axes.ravel()  # Flatten the 2D array of axes

    # Plot for pillar and disclosure scores (Tier 1)
    for i, col in enumerate(model.pillar_disclosure_columns):
        # Scatter plot of actual vs predicted for this specific output
        axes[i].scatter(y_test[col], predictions['pillar_disclosure_scores'][:, i], alpha=0.7)

        # Add perfect prediction line
        min_val = min(y_test[col].min(), predictions['pillar_disclosure_scores'][:, i].min())
        max_val = max(y_test[col].max(), predictions['pillar_disclosure_scores'][:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')

        # Set labels and title
        axes[i].set_xlabel(f'Actual {col}')
        axes[i].set_ylabel(f'Predicted {col}')
        axes[i].set_title(f'Tier 1: {col}')
        axes[i].grid(True, linestyle='--', alpha=0.7)

    # Plot for final ESG score (Tier 2)
    i = len(model.pillar_disclosure_columns)
    axes[i].scatter(y_test[model.final_target], predictions['esg_score'], alpha=0.7)

    # Add perfect prediction line
    min_val = min(y_test[model.final_target].min(), predictions['esg_score'].min())
    max_val = max(y_test[model.final_target].max(), predictions['esg_score'].max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')

    # Set labels and title
    axes[i].set_xlabel(f'Actual {model.final_target}')
    axes[i].set_ylabel(f'Predicted {model.final_target}')
    axes[i].set_title(f'Tier 2: {model.final_target}')
    axes[i].grid(True, linestyle='--', alpha=0.7)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

    return fig