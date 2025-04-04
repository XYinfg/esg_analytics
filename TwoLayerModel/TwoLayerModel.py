from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class TwoLayerModel:
    def __init__(self, cat_features, num_features, layer2_model=None, layer2_coefficients=None,
                 estimator=RandomForestRegressor, estimator_params=None):
        """
        Initialize the two-layer ESG prediction model.

        Parameters:
        -----------
        cat_features : list
            List of categorical feature names
        num_features : list
            List of numerical feature names
        layer2_model : estimator, optional
            Model to use for layer 2 (defaults to OLS)
        estimator : class, optional
            Estimator class to use for layer 1 models (defaults to RandomForestRegressor)
        estimator_params : dict, optional
            Parameters for the estimator in layer 1
        """
        self.cat_features = cat_features
        self.num_features = num_features

        # Default parameters for RandomForestRegressor
        self.estimator_params = {'n_estimators': 100, 'max_depth': 15, 'random_state': 42}
        if estimator_params is not None:
            self.estimator_params.update(estimator_params)

        self.estimator = estimator

        # Names of the intermediate targets (pillar and disclosure scores)
        self.pillar_disclosure_columns = [
            'BESG Environmental Pillar Score',
            'BESG Social Pillar Score',
            'BESG Governance Pillar Score',
            # 'ESG Disclosure Score',
            # 'Environmental Disclosure Score',
            # 'Social Disclosure Score',
            # 'Governance Disclosure Score'
        ]

        # Final target column
        self.final_target = 'BESG ESG Score'

        # Layer 2 model
        self.layer2_model = layer2_model
        self.layer2_coefficients = layer2_coefficients
        if self.layer2_coefficients is not None:
            # Ensure coefficients are a numpy array
            self.layer2_coefficients = np.array(self.layer2_coefficients)

        # Dictionary to store individual models
        self.layer1_models = {}

        # Setup preprocessing pipeline
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

    def fit(self, X_train, y_train):
        """
        Fit both layers of the model.

        Parameters:
        -----------
        X_train : DataFrame
            Features for training
        y_train : DataFrame
            Targets for training (should include all pillar, disclosure, and ESG scores)
        """
        # Preprocess the data once
        X_train_processed = self.preprocessor.fit_transform(X_train)

        # Train separate models for each target in layer 1
        layer1_predictions = np.zeros((X_train.shape[0], len(self.pillar_disclosure_columns)))

        for i, target in enumerate(self.pillar_disclosure_columns):
            print(f"Training model for {target}...")

            # Create a specific model for this target
            model = self.estimator(**self.estimator_params)

            # Train the model
            model.fit(X_train_processed, y_train[target])

            # Store the trained model
            self.layer1_models[target] = model

            # Generate predictions for this target
            layer1_predictions[:, i] = model.predict(X_train_processed)

            # Get feature importances
            if hasattr(model, 'feature_importances_'):
                feature_names = self.preprocessor.get_feature_names_out()
                importances = model.feature_importances_

                # Create and store feature importance DataFrame
                feature_importances = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)

                print(f"Top 5 features for {target}:")
                print(feature_importances.head(5))

        # For Layer 2:
        if self.layer2_coefficients is not None:
            print("Using provided Layer 2 coefficients, skipping Layer 2 fitting.")
            # No need to do anything else, as coefficients are already set
        elif self.layer2_model is not None:
            # Use the provided model
            print("Fitting Layer 2 model using provided estimator...")
            self.layer2_model.fit(layer1_predictions, y_train[self.final_target])
        else:
            # Use statsmodels OLS by default
            print("Fitting Layer 2 model using OLS...")
            # Convert target to numpy array to avoid pandas object type issues
            y_train_array = np.array(y_train[self.final_target])

            # Fit the OLS model
            fitted_model = sm.OLS(y_train_array, layer1_predictions).fit()
            print(fitted_model.summary())

            # Store the coefficients
            self.layer2_coefficients = fitted_model.params
            print("Layer 2 coefficients:", self.layer2_coefficients)

        return self

    def predict(self, X):
        """
        Make predictions using both layers of the model.

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
        # Preprocess the input data
        X_processed = self.preprocessor.transform(X)

        # Get predictions from each Layer 1 model
        pillar_disclosure_preds = np.zeros((X.shape[0], len(self.pillar_disclosure_columns)))

        for i, target in enumerate(self.pillar_disclosure_columns):
            pillar_disclosure_preds[:, i] = self.layer1_models[target].predict(X_processed)

        # Use these predictions with Layer 2 model
        if self.layer2_model is None:
            # Check the length of coefficients to determine if a constant was included
            if len(self.layer2_coefficients) == len(self.pillar_disclosure_columns) + 1:
                # If coefficients include a constant (intercept)
                pillar_preds_with_const = sm.add_constant(pillar_disclosure_preds)
                esg_score_preds = np.dot(pillar_preds_with_const, self.layer2_coefficients)
            else:
                # If coefficients don't include a constant
                esg_score_preds = np.dot(pillar_disclosure_preds, self.layer2_coefficients)
        else:
            # If we're using a scikit-learn model
            esg_score_preds = self.layer2_model.predict(pillar_disclosure_preds)

        # Return the predictions
        return {
            'pillar_disclosure_scores': pillar_disclosure_preds,
            'esg_score': esg_score_preds
        }

    def evaluate(self, X_test, y_test):
        """
        Evaluate the two-layer model on test data.

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

        # Evaluate pillar and disclosure score predictions (Layer 1)
        for i, col in enumerate(self.pillar_disclosure_columns):
            true_values = y_test[col].values
            pred_values = predictions['pillar_disclosure_scores'][:, i]

            mse = mean_squared_error(true_values, pred_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_values, pred_values)
            r2 = r2_score(true_values, pred_values)
            adjusted_r2 = 1 - (1 - r2) * (len(true_values) - 1) / (len(true_values) - 1 - 1)  # 1 feature for each target

            metrics_data.append({
                'Target': col,
                'Layer': 'Layer 1',
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R² Score': r2,
                'Adjusted R² Score': adjusted_r2
            })

        # Evaluate final ESG score prediction (Layer 2)
        true_values = y_test[self.final_target].values
        pred_values = predictions['esg_score']

        mse = mean_squared_error(true_values, pred_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, pred_values)
        r2 = r2_score(true_values, pred_values)
        adjusted_r2 = 1 - (1 - r2) * (len(true_values) - 1) / (len(true_values) - len(self.pillar_disclosure_columns) - 1)

        metrics_data.append({
            'Target': self.final_target,
            'Layer': 'Layer 2',
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R² Score': r2,
            'Adjusted R² Score': adjusted_r2
        })

        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics_data)

        return metrics_df

    def get_feature_importances(self):
        """
        Get feature importances for each target in Layer 1.

        Returns:
        --------
        dict: Dictionary mapping target names to DataFrames of feature importances
        """
        importances = {}

        for target, model in self.layer1_models.items():
            if hasattr(model, 'feature_importances_'):
                feature_names = self.preprocessor.get_feature_names_out()
                imp = model.feature_importances_

                # Create DataFrame
                df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': imp
                }).sort_values(by='Importance', ascending=False)

                importances[target] = df

        return importances

def plot_actual_vs_predicted_two_layer(model, X_test, y_test):
    """
    Plot actual vs predicted values for all targets in the two-layer model.

    Parameters:
    -----------
    model : TwoLayerModel
        The trained two-layer model
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
    # Dynamically adjust the number of rows and columns for the subplot grid
    cols = min(3, n_outputs)  # Limit the number of columns to a maximum of 3
    rows = (n_outputs + cols - 1) // cols  # Ceiling division to determine number of rows
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)  # Ensure axes is always a 1D array

    # Plot for pillar and disclosure scores (Layer 1)
    for i, col in enumerate(model.pillar_disclosure_columns):
        # Scatter plot of actual vs predicted for this specific output
        axes[i].scatter(y_test[col], predictions['pillar_disclosure_scores'][:, i], alpha=0.7)

        # Add perfect prediction line
        min_val = min(y_test[col].min(), predictions['pillar_disclosure_scores'][:, i].min())
        max_val = max(y_test[col].max(), predictions['pillar_disclosure_scores'][:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')

        # Calculate R² for this prediction
        r2 = r2_score(y_test[col], predictions['pillar_disclosure_scores'][:, i])

        # Set labels and title
        axes[i].set_xlabel(f'Actual {col}')
        axes[i].set_ylabel(f'Predicted {col}')
        axes[i].set_title(f'Layer 1: {col}\nR² = {r2:.4f}')
        axes[i].grid(True, linestyle='--', alpha=0.7)

    # Plot for final ESG score (Layer 2)
    i = len(model.pillar_disclosure_columns)
    axes[i].scatter(y_test[model.final_target], predictions['esg_score'], alpha=0.7)

    # Add perfect prediction line
    min_val = min(y_test[model.final_target].min(), predictions['esg_score'].min())
    max_val = max(y_test[model.final_target].max(), predictions['esg_score'].max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')

    # Calculate R² for the final prediction
    r2 = r2_score(y_test[model.final_target], predictions['esg_score'])

    # Set labels and title
    axes[i].set_xlabel(f'Actual {model.final_target}')
    axes[i].set_ylabel(f'Predicted {model.final_target}')
    axes[i].set_title(f'Layer 2: {model.final_target}\nR² = {r2:.4f}')
    axes[i].grid(True, linestyle='--', alpha=0.7)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Add a title to the figure
    plt.suptitle('Actual vs Predicted Values for Two-Layer Model', fontsize=16, y=1.05)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_feature_importances(model, top_n=15, figsize=(15, 18), color_palette='viridis'):
    """
    Plot feature importances for each target in Layer 1 with enhanced styling.

    Parameters:
    -----------
    model : TwoLayerModel
        The trained two-layer model
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size (width, height)
    color_palette : str
        Color palette to use for bars
    """

    # Get feature importances for all targets
    importances = model.get_feature_importances()
    n_targets = len(importances)

    if n_targets == 0:
        print("No feature importances available to plot.")
        return

    # Use a vertical layout (one subplot per row) for better readability
    fig, axes = plt.subplots(n_targets, 1, figsize=figsize)

    # If there's only one target, make axes iterable
    if n_targets == 1:
        axes = [axes]

    for i, (target, imp_df) in enumerate(importances.items()):
        # Get top N features
        top_features = imp_df.head(top_n)

        # Use a custom color map for each target for better distinction
        # You can choose different palettes: viridis, plasma, mako, rocket, etc.
        custom_colors = sns.color_palette(color_palette, n_colors=len(top_features))

        # Plot horizontal bar chart with enhanced styling
        bars = sns.barplot(
            x='Importance',
            y='Feature',
            data=top_features,
            ax=axes[i],
            palette=custom_colors,
            edgecolor='white',
            linewidth=0.8
        )

        # Add value labels to the right of each bar
        for j, v in enumerate(top_features['Importance']):
            axes[i].text(
                v + 0.001,  # Small offset from the end of the bar
                j,
                f"{v:.3f}",
                va='center',
                fontsize=10,
                fontweight='bold'
            )

        # Customize grid for better readability
        axes[i].grid(True, axis='x', linestyle='--', alpha=0.7)

        # Add a title for each subplot with the target name
        target_name = target.replace('BESG ', '').replace(' Pillar Score', '')
        axes[i].set_title(
            f"{target_name} Pillar: Top {top_n} Important Features",
            fontsize=16,
            fontweight='bold',
            pad=20
        )

        # Customize labels
        axes[i].set_xlabel('Feature Importance', fontsize=14)
        axes[i].set_ylabel('Feature Name', fontsize=14)

        # Improve tick label sizes
        axes[i].tick_params(axis='both', labelsize=12)

        # Remove spines for cleaner look
        for spine in ['right', 'top']:
            axes[i].spines[spine].set_visible(False)

    # Add an overall title
    plt.suptitle(
        'Feature Importance Analysis for ESG Pillars',
        fontsize=20,
        fontweight='bold',
        y=1.02
    )

    # Add a subtitle with explanation
    plt.figtext(
        0.5,
        0.99,
        "Higher values indicate greater impact on model predictions",
        ha='center',
        fontsize=14,
        fontstyle='italic'
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)  # Add more space between subplots
    plt.show()