import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class ESGTimeSeriesDataset(Dataset):
    """
    Dataset class for ESG time series data
    """
    def __init__(self, X, y, company_groups, seq_length=3):
        self.X = X
        self.y = y
        self.company_groups = company_groups
        self.seq_length = seq_length

        # Create valid indices that respect company boundaries
        self.valid_indices = self._get_valid_indices()

    def _get_valid_indices(self):
        """
        Create valid starting indices that ensure sequences don't cross company boundaries
        """
        valid_indices = []
        company_ids = np.unique(self.company_groups)

        for company_id in company_ids:
            # Get all indices for this company
            company_indices = np.where(self.company_groups == company_id)[0]

            # If we have enough data points for this company to form at least one sequence
            if len(company_indices) >= self.seq_length:
                # Add all valid starting indices for this company
                for i in range(len(company_indices) - self.seq_length + 1):
                    valid_indices.append(company_indices[i])

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the actual starting index from our valid indices
        start_idx = self.valid_indices[idx]

        # Get sequence of features and target
        X_seq = self.X[start_idx:start_idx+self.seq_length]
        y_target = self.y[start_idx+self.seq_length-1]

        return {
            'features': torch.tensor(X_seq, dtype=torch.float32),
            'target': torch.tensor(y_target, dtype=torch.float32)
        }

class BidirectionalLSTMWithAttention(nn.Module):
    """
    Bidirectional LSTM model with attention mechanism for ESG score prediction
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM layers with batch normalization
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),  # 2 for bidirectional
            nn.Tanh(),
            nn.Softmax(dim=1)
        )

        # Batch normalization before fully connected layer
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # Fully connected output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        batch_size = x.size(0)

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  # 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        # Forward propagate LSTM
        # lstm_out: (batch_size, seq_length, hidden_size*2)
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Apply attention
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_length, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size*2)

        # Apply batch normalization
        normed_context = self.batch_norm(context_vector)

        # Pass through fully connected layer
        out = self.fc(normed_context)  # (batch_size, output_size)

        return out

def prepare_data_with_cv(df, target_cols, n_splits=5, seq_length=3):
    """
    Prepare data with cross-validation folds
    """
    # Sort data by company and year
    data_sorted = df.sort_values(['Company', 'Year'])

    # Extract features and targets
    cols_to_drop = ['Company', 'Year'] + target_cols
    X = data_sorted.drop(cols_to_drop, axis=1)
    y = data_sorted[target_cols].values

    # Get company groups for splitting
    company_groups = data_sorted['Company'].values
    unique_companies = np.unique(company_groups)

    # Get numerical features
    num_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X[num_features]), columns=num_features)

    # Ensure all data is numeric and handle NaNs
    X_scaled = X_scaled.fillna(0).astype(np.float32)

    # Create cross-validation folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_splits = []

    for train_idx, test_idx in kf.split(unique_companies):
        # Get companies for train and test
        train_companies = unique_companies[train_idx]
        test_companies = unique_companies[test_idx]

        # Create masks for train and test
        train_mask = np.isin(company_groups, train_companies)
        test_mask = np.isin(company_groups, test_companies)

        # Split data
        X_train, y_train = X_scaled.values[train_mask], y[train_mask]
        X_test, y_test = X_scaled.values[test_mask], y[test_mask]

        # Get company groups for train and test
        train_company_groups = company_groups[train_mask]
        test_company_groups = company_groups[test_mask]

        # Create datasets
        train_dataset = ESGTimeSeriesDataset(
            X_train, y_train, train_company_groups, seq_length
        )
        test_dataset = ESGTimeSeriesDataset(
            X_test, y_test, test_company_groups, seq_length
        )

        cv_splits.append((train_dataset, test_dataset))

    return cv_splits, X_scaled.shape[1]

def train_model_with_cv(df, target_cols, hidden_sizes=[128, 256, 512],
                        num_layers_options=[2, 3, 4], sequence_lengths=[3, 5, 7],
                        learning_rates=[0.001, 0.005], n_splits=5, epochs=50,
                        patience=10, batch_size=32):
    """
    Train model with cross-validation and hyperparameter tuning
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    best_val_loss = float('inf')
    best_params = {}
    best_model = None
    history = {'train_loss': [], 'val_loss': []}

    # Try different hyperparameter combinations
    for hidden_size in hidden_sizes:
        for num_layers in num_layers_options:
            for seq_length in sequence_lengths:
                for lr in learning_rates:
                    print(f"\nTesting with hidden_size={hidden_size}, num_layers={num_layers}, "
                          f"seq_length={seq_length}, lr={lr}")

                    # Prepare data with current sequence length
                    cv_splits, input_size = prepare_data_with_cv(df, target_cols, n_splits, seq_length)
                    output_size = len(target_cols)

                    # Track performance across folds
                    fold_val_losses = []

                    for fold, (train_dataset, val_dataset) in enumerate(cv_splits):
                        print(f"  Fold {fold+1}/{n_splits}")

                        # Create data loaders
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size)

                        # Initialize model
                        model = BidirectionalLSTMWithAttention(
                            input_size, hidden_size, num_layers, output_size
                        ).to(device)

                        # Define loss function and optimizer
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.AdamW(
                            model.parameters(), lr=lr, weight_decay=1e-5
                        )

                        # Define learning rate scheduler
                        steps_per_epoch = len(train_loader)
                        scheduler = OneCycleLR(
                            optimizer, max_lr=lr,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            pct_start=0.3
                        )

                        # Train the model
                        best_fold_val_loss = float('inf')
                        patience_counter = 0

                        for epoch in range(epochs):
                            # Training
                            model.train()
                            train_loss = 0.0

                            for batch in train_loader:
                                features = batch['features'].to(device)
                                targets = batch['target'].to(device)

                                optimizer.zero_grad()
                                outputs = model(features)

                                loss = criterion(outputs, targets)
                                loss.backward()

                                # Gradient clipping
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                                optimizer.step()
                                scheduler.step()

                                train_loss += loss.item()

                            train_loss /= len(train_loader)

                            # Validation
                            model.eval()
                            val_loss = 0.0

                            with torch.no_grad():
                                for batch in val_loader:
                                    features = batch['features'].to(device)
                                    targets = batch['target'].to(device)

                                    outputs = model(features)
                                    loss = criterion(outputs, targets)
                                    val_loss += loss.item()

                            val_loss /= len(val_loader)

                            print(f"    Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                            # Early stopping
                            if val_loss < best_fold_val_loss:
                                best_fold_val_loss = val_loss
                                patience_counter = 0
                            else:
                                patience_counter += 1
                                if patience_counter >= patience:
                                    print(f"    Early stopping at epoch {epoch+1}")
                                    break

                        fold_val_losses.append(best_fold_val_loss)

                    # Calculate average validation loss across folds
                    avg_val_loss = sum(fold_val_losses) / len(fold_val_losses)
                    print(f"  Average validation loss: {avg_val_loss:.4f}")

                    # Update best model if better
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_params = {
                            'hidden_size': hidden_size,
                            'num_layers': num_layers,
                            'seq_length': seq_length,
                            'learning_rate': lr
                        }

    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Train final model with best parameters on all data (FIXED VERSION)
    print("\nTraining final model with best parameters...")

    # Get best parameters
    hidden_size = best_params['hidden_size']
    num_layers = best_params['num_layers']
    seq_length = best_params['seq_length']
    lr = best_params['learning_rate']

    # Sort data by company and year
    data_sorted = df.sort_values(['Company', 'Year'])

    # Extract features and targets
    cols_to_drop = ['Company', 'Year'] + target_cols
    X = data_sorted.drop(cols_to_drop, axis=1)
    y = data_sorted[target_cols].values

    # Get company groups for splitting
    company_groups = data_sorted['Company'].values
    unique_companies = np.unique(company_groups)

    # Split companies 80/20
    np.random.seed(42)
    np.random.shuffle(unique_companies)
    n_train = int(len(unique_companies) * 0.8)
    train_companies = unique_companies[:n_train]
    val_companies = unique_companies[n_train:]

    # Create masks
    train_mask = np.isin(company_groups, train_companies)
    val_mask = np.isin(company_groups, val_companies)

    # Get numerical features
    num_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X[num_features]), columns=num_features)
    X_scaled = X_scaled.fillna(0).astype(np.float32)

    # Split data
    X_train, y_train = X_scaled.values[train_mask], y[train_mask]
    X_val, y_val = X_scaled.values[val_mask], y[val_mask]

    # Get company groups
    train_company_groups = company_groups[train_mask]
    val_company_groups = company_groups[val_mask]

    # Create datasets
    train_dataset = ESGTimeSeriesDataset(X_train, y_train, train_company_groups, seq_length)
    val_dataset = ESGTimeSeriesDataset(X_val, y_val, val_company_groups, seq_length)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    input_size = X_scaled.shape[1]
    output_size = len(target_cols)

    best_model = BidirectionalLSTMWithAttention(
        input_size, hidden_size, num_layers, output_size
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        best_model.parameters(), lr=lr, weight_decay=1e-5
    )

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.3
    )

    # Train final model
    best_final_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        best_model.train()
        train_loss = 0.0

        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = best_model(features)

            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(best_model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation
        best_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = batch['target'].to(device)

                outputs = best_model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_final_val_loss:
            best_final_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(best_model.state_dict(), 'best_lstm_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    best_model.load_state_dict(torch.load('best_lstm_model.pt'))

    return best_model, criterion, train_loader, val_loader, val_dataset, X_scaled.columns.tolist(), history, device, best_params


def evaluate_lstm_model(model, test_loader, criterion, device, target_names=None):
    """
    Comprehensively evaluate the LSTM model on test data

    Parameters:
    -----------
    model : torch.nn.Module
        The trained LSTM model to evaluate
    test_loader : torch.utils.data.DataLoader
        DataLoader containing the test data
    criterion : torch.nn.Module
        Loss function
    device : torch.device
        Device to run the evaluation on
    target_names : list, optional
        Names of the target variables for multi-output models

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics and results
    """
    model.to(device)
    model.eval()

    test_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)

            outputs = model(features)

            # Handle potential dimension mismatch
            if outputs.shape != targets.shape:
                if len(outputs.shape) > len(targets.shape):
                    outputs = outputs.squeeze()
                else:
                    targets = targets.squeeze()

            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # Store predictions and targets
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Combine batches
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # For single target case, ensure proper dimensions
    if all_predictions.shape[1] == 1:
        all_predictions = all_predictions.flatten()
        all_targets = all_targets.flatten()

        # Calculate overall metrics
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)

        metrics = {
            'test_loss': test_loss / len(test_loader),
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

        result = {
            'metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets
        }

    else:
        # For multi-target case, calculate metrics for each target
        if target_names is None:
            target_names = [f"Target_{i}" for i in range(all_predictions.shape[1])]

        metrics = {
            'test_loss': test_loss / len(test_loader),
            'mse': {},
            'rmse': {},
            'mae': {},
            'r2': {}
        }

        for i, name in enumerate(target_names):
            y_true = all_targets[:, i]
            y_pred = all_predictions[:, i]

            metrics['mse'][name] = mean_squared_error(y_true, y_pred)
            metrics['rmse'][name] = np.sqrt(metrics['mse'][name])
            metrics['mae'][name] = mean_absolute_error(y_true, y_pred)
            metrics['r2'][name] = r2_score(y_true, y_pred)

        # Calculate average metrics
        metrics['mse']['average'] = np.mean(list(metrics['mse'].values()))
        metrics['rmse']['average'] = np.mean(list(metrics['rmse'].values()))
        metrics['mae']['average'] = np.mean(list(metrics['mae'].values()))
        metrics['r2']['average'] = np.mean(list(metrics['r2'].values()))

        result = {
            'metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'target_names': target_names
        }

    return result


def generate_evaluation_plots(eval_results, model_name="Improved LSTM", compare_models=None):
    """
    Generate comprehensive evaluation plots

    Parameters:
    -----------
    eval_results : dict
        Results from evaluate_improved_lstm_model
    model_name : str
        Name of the model for plot titles
    compare_models : dict, optional
        Dictionary of other models' metrics for comparison

    Returns:
    --------
    None
    """
    predictions = eval_results['predictions']
    targets = eval_results['targets']
    metrics = eval_results['metrics']

    # Set seaborn style
    sns.set(style="whitegrid")

    # Create figure for multiple plots
    fig = plt.figure(figsize=(20, 15))

    # 1. Predictions vs Actual
    if 'target_names' in eval_results:  # Multi-target case
        target_names = eval_results['target_names']

        # Create subplots for each target
        n_targets = len(target_names)
        rows = int(np.ceil(n_targets / 2))

        for i, name in enumerate(target_names):
            ax = fig.add_subplot(rows, 2, i+1)

            y_true = targets[:, i]
            y_pred = predictions[:, i]

            ax.scatter(y_true, y_pred, alpha=0.5)

            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')

            # Add correlation coefficient
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax.transAxes,
                    verticalalignment='top', fontsize=12)

            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{name}: Predicted vs Actual')

        plt.tight_layout()
        plt.savefig(f'{model_name}_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.show()

    else:  # Single target case
        plt.figure(figsize=(10, 8))
        plt.scatter(targets, predictions, alpha=0.5)

        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        # Add correlation coefficient
        corr = np.corrcoef(targets, predictions)[0, 1]
        plt.text(0.05, 0.95, f'r = {corr:.2f}', transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=12)

        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.title(f'{model_name}: Predicted vs Actual')
        plt.savefig(f'{model_name}_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 2. Error Distribution
    plt.figure(figsize=(12, 8))

    if 'target_names' in eval_results:  # Multi-target case
        for i, name in enumerate(target_names):
            errors = predictions[:, i] - targets[:, i]
            sns.kdeplot(errors, label=name, fill=True, alpha=0.3)
    else:  # Single target case
        errors = predictions - targets
        sns.histplot(errors, bins=30, kde=True, alpha=0.7)

    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'{model_name}: Distribution of Prediction Errors')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Residual Plot
    plt.figure(figsize=(12, 8))

    if 'target_names' in eval_results:  # Multi-target case
        for i, name in enumerate(target_names):
            errors = predictions[:, i] - targets[:, i]
            plt.scatter(targets[:, i], errors, alpha=0.5, label=name)
    else:  # Single target case
        errors = predictions - targets
        plt.scatter(targets, errors, alpha=0.5)

    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Value')
    plt.ylabel('Residual (Error)')
    plt.title(f'{model_name}: Residual Plot')
    if 'target_names' in eval_results:
        plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_residual_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Model Comparison (if provided)
    if compare_models is not None:
        # Add current model metrics
        if 'target_names' in eval_results:  # Multi-target case
            current_metrics = {
                'MSE': metrics['mse']['average'],
                'RMSE': metrics['rmse']['average'],
                'MAE': metrics['mae']['average'],
                'R²': metrics['r2']['average']
            }
        else:  # Single target case
            current_metrics = {
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2']
            }

        # Combine with comparison metrics
        all_models = {model_name: current_metrics}
        all_models.update(compare_models)

        # Create DataFrame for comparison
        models_df = pd.DataFrame(all_models).T

        # Display comparison table
        print("\nModel Performance Comparison:")
        print(models_df)

        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        metrics_list = ['MSE', 'RMSE', 'MAE', 'R²']
        colors = sns.color_palette("Set2", len(all_models))

        for i, metric in enumerate(metrics_list):
            ax = axes[i]

            if metric == 'R²':
                # For R², higher is better
                bars = ax.bar(all_models.keys(), [model_metrics[metric] for model_metrics in all_models.values()],
                              color=colors)
            else:
                # For error metrics, lower is better
                bars = ax.bar(all_models.keys(), [model_metrics[metric] for model_metrics in all_models.values()],
                              color=colors)

            ax.set_title(metric)
            ax.set_ylabel(metric)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8)

            # Adjust y-axis for better visualization
            if metric == 'R²':
                ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(f'{model_name}_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


def analyze_feature_importance(model, test_dataset, feature_names, device, n_samples=100, seq_length=3):
    """
    Analyze feature importance using permutation importance

    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    test_dataset : torch.utils.data.Dataset
        Test dataset
    feature_names : list
        List of feature names
    device : torch.device
        Device to run the analysis on
    n_samples : int
        Number of samples to use for permutation
    seq_length : int
        Sequence length used in the model

    Returns:
    --------
    dict
        Dictionary containing feature importance scores
    """
    model.to(device)
    model.eval()

    # Get a subset of test data
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Calculate baseline predictions
    baseline_preds = []
    baseline_targets = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx >= n_samples:
                break

            features = batch['features'].to(device)
            targets = batch['target'].to(device)

            outputs = model(features)

            baseline_preds.append(outputs.cpu().numpy())
            baseline_targets.append(targets.cpu().numpy())

    baseline_preds = np.vstack(baseline_preds)
    baseline_targets = np.vstack(baseline_targets)

    # Calculate baseline error
    baseline_mse = mean_squared_error(baseline_targets, baseline_preds)

    # Calculate importance for each feature
    importances = {}

    for feat_idx, feature_name in enumerate(feature_names):
        # Reset loader
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Permute the feature and calculate new predictions
        permuted_preds = []
        permuted_targets = []

        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                if idx >= n_samples:
                    break

                features = batch['features'].clone().to(device)
                targets = batch['target'].to(device)

                # Permute this feature across the batch
                for seq_idx in range(seq_length):
                    perm_indices = torch.randperm(features.size(0))
                    features[:, seq_idx, feat_idx] = features[perm_indices, seq_idx, feat_idx]

                outputs = model(features)

                permuted_preds.append(outputs.cpu().numpy())
                permuted_targets.append(targets.cpu().numpy())

        permuted_preds = np.vstack(permuted_preds)
        permuted_targets = np.vstack(permuted_targets)

        # Calculate permuted error
        permuted_mse = mean_squared_error(permuted_targets, permuted_preds)

        # Importance is the increase in error
        importances[feature_name] = permuted_mse - baseline_mse

    # Sort by importance
    importances = {k: v for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}

    return importances


def plot_feature_importance(importance_dict, top_n=20, title='Feature Importance'):
    """
    Plot feature importance

    Parameters:
    -----------
    importance_dict : dict
        Dictionary mapping feature names to importance scores
    top_n : int
        Number of top features to plot
    title : str
        Plot title
    """
    # Get top features
    top_features = list(importance_dict.keys())[:top_n]
    top_values = [importance_dict[f] for f in top_features]

    # Create plot
    plt.figure(figsize=(12, top_n * 0.4 + 2))
    colors = plt.cm.get_cmap('viridis')(np.linspace(0, 0.8, len(top_features)))
    bars = plt.barh(top_features, top_values, color=colors)

    # Add values to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.4f}', ha='left', va='center')

    plt.xlabel('Importance (Increase in MSE when feature is permuted)')
    plt.ylabel('Feature')
    plt.title(title)
    plt.gca().invert_yaxis()  # Display the most important at the top
    plt.tight_layout()
    plt.grid(axis='x')

    plt.savefig(f'{title.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()