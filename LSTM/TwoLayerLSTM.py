import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt

class TwoLayerESGModel(nn.Module):
    """
    Two-layer model for ESG score prediction:
    1. LSTM layer to predict pillar scores and disclosure scores
    2. Linear layer to predict final ESG score from pillar and disclosure scores
    """
    def __init__(self, input_size, hidden_size, num_layers, component_size=7, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Component scores: 3 pillar scores + 4 disclosure scores = 7 components
        self.component_size = component_size

        # Bidirectional LSTM layer to predict the component scores
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism for LSTM
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Tanh(),
            nn.Softmax(dim=1)
        )

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # Component prediction layer (predicts all 7 component scores)
        self.component_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.component_size)
        )

        # Final ESG score predictor (simple linear regression from the 7 components)
        self.esg_predictor = nn.Linear(self.component_size, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

        for m in self.component_predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

        # Initialize the ESG predictor with weights that sum to approximately 1
        # This represents a reasonable starting point since ESG score is weighted average
        nn.init.constant_(self.esg_predictor.weight, 1.0 / self.component_size)
        nn.init.constant_(self.esg_predictor.bias, 0.0)

    def forward(self, x):
        """
        Forward pass:
        1. Process input features through LSTM with attention
        2. Predict component scores
        3. Use component scores to predict final ESG score

        Returns:
        - all_scores: tensor containing component scores and final ESG score
        - component_scores: tensor containing only component scores
        - esg_score: tensor containing only the final ESG score
        """
        batch_size = x.size(0)

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Apply attention
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # Apply batch normalization
        normed_context = self.batch_norm(context_vector)

        # Predict component scores (pillar scores + disclosure scores)
        component_scores = self.component_predictor(normed_context)

        # Predict final ESG score from component scores
        esg_score = self.esg_predictor(component_scores)

        # Combine all scores for convenient output
        # First column is the ESG score, followed by the 7 component scores
        all_scores = torch.cat([esg_score, component_scores], dim=1)

        return all_scores, component_scores, esg_score

class ESGTimeSeriesDataset(Dataset):
    """
    Dataset class for ESG time series data with component-based structure
    """
    def __init__(self, X, y, company_groups, seq_length=3):
        self.X = X
        self.y = y  # y contains all 8 scores (ESG + 7 components)
        self.company_groups = company_groups
        self.seq_length = seq_length

        # Create valid indices that respect company boundaries
        self.valid_indices = self._get_valid_indices()

    def _get_valid_indices(self):
        """Create valid starting indices that ensure sequences don't cross company boundaries"""
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

        # Get sequence of features
        X_seq = self.X[start_idx:start_idx+self.seq_length]

        # Get target values: ESG score is first column, component scores are the rest
        y_target = self.y[start_idx+self.seq_length-1]

        # Split targets into ESG score and component scores
        esg_score = y_target[0:1]  # ESG Score only
        component_scores = y_target[1:]  # 7 component scores

        return {
            'features': torch.tensor(X_seq, dtype=torch.float32),
            'esg_score': torch.tensor(esg_score, dtype=torch.float32),
            'component_scores': torch.tensor(component_scores, dtype=torch.float32),
            'all_scores': torch.tensor(y_target, dtype=torch.float32)
        }

def dynamic_esg_loss(predictions, targets, initial_component_weight=0.7,
                   threshold=30, min_component_weight=0.1, max_component_weight=0.8):
    """
    Dynamic loss function that:
    1. Initially prioritizes component score prediction
    2. Gradually shifts focus to ESG score as component predictions improve

    Args:
        predictions: tuple containing (all_scores, component_scores, esg_score)
        targets: tuple containing (all_scores, component_scores, esg_score)
        initial_component_weight: starting weight for component score loss (default: 0.7)
        threshold: component loss value below which weighting changes (default: 0.1)
        min_component_weight: minimum weight for component loss (default: 0.1)
        max_component_weight: maximum weight for component loss (default: 0.8)

    Returns:
        total_loss: combined weighted loss
        esg_loss: loss for ESG score prediction
        component_loss: loss for component score prediction
        component_weight: the actual weight used for this batch
    """
    # Unpack predictions and targets
    _, pred_components, pred_esg = predictions
    all_targets, target_components, target_esg = targets

    # Check dimensions and adjust if needed
    if pred_components.shape[1] != target_components.shape[1]:
        # If dimensions don't match, we need to adapt
        print(f"Warning: Component dimensions don't match. Pred: {pred_components.shape}, Target: {target_components.shape}")

        # Option 1: Use only the first n components where n is the minimum of the two sizes
        min_components = min(pred_components.shape[1], target_components.shape[1])
        pred_components = pred_components[:, :min_components]
        target_components = target_components[:, :min_components]

        print(f"Adjusted to use {min_components} components")

    # Calculate MSE loss for ESG score
    esg_loss = nn.MSELoss()(pred_esg, target_esg)

    # Calculate MSE loss for component scores
    component_loss = nn.MSELoss()(pred_components, target_components)

    # Determine component weight based on current component loss
    # When component_loss is high, weight is higher (max_component_weight)
    # When component_loss is below threshold, weight decreases toward min_component_weight
    if component_loss > threshold:
        # Above threshold: use initial weight (focus on components)
        component_weight = initial_component_weight
    else:
        # # Below threshold: gradually decrease weight as loss decreases
        # # Calculate a weight between min and max based on component loss
        # # Normalize between 0 and 1, then scale between min and max weights
        normalized_loss = min(component_loss / threshold, 1.0)  # Between 0 and 1
        component_weight = min_component_weight + normalized_loss * (max_component_weight - min_component_weight)

    # Combined loss with dynamic weighting
    total_loss = (1 - component_weight) * esg_loss + component_weight * component_loss

    return total_loss, esg_loss, component_loss, component_weight

def train_two_layer_model(df, target_cols, hidden_size=256, num_layers=2,
                         seq_length=3, learning_rate=0.001, train_size=0.8,
                         epochs=100, patience=15, batch_size=32, dropout=0.2,
                         initial_component_weight=0.7, threshold=0.1):
    """
    Train the two-layer ESG prediction model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Sort data by company and year
    data_sorted = df.sort_values(['Company', 'Year'])

    # Separate features and targets
    # First column is the main ESG score, followed by the 7 component scores
    esg_score_col = target_cols[0]  # 'BESG ESG Score'
    component_cols = target_cols[1:]  # Pillar scores and Disclosure scores

    # Extract features (everything that's not a target or identifier)
    cols_to_drop = ['Company', 'Year'] + target_cols
    X = data_sorted.drop(cols_to_drop, axis=1)

    # Extract targets in specific order: ESG score first, then component scores
    y = data_sorted[[esg_score_col] + component_cols].values

    # Get company groups for splitting
    company_groups = data_sorted['Company'].values
    unique_companies = np.unique(company_groups)

    # Get numerical features
    num_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Scale numerical features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X[num_features]), columns=num_features)

    # Ensure all data is numeric and handle NaNs
    X_scaled = X_scaled.fillna(0).astype(np.float32)

    # Split companies into train and test sets
    n_companies = len(unique_companies)
    n_train = int(n_companies * train_size)

    # Shuffle companies
    np.random.seed(42)
    np.random.shuffle(unique_companies)

    train_companies = unique_companies[:n_train]
    test_companies = unique_companies[n_train:]

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

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    input_size = X_scaled.shape[1]
    component_size = len(component_cols)  # Get actual number of components from data

    model = TwoLayerESGModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        component_size=component_size,  # Pass the correct component size
        dropout=dropout
    ).to(device)

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    # Define learning rate scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.3
    )

    # Train the model
    best_val_loss = float('inf')
    best_esg_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'esg_loss': [],
        'component_loss': [],
        'component_weight': []
    }

    print("Training model...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        esg_loss_total = 0.0
        component_loss_total = 0.0
        component_weight_total = 0.0

        for batch in train_loader:
            features = batch['features'].to(device)
            all_scores = batch['all_scores'].to(device)
            esg_score = batch['esg_score'].to(device)
            component_scores = batch['component_scores'].to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(features)

            # Compute loss with dynamic weighting
            loss, esg_loss, component_loss, component_weight = dynamic_esg_loss(
                predictions,
                (all_scores, component_scores, esg_score),
                initial_component_weight,
                threshold, 0.0, 0.1
            )

            # Backward and optimize
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            esg_loss_total += esg_loss.item()
            component_loss_total += component_loss.item()
            component_weight_total += component_weight

        train_loss /= len(train_loader)
        esg_loss_total /= len(train_loader)
        component_loss_total /= len(train_loader)
        component_weight_total /= len(train_loader)

        history['train_loss'].append(train_loss)
        history['esg_loss'].append(esg_loss_total)
        history['component_loss'].append(component_loss_total)
        history['component_weight'].append(component_weight_total)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(device)
                all_scores = batch['all_scores'].to(device)
                esg_score = batch['esg_score'].to(device)
                component_scores = batch['component_scores'].to(device)

                # Forward pass
                predictions = model(features)

                # Compute loss
                loss, _, _, _ = dynamic_esg_loss(
                    predictions,
                    (all_scores, component_scores, esg_score),
                    initial_component_weight,
                    threshold, 0.0, 1.0
                )

                val_loss += loss.item()

        val_loss /= len(test_loader)
        history['val_loss'].append(val_loss)

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"ESG Loss: {esg_loss_total:.4f} | Component Loss: {component_loss_total:.4f} | "
                  f"Component Weight: {component_weight_total:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_two_layer_model.pt')
        elif esg_loss < best_esg_loss:
            best_esg_loss = esg_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_two_layer_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('best_two_layer_model.pt'))

    return model, train_loader, test_loader, test_dataset, X_scaled.columns.tolist(), history, device, component_cols

def evaluate_two_layer_model(model, test_loader, device, component_cols, esg_col='BESG ESG Score'):
    """
    Evaluate the two-layer model and return detailed metrics
    """
    model.to(device)
    model.eval()

    # Initialize lists to store predictions and targets
    esg_preds = []
    esg_targets = []
    component_preds = []
    component_targets = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            all_scores = batch['all_scores'].cpu().numpy()
            esg_score = batch['esg_score'].cpu().numpy()
            component_scores = batch['component_scores'].cpu().numpy()

            # Forward pass
            all_pred, comp_pred, esg_pred = model(features)

            # Store predictions and targets
            esg_preds.append(esg_pred.cpu().numpy())
            esg_targets.append(esg_score)
            component_preds.append(comp_pred.cpu().numpy())
            component_targets.append(component_scores)

    # Concatenate results
    esg_preds = np.concatenate(esg_preds)
    esg_targets = np.concatenate(esg_targets)
    component_preds = np.concatenate(component_preds)
    component_targets = np.concatenate(component_targets)

    # Calculate metrics for ESG score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    esg_mse = mean_squared_error(esg_targets, esg_preds)
    esg_rmse = np.sqrt(esg_mse)
    esg_mae = mean_absolute_error(esg_targets, esg_preds)
    esg_r2 = r2_score(esg_targets, esg_preds)

    # Calculate metrics for each component
    component_metrics = []
    for i, component in enumerate(component_cols):
        mse = mean_squared_error(component_targets[:, i], component_preds[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(component_targets[:, i], component_preds[:, i])
        r2 = r2_score(component_targets[:, i], component_preds[:, i])

        component_metrics.append({
            'component': component,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })

    # Calculate average component metrics
    avg_component_mse = np.mean([m['mse'] for m in component_metrics])
    avg_component_rmse = np.mean([m['rmse'] for m in component_metrics])
    avg_component_mae = np.mean([m['mae'] for m in component_metrics])
    avg_component_r2 = np.mean([m['r2'] for m in component_metrics])

    # Create result dictionary
    results = {
        'esg': {
            'predictions': esg_preds,
            'targets': esg_targets,
            'mse': esg_mse,
            'rmse': esg_rmse,
            'mae': esg_mae,
            'r2': esg_r2
        },
        'components': {
            'predictions': component_preds,
            'targets': component_targets,
            'metrics': component_metrics,
            'avg_mse': avg_component_mse,
            'avg_rmse': avg_component_rmse,
            'avg_mae': avg_component_mae,
            'avg_r2': avg_component_r2
        }
    }

    return results

def visualize_esg_results(results, component_cols, model_name="Two-Layer ESG Model"):
    """
    Visualize the results of the two-layer ESG model
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set style
    sns.set_style('whitegrid')

    # ESG Score Predictions vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(results['esg']['targets'], results['esg']['predictions'], alpha=0.6)

    # Add perfect prediction line
    min_val = min(results['esg']['targets'].min(), results['esg']['predictions'].min())
    max_val = max(results['esg']['targets'].max(), results['esg']['predictions'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    # Add correlation coefficient
    corr = np.corrcoef(results['esg']['targets'].flatten(), results['esg']['predictions'].flatten())[0, 1]
    plt.text(0.05, 0.95, f'r = {corr:.4f}', transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=12)

    plt.xlabel('Actual ESG Score')
    plt.ylabel('Predicted ESG Score')
    plt.title(f'{model_name}: ESG Score Predictions')
    plt.tight_layout()
    plt.savefig('esg_predictions.png', dpi=300)
    plt.show()

    # Component Score Predictions
    plt.figure(figsize=(15, 10))

    # Create subplots for each component
    n_components = len(component_cols)
    rows = (n_components + 1) // 2

    for i, component in enumerate(component_cols):
        plt.subplot(rows, 2, i+1)
        plt.scatter(
            results['components']['targets'][:, i],
            results['components']['predictions'][:, i],
            alpha=0.6
        )

        # Add perfect prediction line
        comp_min = min(
            results['components']['targets'][:, i].min(),
            results['components']['predictions'][:, i].min()
        )
        comp_max = max(
            results['components']['targets'][:, i].max(),
            results['components']['predictions'][:, i].max()
        )
        plt.plot([comp_min, comp_max], [comp_min, comp_max], 'r--')

        # Add correlation coefficient
        comp_corr = np.corrcoef(
            results['components']['targets'][:, i],
            results['components']['predictions'][:, i]
        )[0, 1]
        plt.text(0.05, 0.95, f'r = {comp_corr:.4f}', transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10)

        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.title(f'{component}')

    plt.tight_layout()
    plt.savefig('component_predictions.png', dpi=300)
    plt.show()

    # Metrics comparison table
    component_metrics = [m for m in results['components']['metrics']]
    metrics_df = pd.DataFrame(component_metrics)
    metrics_df = metrics_df.set_index('component')

    # Add ESG score metrics
    metrics_df.loc['ESG Score'] = [
        results['esg']['mse'],
        results['esg']['rmse'],
        results['esg']['mae'],
        results['esg']['r2']
    ]

    # Add average component metrics
    metrics_df.loc['Avg Component'] = [
        results['components']['avg_mse'],
        results['components']['avg_rmse'],
        results['components']['avg_mae'],
        results['components']['avg_r2']
    ]

    print("\nModel Performance Metrics:")
    display(metrics_df.round(6))

    # Plot metrics comparison
    plt.figure(figsize=(12, 6))

    # Plotting R² values
    plt.subplot(1, 2, 1)
    bars = plt.bar(metrics_df.index, metrics_df['r2'], color=sns.color_palette("viridis", len(metrics_df)))
    plt.ylabel('R² Score')
    plt.title('R² by Component')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)

    # Plotting RMSE values
    plt.subplot(1, 2, 2)
    bars = plt.bar(metrics_df.index, metrics_df['rmse'], color=sns.color_palette("viridis", len(metrics_df)))
    plt.ylabel('RMSE')
    plt.title('RMSE by Component')
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300)
    plt.show()

    # Analyze the ESG Predictor Weights
    return metrics_df

def analyze_esg_predictor(model):
    """
    Analyze the weights of the ESG predictor (second layer)
    """
    # Get the weights and bias of the ESG predictor
    weights = model.esg_predictor.weight.data.cpu().numpy()
    bias = model.esg_predictor.bias.data.cpu().numpy()

    return weights, bias

def plot_dynamic_weight_change(history):
    """
    Plot how the component weight changes throughout training
    """
    # Convert tensors to numpy for plotting, properly detaching if needed
    component_weights = []
    for w in history['component_weight']:
        if isinstance(w, torch.Tensor):
            # Make sure to detach tensors that might have gradients
            component_weights.append(w.detach().cpu().numpy())
        else:
            component_weights.append(w)

    # Also ensure component_loss values are properly detached
    component_losses = []
    for loss in history['component_loss']:
        if isinstance(loss, torch.Tensor):
            component_losses.append(loss.detach().cpu().numpy())
        else:
            component_losses.append(loss)

    component_losses = np.array(component_losses)
    component_weights = np.array(component_weights)

    plt.figure(figsize=(12, 6))

    # Plot component weight over epochs
    plt.subplot(1, 2, 1)
    plt.plot(component_weights, 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Component Weight')
    plt.title('Component Weight During Training')
    plt.grid(True)

    # Plot component loss vs weight
    plt.subplot(1, 2, 2)
    plt.scatter(component_losses, component_weights,
                c=range(len(component_losses)), cmap='viridis', alpha=0.7)
    plt.colorbar(label='Epoch')
    plt.xlabel('Component Loss')
    plt.ylabel('Component Weight')
    plt.title('Component Weight vs. Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
