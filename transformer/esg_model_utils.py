"""
Utility functions for loading and using the trained ESG TabTransformer model
"""

import torch
import pandas as pd
import numpy as np
import joblib
from tab_transformer import TabTransformer

def load_model(model_path='esg_tab_transformer.pt', preprocessor_path='esg_preprocessor.pkl'):
    """
    Load the trained TabTransformer model and preprocessor
    
    Parameters:
    model_path (str): Path to the saved model
    preprocessor_path (str): Path to the saved preprocessor
    
    Returns:
    model (TabTransformer): The loaded model
    preprocessor (ESGPreprocessor): The loaded preprocessor
    """
    # Load the preprocessor
    preprocessor = joblib.load(preprocessor_path)
    
    # Load the model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Recreate the model with the same architecture
    model = TabTransformer(
        cat_dims=checkpoint['cat_dims'],
        num_features=checkpoint['num_features'],
        embed_dim=checkpoint['embed_dim'],
        num_heads=checkpoint['num_heads'],
        num_blocks=checkpoint['num_blocks'],
        ff_dim=checkpoint['ff_dim'],
        output_dim=1,
        dropout=checkpoint['dropout']
    )
    
    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, preprocessor

def predict_esg_score(data, model, preprocessor):
    """
    Predict ESG scores for new data
    
    Parameters:
    data (pd.DataFrame): Dataframe with the same columns as the training data
    model (TabTransformer): The loaded model
    preprocessor (ESGPreprocessor): The loaded preprocessor
    
    Returns:
    predictions (np.array): Predicted ESG scores
    """
    # Ensure the data has the required columns
    cat_features = preprocessor.cat_features
    num_features = preprocessor.num_features
    
    # Check for missing columns
    missing_cols = [col for col in cat_features + num_features if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Data is missing the following columns: {missing_cols}")
    
    # Preprocess the data
    X_cat, X_num = preprocessor.transform(data[cat_features + num_features])
    
    # Convert to PyTorch tensors
    X_cat_tensor = torch.tensor(X_cat, dtype=torch.long)
    X_num_tensor = torch.tensor(X_num, dtype=torch.float32)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs, _ = model(X_cat_tensor, X_num_tensor)
        predictions = outputs.squeeze().numpy()
    
    return predictions

def get_attention_weights(data, model, preprocessor):
    """
    Get attention weights for new data
    
    Parameters:
    data (pd.DataFrame): Dataframe with the same columns as the training data
    model (TabTransformer): The loaded model
    preprocessor (ESGPreprocessor): The loaded preprocessor
    
    Returns:
    attention_weights (list): List of attention weights for each layer
    """
    # Ensure the data has the required columns
    cat_features = preprocessor.cat_features
    num_features = preprocessor.num_features
    
    # Check for missing columns
    missing_cols = [col for col in cat_features + num_features if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Data is missing the following columns: {missing_cols}")
    
    # Preprocess the data
    X_cat, X_num = preprocessor.transform(data[cat_features + num_features])
    
    # Convert to PyTorch tensors
    X_cat_tensor = torch.tensor(X_cat, dtype=torch.long)
    X_num_tensor = torch.tensor(X_num, dtype=torch.float32)
    
    # Get attention weights
    model.eval()
    attention_weights = []
    with torch.no_grad():
        _, attn_weights = model(X_cat_tensor, X_num_tensor)
        attention_weights = [attn.numpy() for attn in attn_weights]
    
    return attention_weights

def analyze_feature_importance(attention_weights, cat_features, layer_idx=0, head_idx=0):
    """
    Analyze feature importance based on attention weights
    
    Parameters:
    attention_weights (list): List of attention weights for each layer
    cat_features (list): List of categorical feature names
    layer_idx (int): Index of the layer to analyze
    head_idx (int): Index of the attention head to analyze
    
    Returns:
    importance_dict (dict): Dictionary mapping feature names to importance scores
    """
    # Take attention weights from a specific layer and head
    attn = attention_weights[layer_idx][:, head_idx, :, :]
    
    # Average attention weights across all samples
    avg_attn = attn.mean(axis=0)
    
    # Get diagonal values (self-attention)
    feature_importance = avg_attn.diagonal()
    
    # Create a dictionary with feature names and importance scores
    importance_dict = {cat_features[i]: float(feature_importance[i]) for i in range(len(cat_features))}
    
    # Sort by importance
    importance_dict = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
    
    return importance_dict
