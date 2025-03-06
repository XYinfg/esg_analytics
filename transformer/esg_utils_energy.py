"""
Utility functions for using the trained ESG TabTransformer model for energy companies
"""

import torch
import pandas as pd
import numpy as np
import joblib
from tab_transformer import TabTransformer
from data_preprocessing import load_and_clean_data, prepare_data_for_modeling

def load_model(model_path='energy_esg_tabtransformer.pt', preprocessor_path='energy_esg_preprocessor.pkl'):
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

def preprocess_new_data(data, cat_features=None, num_features=None):
    """
    Preprocess new data for prediction
    
    Parameters:
    data (pd.DataFrame): Raw data
    cat_features (list): List of categorical features (if None, will be identified)
    num_features (list): List of numerical features (if None, will be identified)
    
    Returns:
    processed_data (pd.DataFrame): Processed data
    cat_features (list): Categorical features
    num_features (list): Numerical features
    """
    # If data is a CSV file path, load it
    if isinstance(data, str):
        data = load_and_clean_data(data)
    else:
        # Apply the same preprocessing as in load_and_clean_data
        from data_preprocessing import load_and_clean_data
        # We can't use the function directly since we already have the dataframe
        # So we'll extract and apply the preprocessing steps
        
        # Convert known numeric columns from strings to float
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Convert binary columns
        binary_cols = [col for col in data.columns if 'Policy' in col or 'Discussed' in col]
        for col in binary_cols:
            if col in data.columns:
                data[col] = data[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).fillna(0)
        
        # Fill missing values
        for col in data.select_dtypes(include=['float64', 'int64']).columns:
            data[col] = data[col].fillna(data[col].median())
        
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown')
    
    # Prepare data for modeling
    processed_data, identified_cat_features, identified_num_features = prepare_data_for_modeling(
        data, cat_features, num_features
    )
    
    # If features weren't provided, use the identified ones
    if cat_features is None:
        cat_features = identified_cat_features
    if num_features is None:
        num_features = identified_num_features
    
    return processed_data, cat_features, num_features

def predict_esg_score(data, model=None, preprocessor=None):
    """
    Predict ESG scores for new data
    
    Parameters:
    data (pd.DataFrame or str): Dataframe or path to CSV file
    model (TabTransformer): The loaded model (if None, will be loaded)
    preprocessor (ESGPreprocessor): The loaded preprocessor (if None, will be loaded)
    
    Returns:
    predictions (np.array): Predicted ESG scores
    """
    # Load model and preprocessor if not provided
    if model is None or preprocessor is None:
        model, preprocessor = load_model()
    
    # Preprocess the data
    processed_data, cat_features, num_features = preprocess_new_data(
        data, preprocessor.cat_features, preprocessor.num_features
    )
    
    # Ensure the data has the required columns
    missing_cat_cols = [col for col in preprocessor.cat_features if col not in processed_data.columns]
    missing_num_cols = [col for col in preprocessor.num_features if col not in processed_data.columns]
    
    if missing_cat_cols or missing_num_cols:
        print(f"Warning: Data is missing the following columns:")
        if missing_cat_cols:
            print(f"Categorical: {missing_cat_cols}")
        if missing_num_cols:
            print(f"Numerical: {missing_num_cols}")
        print("These features will be treated as missing values during prediction.")
    
    # Extract the data for prediction
    X_cat, X_num = preprocessor.transform(processed_data[preprocessor.cat_features + preprocessor.num_features])
    
    # Convert to PyTorch tensors
    X_cat_tensor = torch.tensor(X_cat, dtype=torch.long)
    X_num_tensor = torch.tensor(X_num, dtype=torch.float32)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs, _ = model(X_cat_tensor, X_num_tensor)
        predictions = outputs.squeeze().numpy()
    
    return predictions

def get_feature_importances(model=None, preprocessor=None):
    """
    Get feature importances from a trained model using a small sample dataset
    
    Parameters:
    model (TabTransformer): The loaded model (if None, will be loaded)
    preprocessor (ESGPreprocessor): The loaded preprocessor (if None, will be loaded)
    
    Returns:
    importance_dict (dict): Dictionary mapping feature names to importance scores
    """
    # Load model and preprocessor if not provided
    if model is None or preprocessor is None:
        model, preprocessor = load_model()
    
    # Create a small sample dataset for attention analysis
    sample_size = 10
    sample_data = {}
    
    # Create sample categorical features
    for col in preprocessor.cat_features:
        # For each categorical feature, use the first category
        sample_data[col] = ['Unknown'] * sample_size
    
    # Create sample numerical features
    for col in preprocessor.num_features:
        sample_data[col] = [0.0] * sample_size
    
    sample_df = pd.DataFrame(sample_data)
    
    # Process the data
    X_cat, X_num = preprocessor.transform(sample_df)
    
    # Convert to PyTorch tensors
    X_cat_tensor = torch.tensor(X_cat, dtype=torch.long)
    X_num_tensor = torch.tensor(X_num, dtype=torch.float32)
    
    # Get attention weights
    model.eval()
    with torch.no_grad():
        _, attention_weights = model(X_cat_tensor, X_num_tensor)
    
    # Extract feature importances from attention weights
    importances = {}
    
    # For each layer and head, get feature importances
    for layer_idx in range(len(attention_weights)):
        for head_idx in range(attention_weights[layer_idx].shape[1]):
            # Extract attention weights for this layer and head
            attn = attention_weights[layer_idx][:, head_idx, :, :]
            
            # Average across all samples
            avg_attn = attn.mean(axis=0)
            
            # Get diagonal values (self-attention)
            feature_importance = avg_attn.diagonal()
            
            # Create a dictionary for this layer and head
            layer_head_key = f"layer_{layer_idx}_head_{head_idx}"
            importances[layer_head_key] = {
                preprocessor.cat_features[i]: float(feature_importance[i]) 
                for i in range(len(preprocessor.cat_features))
            }
    
    return importances
