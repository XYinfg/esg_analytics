import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ESGDataset(Dataset):
    """
    Dataset class for ESG data
    """
    def __init__(self, X_cat, X_num, y=None, is_train=True):
        self.X_cat = X_cat
        self.X_num = X_num
        self.y = y
        self.is_train = is_train
    
    def __len__(self):
        return len(self.X_cat)
    
    def __getitem__(self, idx):
        if self.is_train and self.y is not None:
            return {
                'cat_features': torch.tensor(self.X_cat[idx], dtype=torch.long),
                'num_features': torch.tensor(self.X_num[idx], dtype=torch.float32),
                'target': torch.tensor(self.y[idx], dtype=torch.float32)
            }
        else:
            return {
                'cat_features': torch.tensor(self.X_cat[idx], dtype=torch.long),
                'num_features': torch.tensor(self.X_num[idx], dtype=torch.float32)
            }

class TabularEmbedding(nn.Module):
    """
    Embedding layer for categorical features
    """
    def __init__(self, cat_dims, embed_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dims[i], embed_dim) for i in range(len(cat_dims))
        ])
        
    def forward(self, x):
        # x shape: (batch_size, num_cat_features)
        embeddings = [self.embeddings[i](x[:, i]) for i in range(x.shape[1])]
        # Concatenate embeddings along feature dimension
        embeddings = torch.stack(embeddings, dim=1)  # (batch_size, num_cat_features, embed_dim)
        return embeddings

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for TabTransformer
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        
        # Split embeddings into multiple heads
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))
        
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Compute output
        out = torch.matmul(attention, v)  # (batch_size, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
        out = out.reshape(batch_size, seq_len, embed_dim)
        out = self.fc_out(out)
        
        return out, attention

class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and feed-forward network
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Multi-head attention with residual connection
        attn_out, attention_weights = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward network with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x, attention_weights

class TabTransformer(nn.Module):
    """
    TabTransformer model for tabular data
    """
    def __init__(self, cat_dims, num_features, embed_dim, num_heads, num_blocks, ff_dim, output_dim=1, dropout=0.1):
        super().__init__()
        self.cat_embedding = TabularEmbedding(cat_dims, embed_dim)
        self.num_features = num_features
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_blocks)
        ])
        
        # Numerical features processing
        self.num_proj = nn.Linear(num_features, embed_dim) if num_features > 0 else None
        
        # Output layer
        combined_dim = embed_dim * len(cat_dims) + (embed_dim if num_features > 0 else 0)
        self.output_layer = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, output_dim)
        )
        
    def forward(self, cat_features, num_features=None):
        # Process categorical features
        cat_embeddings = self.cat_embedding(cat_features)  # (batch_size, num_cat_features, embed_dim)
        
        # Apply transformer blocks to embeddings
        attention_weights_list = []
        for transformer in self.transformer_blocks:
            cat_embeddings, attention_weights = transformer(cat_embeddings)
            attention_weights_list.append(attention_weights)
        
        # Flatten categorical embeddings
        batch_size = cat_features.shape[0]
        cat_embeddings = cat_embeddings.reshape(batch_size, -1)  # (batch_size, num_cat_features * embed_dim)
        
        # Process numerical features if available
        if num_features is not None and self.num_proj is not None:
            num_embeddings = self.num_proj(num_features)  # (batch_size, embed_dim)
            # Combine categorical and numerical features
            combined = torch.cat([cat_embeddings, num_embeddings], dim=1)
        else:
            combined = cat_embeddings
        
        # Output layer
        output = self.output_layer(combined)
        
        return output, attention_weights_list

class ESGPreprocessor:
    """
    Preprocessor for ESG data
    """
    def __init__(self, cat_features=None, num_features=None):
        self.cat_features = cat_features or []
        self.num_features = num_features or []
        self.cat_encoders = {}
        self.num_scaler = None
        self.cat_dims = []
        
    def fit(self, df):
        # Encode categorical features
        for col in self.cat_features:
            # Create a mapping for each categorical value to an integer
            unique_values = df[col].dropna().unique()
            self.cat_encoders[col] = {val: i for i, val in enumerate(unique_values)}
            # Store dimensions for embeddings
            self.cat_dims.append(len(unique_values) + 1)  # +1 for missing values
        
        # Scale numerical features if needed
        from sklearn.preprocessing import StandardScaler
        if self.num_features:
            self.num_scaler = StandardScaler()
            self.num_scaler.fit(df[self.num_features].fillna(0))
        
        return self
    
    def transform(self, df):
        # Transform categorical features
        X_cat = np.zeros((len(df), len(self.cat_features)), dtype=int)
        for i, col in enumerate(self.cat_features):
            # Handle values not seen during training
            X_cat[:, i] = df[col].map(lambda x: self.cat_encoders[col].get(x, 0)).fillna(0).values
        
        # Transform numerical features
        if self.num_features:
            X_num = self.num_scaler.transform(df[self.num_features].fillna(0))
        else:
            X_num = np.zeros((len(df), 0))
        
        return X_cat, X_num
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, patience=3):
    """
    Train the TabTransformer model
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            cat_features = batch['cat_features'].to(device)
            num_features = batch['num_features'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(cat_features, num_features)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                cat_features = batch['cat_features'].to(device)
                num_features = batch['num_features'].to(device)
                targets = batch['target'].to(device)
                
                outputs, _ = model(cat_features, num_features)
                loss = criterion(outputs.squeeze(), targets)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model, history

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on test data
    """
    model.to(device)
    model.eval()
    test_loss = 0.0
    predictions = []
    targets_list = []
    attention_weights = []
    
    with torch.no_grad():
        for batch in test_loader:
            cat_features = batch['cat_features'].to(device)
            num_features = batch['num_features'].to(device)
            targets = batch['target'].to(device)
            
            outputs, batch_attention_weights = model(cat_features, num_features)
            loss = criterion(outputs.squeeze(), targets)
            
            test_loss += loss.item()
            predictions.extend(outputs.squeeze().cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
            
            # Store attention weights
            for i, attn in enumerate(batch_attention_weights):
                if len(attention_weights) <= i:
                    attention_weights.append([])
                attention_weights[i].append(attn.cpu().numpy())
    
    test_loss /= len(test_loader)
    predictions = np.array(predictions)
    targets_list = np.array(targets_list)
    
    # Aggregate attention weights
    for i in range(len(attention_weights)):
        attention_weights[i] = np.concatenate(attention_weights[i], axis=0)
    
    return test_loss, predictions, targets_list, attention_weights

def get_feature_importance(model, cat_features, attention_weights, layer_idx=0, head_idx=0):
    """
    Get feature importance based on attention weights
    """
    # Take attention weights from a specific layer and head
    attn = attention_weights[layer_idx][:, head_idx, :, :]
    
    # Average attention weights across all samples
    avg_attn = attn.mean(axis=0)
    
    # Get diagonal values (self-attention)
    feature_importance = avg_attn.diagonal()
    
    # Create a dictionary with feature names and importance scores
    importance_dict = {cat_features[i]: float(feature_importance[i]) for i in range(len(cat_features))}
    
    return importance_dict