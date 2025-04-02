# Two-Tiered ESG Model

## Overview
This directory contains the implementation of a novel two-tiered approach to ESG (Environmental, Social, and Governance) score prediction. The model uses a hierarchical structure to enhance prediction accuracy by leveraging relationships between different ESG components.

## Model Architecture

The Two-Tiered ESG Model consists of:

1. **Tier 1 Model**: Predicts ESG Pillars (Environmental, Social, Governance) and disclosure scores using a set of features. This model serves as the foundation for the second tier.
   - **Input**: Features related to ESG metrics
   - **Output**: Predicted scores for each pillar
2. **Tier 2 Model**: Uses the outputs from Tier 1 as the predictors for the Bloomberg ESG score. This model refines the predictions by incorporating additional features and relationships.
   - **Input**: Predicted scores from Tier 1
   - **Output**: Final predicted Bloomberg ESG score

## Contents

- `TwoTierESGModel.py`: Main model implementation
- `TwoTieredESGModel.ipynb`: Jupyter notebook for interactive exploration and demonstration of the model

## Features

- Hierarchical prediction structure that mimics ESG rating methodologies
- Support for both regression and classification tasks
- Feature importance analysis for model interpretability
- Visualization tools for model performance assessment
- Cross-validation support for robust evaluation

## Usage

```python
from TwoTieredModel.model import TwoTieredESGModel

# Initialize the model
model = TwoTieredESGModel(tier1_features, tier2_features)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate performance
performance = model.evaluate(X_test, y_test)