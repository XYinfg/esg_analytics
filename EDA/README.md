# Exploratory Data Analysis

## Overview
This directory contains scripts and notebooks for exploratory data analysis of ESG (Environmental, Social, and Governance) data. The analysis focuses on understanding patterns, correlations, and insights from ESG metrics across different companies and sectors.

## Contents

- `EDA.ipynb`: Main notebook with comprehensive exploratory analysis of ESG data with linear regression and random forest models

## Key Analyses

1. Distribution of ESG scores across sectors
2. Correlation between ESG metrics and financial performance
3. Temporal trends in ESG reporting and performance
4. Feature importance for predicting ESG scores

## Models

The directory includes implementations of several ESG scoring models:

- **Linear Regression**: A simple model to predict ESG scores based on selected features.
- **Random Forest**: An ensemble model to capture non-linear relationships and interactions between features.

## Usage

To run the analyses:

1. Ensure the required datasets are available in the `../Data/` directory
2. Install required dependencies: `pip install -r ../requirements.txt`
3. Run the notebooks or individual scripts

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Visualizations

The analyses produce various visualizations including:
- Correlation heatmaps
- Distribution plots
- Time series charts
- Feature importance graphs
- Prediction vs. actual comparison plots