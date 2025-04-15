# Data Directory

## Overview
This directory contains datasets used for ESG (Environmental, Social, and Governance) analytics and modeling throughout the project.

## Contents

- `energy_cleaned.csv`: Cleaned energy-related ESG data ready for analysis and modeling
- `energy_cleaned_archive.csv`: Archived version of the cleaned energy dataset for reference
- `Bloomberg/`: Contains ESG data from Bloomberg sources
- `ETVI/ETVI_results.csv`: Contains results from the ETVI model

## Usage

The datasets in this directory are used by various components of the project:
- The EDA scripts in the `EDA/` directory
- LSTM models in the `LSTM/` directory
- Transformer models in the `transformer/` directory
- Reinforcement learning agents in the `rl_agent/` directory

## Data Processing Flow

1. Raw data is collected from sources like Bloomberg
2. Data is cleaned and standardized (resulting in `energy_cleaned.csv`)
3. Further processing may be applied in specific model directories as needed

## Notes

- For reproducing results, use the datasets provided in this directory
- Archive versions are maintained for comparison and reference purposes