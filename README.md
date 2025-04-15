# ESG Score Prediction and Optimization Models

This repository contains two advanced models for ESG (Environmental, Social, and Governance) analytics:
1. A TabTransformer-based prediction model for estimating ESG scores
2. A reinforcement learning-based optimization model for ESG strategy planning

Both models are designed to help companies understand, predict, and improve their ESG performance using state-of-the-art machine learning techniques.

## Model 1: Two Layer Model

### Overview

The Two Layer Model prediction model is designed to accurately estimate ESG scores for energy companies based on various environmental, social, and governance metrics.

### Core Components

#### Model Architecture
- Layer 1: Random Forest Regressor - to predict Pillar scores
- Layer 2: OLS Linear Regression - use outputs from Layer 1 to predict ESG Score

#### Data Processing Pipeline
- Comprehensive data preprocessing with robust handling of missing values
- Automatic categorization of features as numerical or categorical
- Industry-specific normalization of metrics

#### Model Utilities
- Feature importance analysis using attention weights
- Prediction functions for new, unseen companies
- Easy loading and saving of models

### Key Features

- **Mixed Data Handling**: Naturally processes both categorical and numerical data
- **Explainability**: Provides feature importance analysis for interpretation
- **Robustness**: Comprehensive preprocessing for real-world data
- **Energy Sector Focus**: Specifically calibrated for energy companies' ESG metrics

## Model 2: ESG Reinforcement Learning Optimization Model

### Overview

The reinforcement learning (RL) optimization model is designed to develop optimal ESG improvement strategies. It simulates the impact of various ESG initiatives and learns effective long-term strategies to maximize ESG performance while balancing financial considerations.

### Core Components

#### 1. ESG Data Analyzer
- Processes and analyzes ESG datasets to extract patterns and benchmarks
- Creates similarity metrics between companies based on ESG profiles
- Calculates industry-specific benchmarks for realistic goal-setting
- Identifies historically effective ESG actions based on real data
- Calibrates expected effects of different ESG initiatives

#### 2. Reinforcement Learning Environment
- Simulates a company's ESG decision-making process
- Represents company state using 24 different metrics (ESG scores, emissions, diversity metrics, etc.)
- Defines 17 distinct ESG actions across three pillars:
  - **Environmental**: Renewable energy investments, waste reduction, etc.
  - **Social**: Diversity programs, employee wellbeing, community engagement
  - **Governance**: Board independence, ethics programs, stakeholder engagement
- Implements realistic action effects with diminishing returns
- Features action cooldown periods to prevent repetitive strategies

#### 3. Reward System
The system uses a sophisticated multi-component reward structure:
- Base improvement rewards for ESG score increases
- Financial penalties for actions that significantly impact profitability
- Pillar balance incentives to prevent one-dimensional ESG strategies
- Diversity bonuses for using varied actions rather than repetition
- Benchmark comparison rewards for moving up industry ranking tiers
- Cost efficiency rewards for maximizing ESG impact per dollar spent

#### 4. Deep Q-Learning Neural Network Architecture
- Implements a dueling DQN architecture to separate value and advantage streams
- Uses noisy networks for improved exploration
- Features batch normalization and dropout for training stability
- Employs double DQN to reduce overestimation bias
- Includes prioritized experience replay for more efficient learning

#### 5. Curriculum Learning
- Progressively increases the difficulty of the learning task
- Adjusts parameters based on company's initial ESG performance
- Starts with simpler, shorter-term strategies before complex multi-step plans
- Dynamically adapts diversity and balance weights throughout training

### Key Features
- **Data-Driven Calibration**: Actions and their effects are calibrated based on real ESG performance data
- **Financial Sustainability**: The agent learns to improve ESG scores while minimizing negative financial impacts
- **Strategic Diversity**: The system encourages varied approaches rather than repetitive actions
- **Pillar Balance**: Rewards balanced improvements across all ESG dimensions rather than focusing only on easy wins
- **Industry Benchmarking**: Performance is evaluated relative to industry peers
- **Visualization & Analysis**: Comprehensive visualizations track ESG progression, financial impacts, and strategy effectiveness

## Usage Flow

### ESG Score Prediction

```python
# Load the trained model and preprocessor
model, preprocessor = load_model()

# Load new data
new_data = pd.read_csv('new_energy_companies.csv')

# Preprocess the data
processed_data, cat_features, num_features = preprocess_new_data(
    new_data, preprocessor.cat_features, preprocessor.num_features
)

# Make predictions
predictions = predict_esg_score(processed_data, model, preprocessor)

# Analyze feature importance
importance = get_feature_importances(model, preprocessor)
```

### ESG Strategy Optimization

```python
# Load ESG data
esg_data = pd.read_csv('energy_esg_data.csv')

# Create the environment for a specific company
env = DataDrivenESGEnvironment(
    esg_data,
    company_idx=0,  # Index of the company to analyze
    max_steps=8,    # Number of actions to simulate
    diversity_weight=3.0,
    balance_weight=2.5
)

# Train the optimization agent
scores, agent, pillar_improvements = data_driven_training(
    env,
    n_episodes=300,
    curriculum_learning=True
)

# Visualize the optimized ESG strategy
results = visualize_esg_strategy(env, agent)
```

## Requirements

- PyTorch 1.9+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- gym (for RL environment)

## Limitations and Future Work

- The prediction model is specifically calibrated for energy sector companies and may require retraining for other industries
- The RL model assumes that action effects are predictable and consistent, which may not always be the case in real-world scenarios
- Future work could include integrating more external factors such as regulatory changes, market conditions, and stakeholder sentiment
