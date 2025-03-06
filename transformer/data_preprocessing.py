"""
Data preprocessing utilities for the ESG dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path='energy_cleaned.csv'):
    """
    Load and preprocess the ESG dataset
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    df (pandas.DataFrame): Preprocessed dataframe
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Print original data information
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns with missing values: {df.isnull().sum().sum()}")
    
    # Convert known numeric columns from strings to float
    numeric_cols = [
        'BESG ESG Score', 'BESG Environmental Pillar Score', 'BESG Social Pillar Score', 
        'BESG Governance Pillar Score', 'ESG Disclosure Score', 'Environmental Disclosure Score', 
        'Social Disclosure Score', 'Governance Disclosure Score', 'Nitrogen Oxide Emissions',
        'VOC Emissions', 'Particulate Emissions', 'Sulphur Dioxide / Sulphur Oxide Emissions',
        'GHG Scope 1', 'GHG Scope 2 Location-Based', 'GHG Scope 3', 'Carbon per Unit of Production',
        'Fuel Used - Natural Gas', 'Energy Per Unit of Production', 'Community Spending',
        'Pct Women in Middle and or Other Management', 'Pct Women in Workforce',
        'Fatalities - Employees', 'Fatalities - Contractors', 'Fatalities - Total',
        'Lost Time Incident Rate - Employees', 'Lost Time Incident Rate - Contractors',
        'Lost Time Incident Rate - Workforce', 'Total Recordable Incident Rate - Employees',
        'Total Recordable Incident Rate - Contractors', 'Total Recordable Incident Rate - Workforce',
        'Number of Employees - CSR', 'Number of Contractors', 'Employee Turnover Pct',
        'Years Auditor Employed', 'Size of Audit Committee', 
        'Number of Independent Directors on Audit Committee', 'Audit Committee Meetings',
        'Audit Committee Meeting Attendance Percentage', 'Board Size',
        'Number of Executives / Company Managers', 'Number of Non Executive Directors on Board',
        'Number of Board Meetings for the Year', 'Board Meeting Attendance Pct',
        'Size of Compensation Committee', 'Num of Independent Directors on Compensation Cmte',
        'Number of Compensation Committee Meetings', 'Compensation Committee Meeting Attendance %',
        'Number of Female Executives', 'Number of Women on Board', 
        'Age of the Youngest Director', 'Age of the Oldest Director',
        'Number of Independent Directors', 'Size of Nomination Committee',
        'Num of Independent Directors on Nomination Cmte', 'Number of Nomination Committee Meetings',
        'Nomination Committee Meeting Attendance Percentage', 'Board Duration (Years)',
        'Carbon Monoxide Emissions', 'CO2 Scope 1', 'Total Energy Consumption',
        'Electricity Used', 'Total Waste', 'Waste Recycled', 'Waste Sent to Landfills',
        'Total Water Withdrawal', 'Total Water Discharged', 'Water Consumption',
        'Pct Women in Senior Management', 'Pct Minorities in Workforce',
        'Pct Employees Unionized', 'Employee Training Cost',
        'Total Hours Spent by Firm - Employee Training', 'Fuel Used - Coal/Lignite',
        'Fuel Used - Crude Oil/Diesel', 'Hazardous Waste',
        'Pct Minorities in Management', 'Pct Disabled in Workforce',
        'CO2 Scope 2 Location-Based', 'Water per Unit of Production', 'Pct Recycled Materials',
        'Number of Suppliers Audited', 'Percentage Suppliers Audited',
        'Number of Supplier Audits Conducted', 'Number Supplier Facilities Audited',
        'Percentage of Suppliers in Non-Compliance', 'Number of Customer Complaints',
        'Raw Materials Used', 'Revenue, Adj', 'Net Income, Adj'
    ]
    
    # Convert string columns to numeric (handling errors)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert binary/categorical string columns to numeric
    binary_cols = [
        'Risks of Climate Change Discussed', 'Policy Against Child Labor', 
        'Gender Pay Gap Breakout', 'Human Rights Policy', 'Equal Opportunity Policy',
        'Business Ethics Policy', 'Anti-Bribery Ethics Policy', 'Health and Safety Policy',
        'Training Policy', 'Social Supply Chain Management', 'Emissions Reduction Initiatives',
        'Climate Change Policy', 'Climate Change Opportunities Discussed', 
        'Energy Efficiency Policy', 'Waste Reduction Policy', 
        'Environmental Supply Chain Management', 'Water Policy', 'Biodiversity Policy',
        'Quality Assurance and Recall Policy', 'Consumer Data Protection Policy',
        'Fair Remuneration Policy', 'Employee CSR Training', 'Renewable Energy Use',
        'Company Conducts Board Evaluations', 'Company Has Executive Share Ownership Guidelines',
        'Director Share Ownership Guidelines', 'Transition Plan Claim',
        'Adopts TNFD Recommendations', 'Zero Deforestation Policy',
        'Board Level Oversight of Biodiversity', 'Executive Level Oversight of Biodiversity',
        'Company Discloses Employee Engagement Score'
    ]
    
    # Convert 'Yes'/'No' strings to 1/0
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).fillna(0)
    
    # Fill missing values for numerical columns with median
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill missing values with the most frequent value
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # Ensure the target column is numeric
    if 'BESG ESG Score' in df.columns:
        df['BESG ESG Score'] = pd.to_numeric(df['BESG ESG Score'], errors='coerce')
        # Drop rows where target is missing
        df = df.dropna(subset=['BESG ESG Score'])
    
    print(f"Processed dataset shape: {df.shape}")
    print(f"Columns with missing values after processing: {df.isnull().sum().sum()}")
    
    return df

def identify_feature_types(df):
    """
    Identify categorical and numerical features in the dataset
    
    Parameters:
    df (pandas.DataFrame): The dataset
    
    Returns:
    cat_features (list): List of categorical feature names
    num_features (list): List of numerical feature names
    """
    # Identify categorical features (excluding the target)
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Identify numerical features (excluding the target)
    num_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'BESG ESG Score' in num_features:
        num_features.remove('BESG ESG Score')
    
    # Ensure 'Year' is treated as a categorical feature
    if 'Year' in num_features:
        num_features.remove('Year')
        if 'Year' not in cat_features:
            cat_features.append('Year')
    
    # Move binary features to categorical features if they're not already there
    binary_features = []
    for col in num_features:
        # Check if column has only 0s and 1s (binary)
        if set(df[col].dropna().unique()).issubset({0, 1}):
            binary_features.append(col)
    
    # Move binary features from numerical to categorical
    for col in binary_features:
        num_features.remove(col)
        if col not in cat_features:
            cat_features.append(col)
    
    print(f"Identified {len(cat_features)} categorical features")
    print(f"Identified {len(num_features)} numerical features")
    
    return cat_features, num_features

def prepare_data_for_modeling(df, cat_features=None, num_features=None):
    """
    Prepare data for modeling by identifying feature types and handling categorical variables
    
    Parameters:
    df (pandas.DataFrame): The dataset
    cat_features (list, optional): List of categorical feature names
    num_features (list, optional): List of numerical feature names
    
    Returns:
    df (pandas.DataFrame): Processed dataframe
    cat_features (list): List of categorical feature names
    num_features (list): List of numerical feature names
    """
    # If feature types not provided, identify them
    if cat_features is None or num_features is None:
        cat_features, num_features = identify_feature_types(df)
    
    # Convert Year to string to ensure it's treated as categorical
    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(str)
    
    # Ensure all categorical features are string type
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df, cat_features, num_features

if __name__ == "__main__":
    # Test the preprocessing functions
    df = load_and_clean_data()
    df, cat_features, num_features = prepare_data_for_modeling(df)
    print("\nSample of preprocessed data:")
    print(df.head())
    print("\nCategorical features:", cat_features[:5], "...")
    print("Numerical features:", num_features[:5], "...")