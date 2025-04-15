import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def clean_data(df):
    """
    Clean the input DataFrame by removing duplicates, handling missing values,
    and ensuring the target column is numeric.
    This function also prints the shape of the DataFrame before and after processing.
    """
    # Remove rows with missing BESG ESG Score
    df.dropna(subset=['BESG ESG Score'], inplace=True)

    # Drop columns with more than 70% missing values
    df.dropna(thresh=0.7*len(df), axis=1, inplace=True)

    # Drop Market Cap column
    try:
        df.drop(columns=['Market Cap ($M)'], inplace=True)
    except KeyError:
        print("Market Cap column not found, skipping drop.")

    # Drop rows with '#N/A Requesting Data...'
    if df['BESG ESG Score'].dtype == 'object':
        df = df[~df['BESG ESG Score'].str.contains('#N/A Requesting Data...', na=False)]
    else:
        # If already converted to numeric, NaN values will handle the invalid entries
        pass

    # Print basic information about the DataFrame
    print(f"Dataset shape: {df.shape}")
    print(f"Number of companies: {df['Company'].nunique()}")
    print(f"Year range: {df['Year'].min()} to {df['Year'].max()}")

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
        'Raw Materials Used', 'Revenue, Adj', 'Net Income, Adj',
    ]

    for col in df.columns:
        if col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

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
        'Company Discloses Employee Engagement Score', 'Verification Type'
    ]

    # Instantiate KNNImputer
    # KNNImputer is used to fill in missing values based on the nearest neighbors
    # Data is not normally distributed, so KNN is a good choice
    # KNNImputer is a non-parametric imputer that uses the k-nearest neighbors algorithm
    # to impute missing values in the dataset
    # It is particularly useful for datasets with a large number of features
    # and can handle both numerical and categorical data
    knn_imputer = KNNImputer(n_neighbors=5)

    # Impute missing values for numeric columns
    numeric_cols_in_df = [col for col in numeric_cols if col in df.columns]
    if len(numeric_cols_in_df) > 0:
        # Check if there are any numeric columns to impute
        print(f"Imputing missing values for numeric columns: {numeric_cols_in_df}")
    else:
        print("No numeric columns found for imputation.")
    df[numeric_cols_in_df] = knn_imputer.fit_transform(df[numeric_cols_in_df])

    # Impute missing values for binary columns
    binary_cols_in_df = [col for col in binary_cols if col in df.columns]
    if len(binary_cols_in_df) > 0:
        # Check if there are any binary columns to impute
        print(f"Imputing missing values for binary columns: {binary_cols_in_df}")
    else:
        print("No binary columns found for imputation.")
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).astype('bool')
    df[binary_cols_in_df] = knn_imputer.fit_transform(df[binary_cols_in_df])
    df[binary_cols_in_df] = df[binary_cols_in_df].astype('bool')

    # Impute missing values for categorical columns
    categorical_cols_in_df = [col for col in df.columns if df[col].dtype == 'object' and col not in ['Company', 'Ticker']]
    if len(categorical_cols_in_df) > 0:
        print(f"Imputing missing values for categorical columns: {categorical_cols_in_df}")
    else:
        print("No categorical columns found for imputation.")
    # For categorical columns, use mode imputation instead of KNN
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['Company', 'Ticker']:
            # Fill missing values with "Unknown"
            df[col].fillna('Unknown', inplace=True)
            # Convert to category type
            df[col] = df[col].astype('category')

    # Convert the target column to numeric
    df['BESG ESG Score'] = pd.to_numeric(df['BESG ESG Score'], errors='coerce')

    # One-hot encoding for categorical columns without Company and Ticker
    df = pd.get_dummies(df, columns=df.select_dtypes(include='category').columns)

    return df
