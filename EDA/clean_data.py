import pandas as pd
import numpy as np

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
    df.drop(columns=['Market Cap ($M)'], inplace=True)

    # Drop rows with '#N/A Requesting Data...'
    df = df[~df['BESG ESG Score'].str.contains('#N/A Requesting Data...')]

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
        'Raw Materials Used', 'Revenue, Adj', 'Net Income, Adj'
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

    # Impute missing values for numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # Impute missing values for binary columns
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).astype('bool')
            df[col] = df[col].fillna(df[col].mode()[0])

    # Impute missing values for categorical columns
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['Company', 'Ticker']:
            df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = df[col].astype('category')

    # One-hot encoding for categorical columns without Company and Ticker
    df = pd.get_dummies(df, columns=df.select_dtypes(include='category').columns)

    return df
