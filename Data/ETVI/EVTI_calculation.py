import pandas as pd
import numpy as np
import os
from functools import reduce
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def read_and_prepare_data():
    print("Reading datasets...")

    # Read World Development Indicators data - THIS IS THE SOURCE OF TRUTH FOR COUNTRIES
    try:
        wdi_data = pd.read_csv('P_Data_Extract_From_World_Development_Indicators_with_metadata.csv', skipfooter=5, engine='python')
        print(f"WDI data loaded: {wdi_data.shape[0]} rows")

        # Extract unique country codes from WDI to use as source of truth
        valid_country_codes = set(wdi_data['Country Code'].unique())
        print(f"Found {len(valid_country_codes)} valid country codes in WDI data")
    except Exception as e:
        print(f"Error loading WDI data: {e}")
        return None, None, None

    # Read Education Statistics data
    try:
        # Try different possible filenames for education data
        possible_filenames = [
            'P_Data_Extract_From_Education_Statistics__All_Indicators _with_metadata.csv',
            'P_Data_Extract_From_Education_Statistics_-_All_Indicators_with_metadata.csv',
            'P_Data_Extract_From_Education_Statistics__All_Indicators_with_metadata.csv'
        ]

        edu_data = None
        for filename in possible_filenames:
            try:
                edu_data = pd.read_csv(filename, skipfooter=5, engine='python')
                print(f"Education data loaded from {filename}: {edu_data.shape[0]} rows")
                break
            except:
                continue

        if edu_data is None:
            print("Warning: Could not find education data file")
            edu_data = pd.DataFrame()
    except Exception as e:
        print(f"Error loading Education data: {e}")
        edu_data = pd.DataFrame()

    # Read TES by Source data (country codes)
    try:
        tes_data = pd.read_csv('TESBySource.csv')
        print(f"TES data loaded: {tes_data.shape[0]} rows")

        # Filter TES data to only include countries from valid_country_codes
        if not tes_data.empty:
            tes_data = tes_data[tes_data['country'].isin(valid_country_codes)]
            print(f"Filtered TES data to {tes_data.shape[0]} rows (matching WDI country codes)")
    except Exception as e:
        print(f"Error loading TES data: {e}")
        tes_data = pd.DataFrame()

    # Read Energy Balance data (country names)
    try:
        energy_balance_data = pd.read_csv('WorldEnergyBalancesHighlights2024.csv', skiprows=1)
        print(f"Energy Balances data loaded: {energy_balance_data.shape[0]} rows")
    except Exception as e:
        print(f"Warning: Could not read WorldEnergyBalancesHighlights2024.csv: {e}")
        energy_balance_data = pd.DataFrame()

    # Get country name to code mapping from WDI data
    country_mapping = None
    if 'Country Name' in wdi_data.columns and 'Country Code' in wdi_data.columns:
        country_mapping = wdi_data[['Country Name', 'Country Code']].drop_duplicates()
        print(f"Country mapping created with {country_mapping.shape[0]} entries")

    return wdi_data, edu_data, tes_data, energy_balance_data, valid_country_codes, country_mapping

def process_energy_balances(energy_data, country_mapping, valid_country_codes):
    """Process Energy Balance data to extract fossil fuel electricity mix by year."""
    if energy_data.empty or country_mapping is None:
        return pd.DataFrame()

    try:
        # Check if required columns exist
        if 'Country' not in energy_data.columns or 'Flow' not in energy_data.columns or 'Product' not in energy_data.columns:
            print("Warning: Required columns missing in Energy Balance data")
            return pd.DataFrame()

        # Filter for electricity output rows
        electricity_data = energy_data[energy_data['Flow'].str.contains('Electricity output', na=False)]

        if electricity_data.empty:
            print("Warning: No electricity output data found")
            return pd.DataFrame()

        # Filter for fossil fuels and total
        fossil_data = electricity_data[electricity_data['Product'].str.contains('Fossil fuels', case=False, na=False)]
        total_data = electricity_data[electricity_data['Product'].str.contains('Total', case=False, na=False)]

        if fossil_data.empty or total_data.empty:
            print("Warning: Missing fossil or total electricity data")
            return pd.DataFrame()

        # Get year columns (numeric columns from 2010 onwards)
        year_columns = []
        for col in energy_data.columns:
            try:
                year = int(col)
                if year >= 2010:
                    year_columns.append(col)
            except (ValueError, TypeError):
                continue

        print(f"Found {len(year_columns)} year columns in Energy Balance data: {year_columns}")

        if not year_columns:
            print("Warning: No valid year columns found in Energy Balance data")
            return pd.DataFrame()

        # Create result dataframe
        result_data = []

        # Process each country
        for country in total_data['Country'].unique():
            country_fossil = fossil_data[fossil_data['Country'] == country]
            country_total = total_data[total_data['Country'] == country]

            if country_fossil.empty or country_total.empty:
                continue

            # Calculate fossil fuel percentage for each year
            for year_col in year_columns:
                # Convert to numeric, handling errors
                fossil_sum = pd.to_numeric(country_fossil[year_col].sum(), errors='coerce')
                total_sum = pd.to_numeric(country_total[year_col].sum(), errors='coerce')

                if pd.notna(fossil_sum) and pd.notna(total_sum) and total_sum > 0:
                    ff_mix = (fossil_sum / total_sum) * 100

                    # Create row for this country-year
                    row = {
                        'Country': country,
                        'Year': int(year_col),
                        'ff_electricity_mix': ff_mix
                    }
                    result_data.append(row)

        if not result_data:
            print("Warning: No valid electricity mix data calculated")
            return pd.DataFrame()

        # Convert to DataFrame
        result_df = pd.DataFrame(result_data)
        print(f"Electricity mix calculated for {len(result_df['Country'].unique())} countries across {len(result_df['Year'].unique())} years")

        # Match country names to codes using the mapping
        result_df = pd.merge(result_df, country_mapping,
                            left_on='Country', right_on='Country Name',
                            how='inner')  # INNER join to keep only matches

        print(f"Energy balance data mapped to country codes: {result_df.shape[0]} rows")

        # Filter to keep only countries in the valid_country_codes list
        result_df = result_df[result_df['Country Code'].isin(valid_country_codes)]
        print(f"Filtered to {result_df.shape[0]} rows matching valid WDI country codes")

        # Final result has Country Code, Year, and electricity mix
        if not result_df.empty:
            result_df = result_df[['Country Code', 'Country Name', 'Year', 'ff_electricity_mix']]

        return result_df

    except Exception as e:
        print(f"Error processing Energy Balance data: {e}")
        return pd.DataFrame()

def process_tes_data(tes_data, valid_country_codes, country_mapping):
    """Process TES data to extract fossil fuel supply mix by year."""
    if tes_data.empty:
        return pd.DataFrame()

    try:
        # Check if required columns exist
        if 'country' not in tes_data.columns or 'product' not in tes_data.columns or 'year' not in tes_data.columns:
            print("Warning: Required columns missing in TES data")
            return pd.DataFrame()

        # Filter for years from 2010 onwards
        recent_tes = tes_data[tes_data['year'] >= 2010].copy()

        if recent_tes.empty:
            print("Warning: No data from 2010 onwards in TES data")
            return pd.DataFrame()

        # Create result dataframe
        result_data = []

        # Process each year
        for year in sorted(recent_tes['year'].unique()):
            year_data = recent_tes[recent_tes['year'] == year]

            # Aggregate by country and product
            tes_agg = year_data.groupby(['country', 'product'])['value'].sum().reset_index()

            # Get unique countries
            countries = tes_agg['country'].unique()

            for country in countries:
                # Skip if country is not in valid_country_codes
                if country not in valid_country_codes:
                    continue

                country_data = tes_agg[tes_agg['country'] == country]

                # Calculate fossil fuel sum
                fossil_fuels = ['COAL', 'NATGAS', 'MTOTOIL']
                fossil_sum = 0
                for fuel in fossil_fuels:
                    fuel_data = country_data[country_data['product'] == fuel]
                    if not fuel_data.empty:
                        fossil_sum += fuel_data['value'].sum()

                # Calculate total energy supply
                total_supply = country_data['value'].sum()

                if total_supply > 0:
                    ff_mix = (fossil_sum / total_supply) * 100

                    # Create row for this country-year
                    row = {
                        'Country Code': country,
                        'Year': year,
                        'ff_supply_mix': ff_mix
                    }
                    result_data.append(row)

        if not result_data:
            print("Warning: No valid supply mix data calculated")
            return pd.DataFrame()

        # Convert to DataFrame
        result_df = pd.DataFrame(result_data)

        # Add Country Name using mapping
        result_df = pd.merge(result_df, country_mapping, on='Country Code', how='inner')

        print(f"Supply mix calculated for {len(result_df['Country Code'].unique())} countries across {len(result_df['Year'].unique())} years")

        # Keep only necessary columns
        result_df = result_df[['Country Code', 'Country Name', 'Year', 'ff_supply_mix']]

        return result_df

    except Exception as e:
        print(f"Error processing TES data: {e}")
        return pd.DataFrame()

def prepare_yearly_data(master_data, valid_country_codes):
    """Convert the master dataset into yearly format with one row per country-year."""
    if master_data is None or master_data.empty:
        print("Error: No master data to process")
        return pd.DataFrame()

    # Filter to include only valid country codes
    if 'Country Code' in master_data.columns:
        master_data = master_data[master_data['Country Code'].isin(valid_country_codes)]
        print(f"Filtered master data to {master_data.shape[0]} rows matching valid WDI country codes")

    # Get all year columns (format: "2010 [YR2010]")
    year_columns = [col for col in master_data.columns if 'YR' in col]

    if not year_columns:
        print("Warning: No year columns found in master data")
        return pd.DataFrame()

    # Map year column names to actual years
    year_mapping = {}
    for col in year_columns:
        # Extract year from format like "2010 [YR2010]"
        year_match = col.split('[YR')
        if len(year_match) > 1:
            year = year_match[1].replace(']', '')
            try:
                year = int(year)
                if year >= 2010:
                    year_mapping[col] = year
            except:
                pass

    print(f"Found {len(year_mapping)} valid year columns from {min(year_mapping.values())} to {max(year_mapping.values())}")

    # Identify series columns
    series_columns = []
    if 'Series Name' in master_data.columns and 'Series Code' in master_data.columns:
        # Find unique series
        series_data = master_data[['Series Name', 'Series Code']].drop_duplicates()
        series_columns = series_data['Series Name'].tolist()
        print(f"Found {len(series_columns)} series to process")

    # Create a long-format dataset
    long_data = []

    # Track progress
    total_countries = master_data['Country Code'].nunique()
    print(f"Processing data for {total_countries} countries across {len(year_mapping)} years...")

    # Process by series for better memory efficiency
    for series_name in series_columns:
        # Filter for this series
        series_rows = master_data[master_data['Series Name'] == series_name]

        # Skip if no data
        if series_rows.empty:
            continue

        # Process each country
        for _, row in series_rows.iterrows():
            country_code = row['Country Code']
            country_name = row['Country Name'] if 'Country Name' in row else None

            # Process each year
            for year_col, year in year_mapping.items():
                if year_col in row and pd.notna(row[year_col]) and row[year_col] != '':
                    # Create a row for this country-year-series
                    long_row = {
                        'Country Code': country_code,
                        'Country Name': country_name,
                        'Year': year,
                        'Series Name': series_name,
                        'Value': row[year_col]
                    }
                    long_data.append(long_row)

    # Convert to DataFrame
    if long_data:
        long_df = pd.DataFrame(long_data)

        # Convert Value to numeric
        long_df['Value'] = pd.to_numeric(long_df['Value'], errors='coerce')

        # Pivot to get series as columns
        pivoted_df = long_df.pivot_table(
            index=['Country Code', 'Country Name', 'Year'],
            columns='Series Name',
            values='Value',
            aggfunc='first'
        ).reset_index()

        print(f"Prepared yearly data: {pivoted_df.shape[0]} country-years, {pivoted_df.shape[1]} columns")
        return pivoted_df
    else:
        print("Warning: No data to convert to yearly format")
        return pd.DataFrame()

def calculate_evti(yearly_data, electricity_data, supply_data):
    """Calculate EVTI for the entire dataset at once."""
    if yearly_data.empty:
        print("Error: No data to calculate EVTI")
        return pd.DataFrame()

    print(f"Calculating EVTI for {yearly_data.shape[0]} country-years...")

    # Make a copy to avoid modifying the original
    data = yearly_data.copy()

    # Merge with electricity mix data
    if not electricity_data.empty:
        data = pd.merge(data, electricity_data, on=['Country Code', 'Country Name', 'Year'], how='left')
        print(f"Merged electricity mix data: {data.shape[0]} rows")

    # Merge with supply mix data
    if not supply_data.empty:
        data = pd.merge(data, supply_data, on=['Country Code', 'Country Name', 'Year'], how='left')
        print(f"Merged supply mix data: {data.shape[0]} rows")

    # Calculate derived indicators
    # 1. Fossil fuel export, share of GDP
    if all(col in data.columns for col in ['Fuel exports (% of merchandise exports)', 'Merchandise exports (current US$)', 'GDP (current US$)']):
        data['ff_export_gdp'] = (data['Fuel exports (% of merchandise exports)'] *
                                data['Merchandise exports (current US$)']) / (data['GDP (current US$)'] * 100)
        print("Calculated fossil fuel export share of GDP")

    # 2. Fossil fuel rents, share of GDP
    rent_columns = ['Oil rents (% of GDP)', 'Natural gas rents (% of GDP)', 'Coal rents (% of GDP)']
    if all(col in data.columns for col in rent_columns):
        data['ff_rents_gdp'] = data[rent_columns].sum(axis=1)
        print("Calculated fossil fuel rents share of GDP")

    # 3. Energy consumption (toe per capita)
    if 'Energy use (kg of oil equivalent per capita)' in data.columns:
        data['energy_consumption'] = data['Energy use (kg of oil equivalent per capita)'] / 1000
        print("Calculated energy consumption (toe per capita)")

    # 4. Energy intensity (toe per thousand USD)
    if 'Energy intensity level of primary energy (MJ/$2017 PPP GDP)' in data.columns:
        # Convert MJ to toe (1 toe = 41,868 MJ)
        data['energy_intensity'] = data['Energy intensity level of primary energy (MJ/$2017 PPP GDP)'] / 41868 * 1000
        print("Calculated energy intensity (toe per thousand USD)")

    # Map variables to their normalized indicator names
    indicator_mapping = {
        'ff_supply_mix': 'ff_supply_mix_norm',
        'ff_electricity_mix': 'ff_electricity_mix_norm',
        'ff_export_gdp': 'ff_export_gdp_norm',
        'ff_rents_gdp': 'ff_rents_gdp_norm',
        'energy_consumption': 'energy_consumption_norm',
        'energy_intensity': 'energy_intensity_norm',
        'Poverty headcount ratio at $3.20 a day (2011 PPP) (% of population)': 'poverty_rate_norm',
        'Gini index': 'gini_index_norm',
        'Income share held by lowest 20%': 'bottom20_income_norm',
        'Unemployment, total (% of total labor force) (modeled ILO estimate)': 'unemployment_rate_norm',
        'Age dependency ratio (% of working-age population)': 'age_dependency_ratio_norm',
        'GDP per capita, PPP (constant 2017 international $)': 'gdp_per_capita_norm',
        'Research and development expenditure (% of GDP)': 'rd_expenditure_norm',
        'Researchers in R&D (per million people)': 'rd_researchers_norm',
        'UIS: Mean years of schooling (ISCED 1 or higher), population 25+ years, both sexes': 'mean_years_schooling_norm',
        'Government expenditure on education as % of GDP (%)': 'education_expenditure_norm',
        'Completion rate, primary education, both sexes (%)': 'primary_completion_rate_norm',
        'Gross enrolment ratio for tertiary education, both sexes (%)': 'tertiary_enrollment_norm',
        'Expense (% of GDP)': 'gov_spending_norm',
        'Revenue, excluding grants (% of GDP)': 'gov_revenue_norm'
    }

    # Calculate 2.5th and 97.5th percentiles for all indicators
    bounds = {}
    for indicator in indicator_mapping.keys():
        if indicator in data.columns:
            # Remove NaN values for percentile calculation
            valid_data = data[indicator].dropna()
            if len(valid_data) > 0:
                lower = np.percentile(valid_data, 2.5)
                upper = np.percentile(valid_data, 97.5)
                bounds[indicator] = {'lower': lower, 'upper': upper}
                print(f"Calculated bounds for {indicator}: lower={lower:.4f}, upper={upper:.4f}")

    # Normalize indicators
    for original, normalized in indicator_mapping.items():
        if original in data.columns and original in bounds:
            lower = bounds[original]['lower']
            upper = bounds[original]['upper']

            # Skip if upper and lower bounds are the same (would cause division by zero)
            if upper == lower:
                print(f"Warning: Upper and lower bounds are the same for {original}. Skipping normalization.")
                continue

            # For most indicators, higher values mean higher vulnerability
            if original in ['Income share held by lowest 20%',
                          'GDP per capita, PPP (constant 2017 international $)',
                          'Research and development expenditure (% of GDP)',
                          'Researchers in R&D (per million people)',
                          'UIS: Mean years of schooling (ISCED 1 or higher), population 25+ years, both sexes',
                          'Government expenditure on education as % of GDP (%)',
                          'Completion rate, primary education, both sexes (%)',
                          'Gross enrolment ratio for tertiary education, both sexes (%)',
                          'Expense (% of GDP)',
                          'Revenue, excluding grants (% of GDP)']:
                # Invert normalization for indicators where higher values mean lower vulnerability
                data[normalized] = 100 - (((data[original] - lower) / (upper - lower)) * 100)
            else:
                data[normalized] = ((data[original] - lower) / (upper - lower)) * 100

            # Ensure values are between 0 and 100
            data[normalized] = data[normalized].clip(0, 100)
            print(f"Normalized {original} to {normalized}")

    # Define the indicators for each dimension
    exposure_indicators = [
        'ff_supply_mix_norm',
        'ff_electricity_mix_norm',
        'ff_export_gdp_norm',
        'ff_rents_gdp_norm'
    ]

    sensitivity_indicators = [
        'energy_consumption_norm',
        'energy_intensity_norm',
        'poverty_rate_norm',
        'gini_index_norm',
        'bottom20_income_norm',
        'unemployment_rate_norm',
        'age_dependency_ratio_norm'
    ]

    adaptive_indicators = [
        'gdp_per_capita_norm',
        'rd_expenditure_norm',
        'rd_researchers_norm',
        'mean_years_schooling_norm',
        'education_expenditure_norm',
        'primary_completion_rate_norm',
        'tertiary_enrollment_norm',
        'gov_spending_norm',
        'gov_revenue_norm'
    ]

    # Calculate dimension scores (use only available indicators in each group)
    # First filter for which indicators are actually available in the dataframe
    available_exposure = [ind for ind in exposure_indicators if ind in data.columns]
    available_sensitivity = [ind for ind in sensitivity_indicators if ind in data.columns]
    available_adaptive = [ind for ind in adaptive_indicators if ind in data.columns]

    # Print availability info
    print(f"\nAvailable exposure indicators: {len(available_exposure)}/{len(exposure_indicators)}")
    print(f"Available sensitivity indicators: {len(available_sensitivity)}/{len(sensitivity_indicators)}")
    print(f"Available adaptive capacity indicators: {len(available_adaptive)}/{len(adaptive_indicators)}")

    # Calculate dimension scores only if there are available indicators
    if available_exposure:
        data['E_dim'] = data[available_exposure].mean(axis=1)
        print(f"Exposure dimension calculated using: {', '.join(available_exposure)}")
    else:
        data['E_dim'] = np.nan
        print("Warning: Could not calculate Exposure dimension due to missing indicators")

    if available_sensitivity:
        data['S_dim'] = data[available_sensitivity].mean(axis=1)
        print(f"Sensitivity dimension calculated using: {', '.join(available_sensitivity)}")
    else:
        data['S_dim'] = np.nan
        print("Warning: Could not calculate Sensitivity dimension due to missing indicators")

    if available_adaptive:
        data['A_dim'] = data[available_adaptive].mean(axis=1)
        print(f"Adaptive Capacity dimension calculated using: {', '.join(available_adaptive)}")
    else:
        data['A_dim'] = np.nan
        print("Warning: Could not calculate Adaptive Capacity dimension due to missing indicators")

    # Calculate ETVI (only if all three dimensions are available)
    if all(dim in data.columns for dim in ['E_dim', 'S_dim', 'A_dim']):
        # Check for at least one valid value in each dimension
        valid_all_dims = (~data[['E_dim', 'S_dim', 'A_dim']].isna()).all(axis=1)
        if valid_all_dims.any():
            data['ETVI'] = (data['E_dim'] * data['S_dim'] * (100 - data['A_dim'])) ** (1/3)
            print(f"ETVI calculated for {valid_all_dims.sum()} country-years")
        else:
            data['ETVI'] = np.nan
            print("Warning: Could not calculate ETVI for any country-year due to missing dimension values")
    else:
        data['ETVI'] = np.nan
        print("Warning: Could not calculate ETVI due to missing dimensions")

    # Keep only requested columns
    output_columns = [
        'Country Code', 'Country Name', 'Year',
        'ff_supply_mix_norm', 'ff_electricity_mix_norm', 'ff_export_gdp_norm',
        'ff_rents_gdp_norm', 'energy_consumption_norm', 'energy_intensity_norm',
        'gini_index_norm', 'bottom20_income_norm', 'unemployment_rate_norm',
        'age_dependency_ratio_norm', 'rd_expenditure_norm', 'rd_researchers_norm',
        'gov_spending_norm', 'gov_revenue_norm',
        'E_dim', 'S_dim', 'A_dim', 'ETVI'
    ]

    # Only keep columns that exist
    final_columns = [col for col in output_columns if col in data.columns]

    # Create final output dataframe
    result_df = data[final_columns].copy()

    # Drop rows with missing ETVIx
    rows_before = result_df.shape[0]
    result_df = result_df.dropna(subset=['ETVI'])
    rows_dropped = rows_before - result_df.shape[0]
    print(f"Dropped {rows_dropped} rows with missing ETVI values")

    print(f"Final output dataframe has {result_df.shape[0]} rows and {len(final_columns)} columns")

    return result_df

def main():
    try:
        print("Starting ETVI calculation...")

        # Step 1: Read and prepare the data
        wdi_data, edu_data, tes_data, energy_balance_data, valid_country_codes, country_mapping = read_and_prepare_data()

        if wdi_data is None:
            print("Error: Failed to prepare WDI dataset")
            return

        # Step 2: Process additional datasets separately with strict country code filtering
        electricity_data = process_energy_balances(energy_balance_data, country_mapping, valid_country_codes)
        supply_data = process_tes_data(tes_data, valid_country_codes, country_mapping)

        # Step 3: Combine WDI and EDU data
        if not edu_data.empty and 'Country Code' in edu_data.columns:
            # Check for duplicate columns (besides Country Code, Country Name)
            common_cols = set(wdi_data.columns) & set(edu_data.columns)
            common_cols = common_cols - {'Country Code', 'Country Name'}
            if common_cols:
                print(f"Warning: Duplicate columns found between WDI and EDU data: {common_cols}")
                # Remove duplicate columns from edu_data
                for col in common_cols:
                    if col in edu_data.columns:
                        edu_data = edu_data.drop(columns=[col])

            # Merge EDU data into WDI data
            master_data = pd.merge(wdi_data, edu_data, on='Country Code', how='left',
                                   suffixes=('', '_edu'))
            # Rename any _edu columns
            edu_cols = [col for col in master_data.columns if col.endswith('_edu')]
            for col in edu_cols:
                if col.replace('_edu', '') not in master_data.columns:
                    master_data = master_data.rename(columns={col: col.replace('_edu', '')})
                else:
                    # Keep the edu suffix if column already exists
                    pass
            print(f"WDI and EDU data merged: {master_data.shape[0]} rows, {master_data.shape[1]} columns")
        else:
            # Just use WDI data
            master_data = wdi_data.copy()

        # Step 4: Prepare yearly data (with strict country code filtering)
        yearly_data = prepare_yearly_data(master_data, valid_country_codes)

        if yearly_data.empty:
            print("Error: Failed to prepare yearly data")
            return

        # Step 5: Calculate EVTI
        evti_data = calculate_evti(yearly_data, electricity_data, supply_data)

        if evti_data.empty:
            print("Error: Failed to calculate EVTI")
            return

        # Step 6: Save results
        evti_data.to_csv('ETVI_Results.csv', index=False)
        print(f"\nETVI calculation completed. Results saved to ETVI_Results.csv")
        print(f"Output contains {evti_data.shape[0]} rows and {evti_data.shape[1]} columns")

        # Step 7: Calculate and save statistics
        stats_data = []
        for year in sorted(evti_data['Year'].unique()):
            year_data = evti_data[evti_data['Year'] == year]
            stats = {
                'Year': year,
                'Countries': year_data.shape[0],
                'Countries with ETVI': year_data['ETVI'].notna().sum(),
                'ETVI Mean': year_data['ETVI'].mean(),
                'ETVI Std': year_data['ETVI'].std(),
                'E_dim Mean': year_data['E_dim'].mean(),
                'E_dim Std': year_data['E_dim'].std(),
                'S_dim Mean': year_data['S_dim'].mean(),
                'S_dim Std': year_data['S_dim'].std(),
                'A_dim Mean': year_data['A_dim'].mean(),
                'A_dim Std': year_data['A_dim'].std()
            }
            stats_data.append(stats)

        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv('ETVI_Statistics.csv', index=False)
        print(f"Statistics saved to ETVI_Statistics.csv")

        # Display summary
        print("\nETVI Statistics by Year:")
        print(stats_df[['Year', 'Countries with ETVI', 'ETVI Mean']])

    except Exception as e:
        print(f"\nError in ETVI calculation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()