import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
from gym import spaces
import os
import random
from collections import deque, namedtuple
import copy
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Enhanced ESG Data Analyzer with Data-Driven Optimizations
class ESGDataAnalyzer:
    """Analyzes ESG dataset to extract patterns, benchmarks, and calibrate action effects with data-driven approaches"""

    def __init__(self, data, scale_factor=0.1):
        """
        Initialize the ESG data analyzer.

        Args:
            data (DataFrame): ESG dataset
            scale_factor (float): Factor to scale ESG scores (0-100 to 0-10)
        """
        self.data = data.copy()
        self.scale_factor = scale_factor
        self.esg_metrics = [
            'BESG ESG Score', 'BESG Environmental Pillar Score', 'BESG Social Pillar Score',
            'BESG Governance Pillar Score', 'ESG Disclosure Score'
        ]

        # Updated environmental metrics based on variable importance
        self.environmental_metrics = [
            'Nitrogen Oxide Emissions', 'GHG Scope 1', 'GHG Scope 2 Location-Based', 'GHG Scope 3',
            'Total Water Withdrawal', 'Total Waste', 'Waste Recycled',
            'Renewable Energy Use'
        ]

        # Updated social metrics based on variable importance
        self.social_metrics = [
            'Total Recordable Incident Rate - Employees', 'Fatalities - Employees',
            'Pct Women in Workforce', 'Pct Women in Senior Management',
            'Employee Turnover Pct', 'Community Spending',
            'Number of Female Executives'
        ]

        # Updated governance metrics based on variable importance
        self.governance_metrics = [
            'Number of Independent Directors', 'Board Size', 'Number of Women on Board',
            'Years Auditor Employed', 'Number of Non Executive Directors on Board',
            'Board Duration (Years)', 'Number of Executives / Company Managers',
            'Age of the Oldest Director', 'Number of Board Meetings for the Year',
            'Audit Committee Meetings', 'Compensation Committee Meeting Attendance %',
            'Age of the Youngest Director'
        ]

        # Store feature importance by pillar
        self.feature_importance = {
            'environmental': {
                'Nitrogen Oxide Emissions': 0.2109,
                'Board Size': 0.0875,
                'GHG Scope 2 Location-Based': 0.0715,
                'Total Recordable Incident Rate - Employees': 0.0590,
                'Number of Women on Board': 0.0575,
                'Years Auditor Employed': 0.0437,
                'Pct Women in Workforce': 0.0423,
                'Fatalities - Employees': 0.0353,
                'Number of Executives / Company Managers': 0.0324,
                'Number of Independent Directors': 0.0253,
                'Number of Female Executives': 0.0265,
                'Number of Board Meetings for the Year': 0.0223,
                'Age of the Youngest Director': 0.0211,
                'Age of the Oldest Director': 0.0197,
                'Number of Non Executive Directors on Board': 0.0198,
                'Audit Committee Meetings': 0.0175,
                'Board Duration (Years)': 0.0065,
                'Compensation Committee Meeting Attendance %': 0.0074
            },
            'governance': {
                'Number of Independent Directors': 0.4177,
                'Board Duration (Years)': 0.0794,
                'Compensation Committee Meeting Attendance %': 0.0626,
                'Age of the Oldest Director': 0.0392,
                'Board Size': 0.0315,
                'Total Recordable Incident Rate - Employees': 0.0273,
                'Number of Women on Board': 0.0261,
                'Audit Committee Meetings': 0.0242,
                'Number of Executives / Company Managers': 0.0189,
                'Years Auditor Employed': 0.0182,
                'Nitrogen Oxide Emissions': 0.0177,
                'Number of Non Executive Directors on Board': 0.0175,
                'Number of Board Meetings for the Year': 0.0224,
                'Age of the Youngest Director': 0.0157,
                'GHG Scope 2 Location-Based': 0.0140,
                'Pct Women in Workforce': 0.0119,
                'Fatalities - Employees': 0.0031,
                'Number of Female Executives': 0.0039
            },
            'social': {
                'Total Recordable Incident Rate - Employees': 0.1651,
                'Fatalities - Employees': 0.0859,
                'Number of Non Executive Directors on Board': 0.0732,
                'Pct Women in Workforce': 0.0465,
                'GHG Scope 2 Location-Based': 0.0419,
                'Nitrogen Oxide Emissions': 0.0415,
                'Years Auditor Employed': 0.0384,
                'Board Duration (Years)': 0.0297,
                'Number of Executives / Company Managers': 0.0299,
                'Number of Women on Board': 0.0292,
                'Audit Committee Meetings': 0.0294,
                'Board Size': 0.0268,
                'Number of Board Meetings for the Year': 0.0260,
                'Age of the Oldest Director': 0.0224,
                'Age of the Youngest Director': 0.0211,
                'Number of Independent Directors': 0.0209,
                'Number of Female Executives': 0.0159,
                'Compensation Committee Meeting Attendance %': 0.0075
            }
        }

        self.financial_metrics = [
            'Revenue, Adj', 'Net Income, Adj', 'Market Cap ($M)'
        ]

        # Preprocess data
        self._preprocess_data()

        # Extract industry information if available
        if 'Industry' in self.data.columns:
            self.industries = self.data['Industry'].unique()
        else:
            # Try to infer industries using clustering
            self._infer_industries()

        # Calculate industry benchmarks
        self.industry_benchmarks = self._calculate_industry_benchmarks()

        # Create company similarity matrix
        self.company_similarity = self._create_company_similarity_matrix()

        # Calculate typical year-over-year changes
        self.yearly_changes = self._calculate_yearly_changes()

        # Analyze correlations between ESG metrics and financial performance
        self.esg_financial_correlations = self._analyze_esg_financial_correlations()

        # Identify most effective actions
        self.effective_actions = self._identify_effective_actions()

        # NEW: Analyze ESG improvement patterns at different score levels
        self.improvement_factors = self._analyze_improvement_factors()

        # NEW: Analyze financial impacts of ESG initiatives
        self.financial_impacts = self._analyze_financial_impacts()

        # NEW: Analyze pillar balance importance
        self.pillar_balance_analysis = self._analyze_pillar_balance_importance()

    def _preprocess_data(self):
        """Preprocess the dataset - handle missing values, outliers, etc."""
        # Convert relevant columns to numeric
        numeric_cols = (
            self.esg_metrics + self.environmental_metrics +
            self.social_metrics + self.governance_metrics +
            self.financial_metrics
        )

        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Scale ESG scores to 0-10 range if they're on 0-100 scale
        for col in self.esg_metrics:
            if col in self.data.columns:
                # Check if the column has values > 10 (likely on 0-100 scale)
                if self.data[col].max() > 10:
                    self.data[col] = self.data[col] * self.scale_factor

        # Fill missing values with reasonable defaults
        if 'BESG ESG Score' in self.data.columns:
            self.data['BESG ESG Score'].fillna(self.data['BESG ESG Score'].mean(), inplace=True)

        # Handle any remaining missing values in key metrics
        for col in self.esg_metrics:
            if col in self.data.columns:
                self.data[col].fillna(self.data[col].mean(), inplace=True)

    def _infer_industries(self):
        """Infer industry groupings using clustering if industry column not available"""
        # Get relevant metrics for clustering
        cluster_metrics = [col for col in self.esg_metrics if col in self.data.columns]
        if not cluster_metrics:
            # Can't infer industries without ESG metrics
            self.industries = np.array(['Unknown'])
            self.data['Industry'] = 'Unknown'
            return

        # Get average metrics by company
        company_avg = self.data.groupby('Company')[cluster_metrics].mean().reset_index()

        # Drop companies with missing values
        company_avg.dropna(inplace=True)

        if len(company_avg) < 5:
            # Not enough data to cluster
            self.industries = np.array(['Unknown'])
            self.data['Industry'] = 'Unknown'
            return

        # Scale the data
        X = company_avg[cluster_metrics].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine optimal number of clusters (simple method)
        max_clusters = min(10, len(company_avg) // 2)
        if max_clusters < 2:
            max_clusters = 2

        inertias = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        # Simple elbow method
        if len(inertias) > 1:
            # Calculate delta of inertias
            deltas = np.array([inertias[i] - inertias[i+1] for i in range(len(inertias)-1)])
            # Find elbow point
            elbow = np.argmin(deltas) + 2
        else:
            elbow = 2

        # Cluster companies
        kmeans = KMeans(n_clusters=elbow, random_state=42)
        company_avg['Industry'] = kmeans.fit_predict(X_scaled)
        company_avg['Industry'] = 'Cluster_' + company_avg['Industry'].astype(str)

        # Map industry clusters back to original data
        industry_map = dict(zip(company_avg['Company'], company_avg['Industry']))
        self.data['Industry'] = self.data['Company'].map(industry_map)
        self.data['Industry'].fillna('Unknown', inplace=True)

        self.industries = self.data['Industry'].unique()

    def _calculate_industry_benchmarks(self):
        """Calculate industry benchmarks for ESG metrics"""
        benchmarks = {}

        if 'Industry' not in self.data.columns:
            return benchmarks

        # For each industry, calculate percentiles for ESG metrics
        for industry in self.industries:
            industry_data = self.data[self.data['Industry'] == industry]
            if len(industry_data) == 0:
                continue

            industry_benchmarks = {}
            for metric in self.esg_metrics:
                if metric in self.data.columns:
                    # Calculate percentiles
                    p25 = industry_data[metric].quantile(0.25)
                    p50 = industry_data[metric].quantile(0.50)
                    p75 = industry_data[metric].quantile(0.75)
                    p90 = industry_data[metric].quantile(0.90)

                    industry_benchmarks[metric] = {
                        'p25': p25, 'p50': p50, 'p75': p75, 'p90': p90,
                        'mean': industry_data[metric].mean(),
                        'std': industry_data[metric].std()
                    }

            benchmarks[industry] = industry_benchmarks

        return benchmarks

    def _create_company_similarity_matrix(self):
        """Create a matrix of company similarities based on ESG metrics"""
        # Get all companies
        companies = self.data['Company'].unique()
        similarity_matrix = pd.DataFrame(index=companies, columns=companies)

        # Features to use for similarity
        features = [col for col in self.esg_metrics if col in self.data.columns]

        if not features:
            # No ESG metrics available, can't compute similarity
            return pd.DataFrame()

        # Get latest data for each company
        latest_data = {}
        for company in companies:
            company_data = self.data[self.data['Company'] == company]
            if len(company_data) == 0:
                continue

            # Get most recent year's data
            latest_year = company_data['Year'].max()
            latest = company_data[company_data['Year'] == latest_year][features].values

            if len(latest) > 0 and not np.isnan(latest).all():
                latest_data[company] = latest[0]

        # Calculate cosine similarity between companies
        for company1 in latest_data:
            for company2 in latest_data:
                if company1 == company2:
                    similarity_matrix.loc[company1, company2] = 1.0
                else:
                    # Calculate cosine similarity
                    v1 = latest_data[company1].reshape(1, -1)
                    v2 = latest_data[company2].reshape(1, -1)

                    # Handle potential NaNs
                    mask = ~np.isnan(v1) & ~np.isnan(v2)
                    if np.any(mask):
                        v1_clean = v1[mask].reshape(1, -1)
                        v2_clean = v2[mask].reshape(1, -1)

                        if v1_clean.size > 0 and v2_clean.size > 0:
                            try:
                                sim = cosine_similarity(v1_clean, v2_clean)[0][0]
                                similarity_matrix.loc[company1, company2] = sim
                            except:
                                similarity_matrix.loc[company1, company2] = np.nan
                        else:
                            similarity_matrix.loc[company1, company2] = np.nan
                    else:
                        similarity_matrix.loc[company1, company2] = np.nan

        # Fill NaNs with 0 (no similarity)
        similarity_matrix.fillna(0, inplace=True)

        return similarity_matrix

    def _calculate_yearly_changes(self):
        """Calculate typical year-over-year changes in ESG metrics"""
        yearly_changes = {}

        # Get all numeric columns
        numeric_cols = (
            self.esg_metrics + self.environmental_metrics +
            self.social_metrics + self.governance_metrics
        )

        available_cols = [col for col in numeric_cols if col in self.data.columns]

        # Calculate year-over-year changes for each company
        companies = self.data['Company'].unique()

        for company in companies:
            company_data = self.data[self.data['Company'] == company].sort_values('Year')

            if len(company_data) < 2:
                continue

            # Calculate changes for each column
            for col in available_cols:
                if col not in yearly_changes:
                    yearly_changes[col] = []

                # Calculate diff and drop first row (which will be NaN)
                diffs = company_data[col].diff().dropna().tolist()
                yearly_changes[col].extend(diffs)

        # Calculate statistics for each metric
        changes_stats = {}
        for col, changes in yearly_changes.items():
            if not changes:
                continue

            # Remove outliers (simple method: keep values within 3 std of mean)
            changes = np.array(changes)
            mean = np.mean(changes)
            std = np.std(changes)

            # Filter out extreme outliers
            filtered_changes = changes[(changes >= mean - 3*std) & (changes <= mean + 3*std)]

            if len(filtered_changes) > 0:
                changes_stats[col] = {
                    'mean': np.mean(filtered_changes),
                    'median': np.median(filtered_changes),
                    'std': np.std(filtered_changes),
                    'p25': np.percentile(filtered_changes, 25),
                    'p75': np.percentile(filtered_changes, 75),
                    'p90': np.percentile(filtered_changes, 90),
                    'min': np.min(filtered_changes),
                    'max': np.max(filtered_changes)
                }

        return changes_stats

    def _analyze_esg_financial_correlations(self):
        """Analyze correlations between ESG metrics and financial performance"""
        correlations = {}

        # Financial metrics to analyze
        fin_metrics = [col for col in self.financial_metrics if col in self.data.columns]

        if not fin_metrics:
            return correlations

        # ESG metrics to correlate
        esg_cols = [col for col in self.esg_metrics if col in self.data.columns]

        if not esg_cols:
            return correlations

        # Calculate correlations
        corr_data = self.data[esg_cols + fin_metrics].corr()

        # Extract correlations between ESG and financial metrics
        for esg_col in esg_cols:
            correlations[esg_col] = {}
            for fin_col in fin_metrics:
                correlations[esg_col][fin_col] = corr_data.loc[esg_col, fin_col]

        return correlations

    def _identify_effective_actions(self):
        """Identify the most effective ESG actions based on historical data"""
        effective_actions = {
            'environmental': [],
            'social': [],
            'governance': []
        }

        # Find companies with significant ESG improvement
        if 'BESG ESG Score' in self.data.columns:
            # Group by company
            company_data = self.data.groupby('Company')

            for company, group in company_data:
                if len(group) < 2:
                    continue

                # Sort by year
                group = group.sort_values('Year')

                # Calculate overall improvement
                esg_start = group['BESG ESG Score'].iloc[0]
                esg_end = group['BESG ESG Score'].iloc[-1]
                esg_improvement = esg_end - esg_start

                # Only consider significant improvements
                if esg_improvement > 1.0:  # More than 1 point improvement on 0-10 scale
                    # Check where the improvement came from
                    for pillar, metrics in zip(
                        ['environmental', 'social', 'governance'],
                        [
                            ['BESG Environmental Pillar Score'] + self.environmental_metrics,
                            ['BESG Social Pillar Score'] + self.social_metrics,
                            ['BESG Governance Pillar Score'] + self.governance_metrics
                        ]
                    ):
                        # Check available metrics
                        available_metrics = [m for m in metrics if m in self.data.columns]

                        if not available_metrics:
                            continue

                        # Find metrics with significant change
                        changed_metrics = []
                        for metric in available_metrics:
                            if metric not in group.columns:
                                continue

                            start_val = group[metric].iloc[0]
                            end_val = group[metric].iloc[-1]

                            # Check if numerical
                            if pd.notnull(start_val) and pd.notnull(end_val):
                                # Calculate relative change
                                if start_val != 0:
                                    rel_change = (end_val - start_val) / abs(start_val)
                                else:
                                    rel_change = np.inf if end_val > 0 else -np.inf if end_val < 0 else 0

                                # Significant change threshold depends on the metric
                                if (metric == 'BESG Environmental Pillar Score' and end_val - start_val > 0.5) or \
                                   (metric == 'BESG Social Pillar Score' and end_val - start_val > 0.5) or \
                                   (metric == 'BESG Governance Pillar Score' and end_val - start_val > 0.5) or \
                                   (metric.startswith('GHG') and rel_change < -0.1) or \
                                   (metric == 'Renewable Energy Use' and rel_change > 0.1) or \
                                   (metric in ['Pct Women in Workforce', 'Pct Women in Senior Management'] and rel_change > 0.1) or \
                                   (metric == 'Employee Turnover Pct' and rel_change < -0.1) or \
                                   (metric == 'Number of Independent Directors' and rel_change > 0.1):
                                    changed_metrics.append((metric, end_val - start_val))

                        if changed_metrics:
                            # Add to effective actions
                            effective_actions[pillar].append({
                                'company': company,
                                'improvement': esg_improvement,
                                'metrics': changed_metrics,
                                'year_span': (group['Year'].iloc[0], group['Year'].iloc[-1])
                            })

        # Sort effective actions by improvement magnitude
        for pillar in effective_actions:
            effective_actions[pillar] = sorted(
                effective_actions[pillar],
                key=lambda x: x['improvement'],
                reverse=True
            )

        return effective_actions

    # NEW: Data-driven analysis of ESG improvement patterns
    def _analyze_improvement_factors(self):
        """Analyze how ESG improvement varies at different score levels"""
        if 'BESG ESG Score' not in self.data.columns:
            return {}

        # Define score ranges for analysis
        score_ranges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10)]
        improvement_data = {}

        # Store all improvements to calculate median later
        all_improvements = []

        for low, high in score_ranges:
            # Track improvements for this score range
            range_improvements = []

            # Find companies with scores in this range
            for company in self.data['Company'].unique():
                company_data = self.data[self.data['Company'] == company].sort_values('Year')

                if len(company_data) < 2:
                    continue

                # Check each year for scores in this range
                for i in range(len(company_data) - 1):
                    current_score = company_data['BESG ESG Score'].iloc[i]
                    next_score = company_data['BESG ESG Score'].iloc[i+1]

                    # If starting score is in this range, record the improvement
                    if low <= current_score < high:
                        improvement = next_score - current_score
                        range_improvements.append(improvement)
                        all_improvements.append(improvement)

            # Store statistics if we have data
            if range_improvements:
                improvement_data[(low, high)] = {
                    'count': len(range_improvements),
                    'mean': np.mean(range_improvements),
                    'median': np.median(range_improvements),
                    'p25': np.percentile(range_improvements, 25),
                    'p75': np.percentile(range_improvements, 75)
                }

        # Calculate overall median improvement for normalization
        if all_improvements:
            overall_median = np.median(all_improvements)

            # Create normalized factors
            improvement_factors = {}
            for score_range, stats in improvement_data.items():
                if overall_median != 0:
                    # FIXED: Cap the factor to a reasonable value (2.0 max)
                    factor = min(2.0, stats['median'] / overall_median)
                else:
                    factor = 1.0
                improvement_factors[score_range] = factor

            return improvement_factors

        # FIXED: Default factors with a reasonable progression when no data
        return {
            (0, 1): 1.5,    # Easiest to improve from very low scores
            (1, 2): 1.3,
            (2, 3): 1.2,
            (3, 4): 1.1,
            (4, 5): 1.0,    # Baseline
            (5, 6): 0.9,
            (6, 7): 0.8,
            (7, 8): 0.7,
            (8, 9): 0.5,
            (9, 10): 0.3    # Very hard to improve near perfect scores
        }

    # NEW: Data-driven analysis of financial impacts
    def _analyze_financial_impacts(self):
        """Analyze financial impacts of ESG initiatives based on historical data"""
        if 'Net Income, Adj' not in self.data.columns or 'BESG ESG Score' not in self.data.columns:
            return {}

        # Initiative mapping based on metric changes
        initiative_indicators = {
            'renewable_investment': ['Renewable Energy Use', 'GHG Scope 1', 'GHG Scope 2 Location-Based'],
            'water_conservation': ['Total Water Withdrawal'],
            'waste_reduction': ['Total Waste', 'Waste Recycled'],
            'diversity_inclusion': ['Pct Women in Workforce', 'Pct Women in Senior Management'],
            'employee_wellbeing': ['Employee Turnover Pct'],
            'board_independence': ['Number of Independent Directors']
        }

        # Detect initiatives based on significant metric changes
        for company in self.data['Company'].unique():
            for initiative, metrics in initiative_indicators.items():
                init_col = f"{initiative}_implemented"
                self.data[init_col] = 0

        # Identify implementation years for each company and initiative
        for company in self.data['Company'].unique():
            company_data = self.data[self.data['Company'] == company].sort_values('Year')

            if len(company_data) < 2:
                continue

            for initiative, metrics in initiative_indicators.items():
                available_metrics = [m for m in metrics if m in company_data.columns]

                if not available_metrics:
                    continue

                for i in range(len(company_data) - 1):
                    # Check for significant improvements in relevant metrics
                    significant_changes = 0

                    for metric in available_metrics:
                        current_val = company_data[metric].iloc[i]
                        next_val = company_data[metric].iloc[i+1]

                        if pd.notnull(current_val) and pd.notnull(next_val):
                            # Check for significant improvement
                            if metric == 'Renewable Energy Use' and next_val > current_val + 5:
                                significant_changes += 1
                            elif metric.startswith('GHG') and next_val < current_val * 0.9:
                                significant_changes += 1
                            elif metric == 'Total Water Withdrawal' and next_val < current_val * 0.9:
                                significant_changes += 1
                            elif metric == 'Total Waste' and next_val < current_val * 0.9:
                                significant_changes += 1
                            elif metric == 'Waste Recycled' and next_val > current_val + 5:
                                significant_changes += 1
                            elif metric in ['Pct Women in Workforce', 'Pct Women in Senior Management'] and next_val > current_val + 3:
                                significant_changes += 1
                            elif metric == 'Employee Turnover Pct' and next_val < current_val * 0.9:
                                significant_changes += 1
                            elif metric == 'Number of Independent Directors' and next_val > current_val:
                                significant_changes += 1

                    # If enough metrics improved, mark as initiative implementation
                    if significant_changes >= min(2, len(available_metrics)):
                        implementation_year = company_data['Year'].iloc[i+1]
                        self.data.loc[(self.data['Company'] == company) &
                                      (self.data['Year'] == implementation_year),
                                      f"{initiative}_implemented"] = 1

        # Analyze financial impacts
        financial_impacts = {}

        for initiative in initiative_indicators.keys():
            implementation_col = f"{initiative}_implemented"

            # Skip if no implementations detected
            if implementation_col not in self.data.columns or self.data[implementation_col].sum() == 0:
                continue

            # Get companies that implemented this initiative
            companies_with_initiative = self.data[self.data[implementation_col] == 1]['Company'].unique()

            if len(companies_with_initiative) == 0:
                continue

            # Calculate financial impacts
            immediate_impacts = []  # Same year
            year1_impacts = []      # One year later
            year2_impacts = []      # Two years later

            for company in companies_with_initiative:
                company_data = self.data[self.data['Company'] == company].sort_values('Year')
                implementation_years = company_data[company_data[implementation_col] == 1]['Year'].tolist()

                for impl_year in implementation_years:
                    # Get financial data before implementation
                    prev_years = company_data[company_data['Year'] < impl_year]['Year'].tolist()
                    if not prev_years:
                        continue

                    prev_year = max(prev_years)
                    prev_income = company_data[company_data['Year'] == prev_year]['Net Income, Adj'].iloc[0]

                    # Get financial data after implementation
                    impl_income = company_data[company_data['Year'] == impl_year]['Net Income, Adj'].iloc[0]

                    year1_data = company_data[company_data['Year'] == impl_year + 1]
                    year1_income = year1_data['Net Income, Adj'].iloc[0] if not year1_data.empty else None

                    year2_data = company_data[company_data['Year'] == impl_year + 2]
                    year2_income = year2_data['Net Income, Adj'].iloc[0] if not year2_data.empty else None

                    # Calculate relative impacts
                    if prev_income != 0:
                        immediate_impacts.append((impl_income - prev_income) / abs(prev_income))
                        if year1_income is not None:
                            year1_impacts.append((year1_income - prev_income) / abs(prev_income))
                        if year2_income is not None:
                            year2_impacts.append((year2_income - prev_income) / abs(prev_income))

            # Store impact data if available
            if immediate_impacts:
                financial_impacts[initiative] = {
                    'count': len(immediate_impacts),
                    'immediate': np.median(immediate_impacts),
                    'year1': np.median(year1_impacts) if year1_impacts else None,
                    'year2': np.median(year2_impacts) if year2_impacts else None
                }

        # Create default values for missing initiatives based on available data
        all_immediate_impacts = []
        for initiative, impacts in financial_impacts.items():
            all_immediate_impacts.append(impacts['immediate'])

        # Use median impact for missing initiatives
        default_impact = np.median(all_immediate_impacts) if all_immediate_impacts else -0.05

        # Ensure we have data for all initiatives
        for initiative in initiative_indicators.keys():
            if initiative not in financial_impacts:
                financial_impacts[initiative] = {
                    'count': 0,
                    'immediate': default_impact,
                    'year1': default_impact * 0.5,  # Assume recovery in year 1
                    'year2': default_impact * 0.2   # Assume more recovery in year 2
                }

        return financial_impacts

    # NEW: Data-driven analysis of pillar balance importance
    def _analyze_pillar_balance_importance(self):
        """Analyze whether balanced or focused ESG approach leads to better outcomes"""
        if not all(p in self.data.columns for p in ['BESG Environmental Pillar Score', 'BESG Social Pillar Score', 'BESG Governance Pillar Score', 'BESG ESG Score']):
            # Default to slight importance if we can't analyze
            return {
                'balanced_approach_avg_improvement': 0.2,
                'focused_approach_avg_improvement': 0.15,
                'balance_advantage': 0.05,
                'recommendation': 'balanced',
                'optimal_weight': 1.0
            }

        # Track improvements with different balance levels
        balance_improvements = {
            'high_balance': [],    # Low standard deviation between pillars
            'medium_balance': [],  # Medium standard deviation
            'low_balance': []      # High standard deviation (focused approach)
        }

        # Analyze for each company year-to-year
        for company in self.data['Company'].unique():
            company_data = self.data[self.data['Company'] == company].sort_values('Year')

            if len(company_data) < 2:
                continue

            for i in range(len(company_data) - 1):
                current_row = company_data.iloc[i]
                next_row = company_data.iloc[i+1]

                # Calculate pillar balance
                pillars = ['BESG Environmental Pillar Score', 'BESG Social Pillar Score', 'BESG Governance Pillar Score']
                pillar_scores = [current_row[p] for p in pillars]

                # Skip if any pillar score is missing
                if any(pd.isna(score) for score in pillar_scores):
                    continue

                # Calculate standard deviation as measure of balance
                pillar_std = np.std(pillar_scores)

                # Calculate ESG improvement
                current_esg = current_row['BESG ESG Score']
                next_esg = next_row['BESG ESG Score']
                improvement = next_esg - current_esg

                # Categorize by balance level
                if pillar_std < 0.5:  # Highly balanced
                    balance_improvements['high_balance'].append(improvement)
                elif pillar_std < 1.5:  # Moderately balanced
                    balance_improvements['medium_balance'].append(improvement)
                else:  # Focused/imbalanced
                    balance_improvements['low_balance'].append(improvement)

        # Calculate average improvements for each balance level
        avg_improvements = {}
        for balance_level, improvements in balance_improvements.items():
            if improvements:
                avg_improvements[balance_level] = np.mean(improvements)
            else:
                avg_improvements[balance_level] = 0

        # Determine if balance matters
        if avg_improvements.get('high_balance', 0) > avg_improvements.get('low_balance', 0):
            recommendation = 'balanced'
            balance_advantage = avg_improvements.get('high_balance', 0) - avg_improvements.get('low_balance', 0)

            # Calculate optimal weight based on advantage magnitude
            if balance_advantage > 0.5:  # Large advantage
                optimal_weight = 3.0
            elif balance_advantage > 0.2:  # Moderate advantage
                optimal_weight = 2.0
            else:  # Small advantage
                optimal_weight = 1.0
        else:
            recommendation = 'focused'
            balance_advantage = avg_improvements.get('low_balance', 0) - avg_improvements.get('high_balance', 0)
            optimal_weight = 0.1  # Minimal penalty for imbalance

        return {
            'balanced_approach_avg_improvement': avg_improvements.get('high_balance', 0),
            'focused_approach_avg_improvement': avg_improvements.get('low_balance', 0),
            'medium_balance_avg_improvement': avg_improvements.get('medium_balance', 0),
            'balance_advantage': balance_advantage,
            'recommendation': recommendation,
            'optimal_weight': optimal_weight
        }

    def get_improvement_factor(self, current_score):
        """Get data-driven multiplier for ESG score improvement based on current score"""
        # FIXED: Ensure the score is in the valid range
        current_score = max(0, min(10, current_score))

        # Find the appropriate score range
        for (low, high), factor in self.improvement_factors.items():
            if low <= current_score < high:
                # FIXED: Add a sanity cap to prevent excessive factors
                return min(2.0, factor)

        # Default factor if no match
        return 1.0

    def get_financial_impact(self, initiative, timeframe='immediate'):
        """Get data-driven financial impact for a specific initiative"""
        if initiative in self.financial_impacts:
            if timeframe in self.financial_impacts[initiative]:
                return self.financial_impacts[initiative][timeframe]

        # Default impact if no data
        return -0.05

    def get_pillar_balance_recommendation(self):
        """Get data-driven recommendation for pillar balance importance"""
        return self.pillar_balance_analysis

    def get_similar_companies(self, company, top_n=5):
        """Get most similar companies to the given company"""
        if self.company_similarity.empty or company not in self.company_similarity.index:
            return []

        similarities = self.company_similarity.loc[company].sort_values(ascending=False)
        # Skip the first one (which is the company itself)
        return similarities.index[1:top_n+1].tolist()

    def get_industry_benchmark(self, company, metric):
        """Get industry benchmark for a company and metric"""
        if 'Industry' not in self.data.columns or company not in self.data['Company'].values:
            return None

        # Get company's industry
        industry = self.data[self.data['Company'] == company]['Industry'].iloc[0]

        # Return benchmark if available
        if industry in self.industry_benchmarks and metric in self.industry_benchmarks[industry]:
            return self.industry_benchmarks[industry][metric]

        return None

    def get_typical_change(self, metric):
        """Get typical year-over-year change for a metric"""
        if metric in self.yearly_changes:
            return self.yearly_changes[metric]

        return None

    def get_financial_correlation(self, esg_metric):
        """Get financial correlations for an ESG metric"""
        if esg_metric in self.esg_financial_correlations:
            return self.esg_financial_correlations[esg_metric]

        return None

    def get_effective_actions_for_pillar(self, pillar):
        """Get effective actions for a specific pillar"""
        if pillar in self.effective_actions:
            return self.effective_actions[pillar]

        return []

    def calibrate_action_effects(self):
        """Calibrate action effects based on historical data and variable importance"""
        calibrated_effects = {}

        # Environmental actions
        env_actions = {
            'Emissions Reduction Program': {
                'metrics': ['Nitrogen Oxide Emissions', 'GHG Scope 1', 'GHG Scope 2 Location-Based',
                           'BESG Environmental Pillar Score'],
                'direction': [-1, -1, -1, 1],  # 1 for increase, -1 for decrease
                'initiative_type': 'emissions_reduction'
            },
            'Renewable Energy Investment': {
                'metrics': ['Renewable Energy Use', 'GHG Scope 1', 'GHG Scope 2 Location-Based',
                           'BESG Environmental Pillar Score'],
                'direction': [1, -1, -1, 1],
                'initiative_type': 'renewable_investment'
            },
            'Water Conservation Program': {
                'metrics': ['Total Water Withdrawal', 'BESG Environmental Pillar Score'],
                'direction': [-1, 1],
                'initiative_type': 'water_conservation'
            },
            'Waste Reduction Initiative': {
                'metrics': ['Total Waste', 'Waste Recycled', 'BESG Environmental Pillar Score'],
                'direction': [-1, 1, 1],
                'initiative_type': 'waste_reduction'
            },
            'Energy Efficiency Upgrade': {
                'metrics': ['GHG Scope 1', 'GHG Scope 2 Location-Based', 'BESG Environmental Pillar Score'],
                'direction': [-1, -1, 1],
                'initiative_type': 'renewable_investment'
            },
            'Supply Chain Emissions Program': {
                'metrics': ['GHG Scope 3', 'BESG Environmental Pillar Score'],
                'direction': [-1, 1],
                'initiative_type': 'emissions_reduction'
            }
        }

        # Social actions
        soc_actions = {
            'Workplace Safety Program': {
                'metrics': ['Total Recordable Incident Rate - Employees', 'Fatalities - Employees',
                          'BESG Social Pillar Score'],
                'direction': [-1, -1, 1],
                'initiative_type': 'workplace_safety'
            },
            'Diversity & Inclusion Program': {
                'metrics': ['Pct Women in Workforce', 'Pct Women in Senior Management',
                          'Number of Female Executives', 'BESG Social Pillar Score'],
                'direction': [1, 1, 1, 1],
                'initiative_type': 'diversity_inclusion'
            },
            'Employee Wellbeing Initiative': {
                'metrics': ['Employee Turnover Pct', 'Total Recordable Incident Rate - Employees',
                          'BESG Social Pillar Score'],
                'direction': [-1, -1, 1],
                'initiative_type': 'employee_wellbeing'
            },
            'Community Engagement Project': {
                'metrics': ['Community Spending', 'BESG Social Pillar Score'],
                'direction': [1, 1],
                'initiative_type': 'diversity_inclusion'
            },
            'Worker Safety Program': {
                'metrics': ['Fatalities - Employees', 'Total Recordable Incident Rate - Employees',
                          'BESG Social Pillar Score'],
                'direction': [-1, -1, 1],
                'initiative_type': 'workplace_safety'
            },
            'Workforce Development Training': {
                'metrics': ['Employee Turnover Pct', 'BESG Social Pillar Score'],
                'direction': [-1, 1],
                'initiative_type': 'employee_wellbeing'
            }
        }

        # Governance actions
        gov_actions = {
            'Board Independence Enhancement': {
                'metrics': ['Number of Independent Directors', 'Number of Non Executive Directors on Board',
                          'BESG Governance Pillar Score'],
                'direction': [1, 1, 1],
                'initiative_type': 'board_independence'
            },
            'Board Diversity Improvement': {
                'metrics': ['Number of Women on Board', 'BESG Governance Pillar Score'],
                'direction': [1, 1],
                'initiative_type': 'board_diversity'
            },
            'Audit Process Optimization': {
                'metrics': ['Years Auditor Employed', 'Audit Committee Meetings', 'BESG Governance Pillar Score'],
                'direction': [-1, 1, 1],  # Reduce years auditor employed (fresher perspective)
                'initiative_type': 'audit_improvement'
            },
            'Board Effectiveness Program': {
                'metrics': ['Number of Board Meetings for the Year', 'Compensation Committee Meeting Attendance %',
                          'BESG Governance Pillar Score'],
                'direction': [1, 1, 1],
                'initiative_type': 'board_effectiveness'
            },
            'Executive Compensation Reform': {
                'metrics': ['Compensation Committee Meeting Attendance %', 'BESG Governance Pillar Score'],
                'direction': [1, 1],
                'initiative_type': 'board_independence'
            },
            'Ethics & Compliance Program': {
                'metrics': ['BESG Governance Pillar Score'],
                'direction': [1],
                'initiative_type': 'board_independence'
            },
            'Stakeholder Engagement Policy': {
                'metrics': ['Number of Board Meetings for the Year', 'BESG Governance Pillar Score'],
                'direction': [1, 1],
                'initiative_type': 'board_independence'
            }
        }

        # Integrated actions
        int_actions = {
            'Comprehensive ESG Disclosure': {
                'metrics': ['ESG Disclosure Score', 'Environmental Disclosure Score', 'Social Disclosure Score',
                          'Governance Disclosure Score', 'BESG ESG Score', 'BESG Environmental Pillar Score',
                          'BESG Social Pillar Score', 'BESG Governance Pillar Score'],
                'direction': [1, 1, 1, 1, 1, 1, 1, 1],
                'initiative_type': 'esg_disclosure'
            },
            'Integrated Sustainability Strategy': {
                'metrics': ['BESG Environmental Pillar Score', 'BESG Social Pillar Score',
                         'BESG Governance Pillar Score', 'BESG ESG Score'],
                'direction': [1, 1, 1, 1],
                'initiative_type': 'sustainability_strategy'
            }
        }

        # Combine all actions
        all_actions = {}
        all_actions.update(env_actions)
        all_actions.update(soc_actions)
        all_actions.update(gov_actions)
        all_actions.update(int_actions)

        # Calibrate each action
        for action_name, action_info in all_actions.items():
            calibrated_effects[action_name] = {}

            for metric, direction in zip(action_info['metrics'], action_info['direction']):
                # Get typical change for this metric
                typical_change = self.get_typical_change(metric)

                # Use feature importance to scale the effect if available
                importance_factor = 1.0
                if metric in self.feature_importance.get('environmental', {}) and 'Environmental' in action_name:
                    importance_factor = self.feature_importance['environmental'].get(metric, 1.0) * 5
                elif metric in self.feature_importance.get('social', {}) and 'Social' in action_name:
                    importance_factor = self.feature_importance['social'].get(metric, 1.0) * 5
                elif metric in self.feature_importance.get('governance', {}) and 'Governance' in action_name:
                    importance_factor = self.feature_importance['governance'].get(metric, 1.0) * 5

                if typical_change:
                    # Use percentile depending on direction and effectiveness
                    if direction > 0:  # Want to increase
                        # Use p75 or p90 for positive changes
                        if 'p90' in typical_change:
                            change = typical_change['p90'] * importance_factor
                        elif 'p75' in typical_change:
                            change = typical_change['p75'] * importance_factor
                        else:
                            change = typical_change.get('max', 0.1) * importance_factor
                    else:  # Want to decrease
                        # Use p25 or p10 for negative changes
                        if 'p25' in typical_change:
                            change = -abs(typical_change['p25']) * importance_factor
                        else:
                            change = typical_change.get('min', -0.1) * importance_factor

                    # Make sure change is in the right direction
                    if (direction > 0 and change < 0) or (direction < 0 and change > 0):
                        change = -change

                    # Special scaling for ESG scores
                    if metric in self.esg_metrics:
                        # Make changes more impactful for direct action on a pillar
                        if metric == 'BESG Environmental Pillar Score' and action_name in env_actions:
                            change = max(change, 0.2)
                        elif metric == 'BESG Social Pillar Score' and action_name in soc_actions:
                            change = max(change, 0.2)
                        elif metric == 'BESG Governance Pillar Score' and action_name in gov_actions:
                            change = max(change, 0.2)
                        elif metric == 'BESG ESG Score':
                            # Overall ESG score changes should be smaller
                            change = min(change, 0.15)

                    calibrated_effects[action_name][metric] = change
                else:
                    # Default values if no data available, scaled by feature importance
                    if metric in self.esg_metrics:
                        # Default for ESG scores: modest improvements
                        calibrated_effects[action_name][metric] = 0.1 * direction
                    elif metric == 'Nitrogen Oxide Emissions':
                        calibrated_effects[action_name][metric] = -5000 * direction * importance_factor
                    elif metric in ['GHG Scope 1', 'GHG Scope 2 Location-Based', 'GHG Scope 3']:
                        # Default GHG reductions
                        if metric == 'GHG Scope 1':
                            calibrated_effects[action_name][metric] = -5000 * direction * importance_factor
                        elif metric == 'GHG Scope 2 Location-Based':
                            calibrated_effects[action_name][metric] = -4000 * direction * importance_factor
                        elif metric == 'GHG Scope 3':
                            calibrated_effects[action_name][metric] = -80000 * direction * importance_factor
                    elif metric == 'Renewable Energy Use':
                        calibrated_effects[action_name][metric] = 8 * direction * importance_factor
                    elif metric in ['Pct Women in Workforce', 'Pct Women in Senior Management']:
                        calibrated_effects[action_name][metric] = 3 * direction * importance_factor
                    elif metric == 'Number of Female Executives':
                        calibrated_effects[action_name][metric] = 1 * direction * importance_factor
                    elif metric == 'Number of Women on Board':
                        calibrated_effects[action_name][metric] = 1 * direction * importance_factor
                    elif metric == 'Total Recordable Incident Rate - Employees':
                        calibrated_effects[action_name][metric] = -0.5 * direction * importance_factor
                    elif metric == 'Fatalities - Employees':
                        calibrated_effects[action_name][metric] = -1 * direction * importance_factor
                    elif metric == 'Employee Turnover Pct':
                        calibrated_effects[action_name][metric] = -2 * direction * importance_factor
                    elif metric == 'Community Spending':
                        calibrated_effects[action_name][metric] = 500 * direction * importance_factor
                    elif metric == 'Number of Independent Directors':
                        calibrated_effects[action_name][metric] = 1 * direction * importance_factor
                    elif metric == 'Number of Non Executive Directors on Board':
                        calibrated_effects[action_name][metric] = 1 * direction * importance_factor
                    elif metric == 'Board Size':
                        # No direct change to board size for most actions
                        calibrated_effects[action_name][metric] = 0 * direction
                    elif metric == 'Years Auditor Employed':
                        calibrated_effects[action_name][metric] = -1 * direction * importance_factor
                    elif metric == 'Audit Committee Meetings':
                        calibrated_effects[action_name][metric] = 1 * direction * importance_factor
                    elif metric == 'Number of Board Meetings for the Year':
                        calibrated_effects[action_name][metric] = 2 * direction * importance_factor
                    elif metric == 'Compensation Committee Meeting Attendance %':
                        calibrated_effects[action_name][metric] = 5 * direction * importance_factor
                    elif metric == 'Board Duration (Years)':
                        # Typically changes slowly
                        calibrated_effects[action_name][metric] = 0.2 * direction * importance_factor
                    elif metric == 'Age of the Oldest Director' or metric == 'Age of the Youngest Director':
                        # Age generally doesn't change directly from actions
                        calibrated_effects[action_name][metric] = 0 * direction
                    else:
                        # Generic default
                        calibrated_effects[action_name][metric] = 0.1 * direction * importance_factor

            # Store the initiative type for financial impact reference
            calibrated_effects[action_name]['_initiative_type'] = action_info.get('initiative_type', 'renewable_investment')

        return calibrated_effects

    def generate_esg_insights(self, company=None):
        """Generate insights about ESG patterns and opportunities"""
        insights = []

        # If company is specified, generate company-specific insights
        if company is not None:
            # Check if company exists in the data
            if company not in self.data['Company'].values:
                insights.append(f"Company '{company}' not found in the dataset.")
                return insights

            company_data = self.data[self.data['Company'] == company].sort_values('Year')

            if len(company_data) == 0:
                insights.append(f"No data available for '{company}'.")
                return insights

            # Get latest year data
            latest_year = company_data['Year'].max()
            latest_data = company_data[company_data['Year'] == latest_year]

            # 1. Overall ESG performance
            if 'BESG ESG Score' in latest_data.columns:
                esg_score = latest_data['BESG ESG Score'].iloc[0]
                insights.append(f"Overall ESG Score: {esg_score:.2f} (out of 10)")

                # Compare to industry benchmarks
                industry_benchmark = self.get_industry_benchmark(company, 'BESG ESG Score')
                if industry_benchmark:
                    if esg_score < industry_benchmark['p25']:
                        insights.append("⚠️ ESG performance is in the BOTTOM 25% of industry peers.")
                    elif esg_score > industry_benchmark['p75']:
                        insights.append("✅ ESG performance is in the TOP 25% of industry peers.")
                    elif esg_score > industry_benchmark['p50']:
                        insights.append("ESG performance is ABOVE AVERAGE compared to industry peers.")
                    else:
                        insights.append("ESG performance is BELOW AVERAGE compared to industry peers.")

            # 2. Pillar analysis
            pillars = ['BESG Environmental Pillar Score', 'BESG Social Pillar Score', 'BESG Governance Pillar Score']
            available_pillars = [p for p in pillars if p in latest_data.columns]

            if available_pillars:
                pillar_scores = {p: latest_data[p].iloc[0] for p in available_pillars}
                best_pillar = max(pillar_scores.items(), key=lambda x: x[1])
                worst_pillar = min(pillar_scores.items(), key=lambda x: x[1])

                insights.append(f"Strongest pillar: {best_pillar[0].split(' ')[1]} ({best_pillar[1]:.2f})")
                insights.append(f"Weakest pillar: {worst_pillar[0].split(' ')[1]} ({worst_pillar[1]:.2f})")

                # Pillar-specific insights
                for pillar, score in pillar_scores.items():
                    name = pillar.split(' ')[1]
                    benchmark = self.get_industry_benchmark(company, pillar)

                    if benchmark:
                        if score < benchmark['p25']:
                            insights.append(f"⚠️ {name} performance is in the BOTTOM 25% of industry peers.")
                        elif score > benchmark['p75']:
                            insights.append(f"✅ {name} performance is in the TOP 25% of industry peers.")

            # 3. Historical trend analysis
            if len(company_data) > 1 and 'BESG ESG Score' in company_data.columns:
                esg_trend = company_data['BESG ESG Score'].tolist()
                first_score = esg_trend[0]
                last_score = esg_trend[-1]

                if last_score > first_score:
                    insights.append(f"✅ ESG Score has IMPROVED from {first_score:.2f} to {last_score:.2f} over {len(esg_trend)} years.")
                elif last_score < first_score:
                    insights.append(f"⚠️ ESG Score has DECLINED from {first_score:.2f} to {last_score:.2f} over {len(esg_trend)} years.")
                else:
                    insights.append(f"ESG Score has remained STABLE at {last_score:.2f} over {len(esg_trend)} years.")

                # Check for specific improvement areas
                for metric in self.environmental_metrics + self.social_metrics + self.governance_metrics:
                    if metric in company_data.columns:
                        values = company_data[metric].tolist()
                        if len(values) > 1 and not np.isnan(values).all():
                            first_val = values[0]
                            last_val = values[-1]

                            if not pd.isna(first_val) and not pd.isna(last_val):
                                if metric == 'Renewable Energy Use' and last_val > first_val + 5:
                                    insights.append(f"✅ Significant increase in renewable energy use from {first_val:.1f}% to {last_val:.1f}%.")
                                elif metric.startswith('GHG') and last_val < first_val * 0.85:
                                    insights.append(f"✅ Significant reduction in {metric} emissions.")
                                elif metric == 'Waste Recycled' and last_val > first_val + 10:
                                    insights.append(f"✅ Improved waste recycling from {first_val:.1f}% to {last_val:.1f}%.")
                                elif metric == 'Employee Turnover Pct' and last_val < first_val * 0.8:
                                    insights.append(f"✅ Reduced employee turnover from {first_val:.1f}% to {last_val:.1f}%.")

            # 4. Recommendations for improvement
            insights.append("\nRecommendations for improvement:")

            # Find weakest pillar for focused recommendations
            if available_pillars:
                weakest = worst_pillar[0].split(' ')[1].lower()
                if weakest == 'environmental':
                    if 'Renewable Energy Use' in latest_data.columns and latest_data['Renewable Energy Use'].iloc[0] < 20:
                        insights.append("- Increase renewable energy usage")
                    if any(m in latest_data.columns for m in ['GHG Scope 1', 'GHG Scope 2 Location-Based']):
                        insights.append("- Implement energy efficiency programs to reduce emissions")
                    if 'Waste Recycled' in latest_data.columns and latest_data['Waste Recycled'].iloc[0] < 50:
                        insights.append("- Improve waste recycling practices")
                elif weakest == 'social':
                    if 'Pct Women in Workforce' in latest_data.columns and latest_data['Pct Women in Workforce'].iloc[0] < 40:
                        insights.append("- Enhance diversity and inclusion initiatives")
                    if 'Employee Turnover Pct' in latest_data.columns and latest_data['Employee Turnover Pct'].iloc[0] > 15:
                        insights.append("- Implement employee retention programs")
                    insights.append("- Expand community engagement projects")
                elif weakest == 'governance':
                    if 'Number of Independent Directors' in latest_data.columns:
                        board_size = latest_data.get('Board Size', pd.Series([10])).iloc[0]
                        indep_dirs = latest_data['Number of Independent Directors'].iloc[0]
                        if indep_dirs / board_size < 0.5:
                            insights.append("- Increase board independence")
                    insights.append("- Strengthen ethics and compliance programs")
                    insights.append("- Improve stakeholder engagement policies")

            # 5. Similar companies to learn from
            similar_companies = self.get_similar_companies(company, top_n=3)
            if similar_companies:
                insights.append("\nSimilar companies for benchmarking:")
                for similar_company in similar_companies:
                    if similar_company in self.data['Company'].values:
                        similar_score = self.data[self.data['Company'] == similar_company]['BESG ESG Score'].max()
                        insights.append(f"- {similar_company} (ESG Score: {similar_score:.2f})")
        else:
            # Generate general insights about the dataset

            # 1. Overall ESG score distribution
            if 'BESG ESG Score' in self.data.columns:
                avg_score = self.data['BESG ESG Score'].mean()
                median_score = self.data['BESG ESG Score'].median()
                insights.append(f"Average ESG Score across all companies: {avg_score:.2f}")
                insights.append(f"Median ESG Score: {median_score:.2f}")

                # Top and bottom performers
                top_companies = self.data.sort_values('BESG ESG Score', ascending=False).drop_duplicates('Company').head(5)
                bottom_companies = self.data.sort_values('BESG ESG Score').drop_duplicates('Company').head(5)

                insights.append("\nTop 5 ESG performers:")
                for _, row in top_companies.iterrows():
                    insights.append(f"- {row['Company']}: {row['BESG ESG Score']:.2f}")

                insights.append("\nBottom 5 ESG performers:")
                for _, row in bottom_companies.iterrows():
                    insights.append(f"- {row['Company']}: {row['BESG ESG Score']:.2f}")

            # 2. ESG trends over time
            years = sorted(self.data['Year'].unique())
            if len(years) > 1 and 'BESG ESG Score' in self.data.columns:
                yearly_avgs = self.data.groupby('Year')['BESG ESG Score'].mean()

                first_year = years[0]
                last_year = years[-1]
                first_avg = yearly_avgs[first_year]
                last_avg = yearly_avgs[last_year]

                if last_avg > first_avg:
                    insights.append(f"\nOverall ESG performance has IMPROVED from {first_avg:.2f} to {last_avg:.2f} between {first_year} and {last_year}.")
                elif last_avg < first_avg:
                    insights.append(f"\nOverall ESG performance has DECLINED from {first_avg:.2f} to {last_avg:.2f} between {first_year} and {last_year}.")
                else:
                    insights.append(f"\nOverall ESG performance has remained STABLE between {first_year} and {last_year}.")

            # 3. Pillar comparisons
            pillars = ['BESG Environmental Pillar Score', 'BESG Social Pillar Score', 'BESG Governance Pillar Score']
            available_pillars = [p for p in pillars if p in self.data.columns]

            if available_pillars:
                pillar_avgs = {p.split(' ')[1]: self.data[p].mean() for p in available_pillars}
                sorted_pillars = sorted(pillar_avgs.items(), key=lambda x: x[1], reverse=True)

                insights.append("\nPillar performance across all companies:")
                for pillar, avg in sorted_pillars:
                    insights.append(f"- {pillar}: {avg:.2f}")

            # 4. Most effective ESG improvements
            env_actions = self.get_effective_actions_for_pillar('environmental')
            soc_actions = self.get_effective_actions_for_pillar('social')
            gov_actions = self.get_effective_actions_for_pillar('governance')

            insights.append("\nMost effective ESG improvements observed:")

            if env_actions:
                top_env = env_actions[0]
                insights.append(f"- Environmental: {top_env['company']} improved by {top_env['improvement']:.2f} points")
                for metric, change in top_env['metrics'][:2]:  # Show top 2 metrics
                    insights.append(f"  • {metric}: {'+' if change > 0 else ''}{change:.2f}")

            if soc_actions:
                top_soc = soc_actions[0]
                insights.append(f"- Social: {top_soc['company']} improved by {top_soc['improvement']:.2f} points")
                for metric, change in top_soc['metrics'][:2]:  # Show top 2 metrics
                    insights.append(f"  • {metric}: {'+' if change > 0 else ''}{change:.2f}")

            if gov_actions:
                top_gov = gov_actions[0]
                insights.append(f"- Governance: {top_gov['company']} improved by {top_gov['improvement']:.2f} points")
                for metric, change in top_gov['metrics'][:2]:  # Show top 2 metrics
                    insights.append(f"  • {metric}: {'+' if change > 0 else ''}{change:.2f}")

            # 5. ESG-Financial correlations
            if self.esg_financial_correlations:
                insights.append("\nESG-Financial performance correlations:")
                for esg_metric, correlations in self.esg_financial_correlations.items():
                    name = esg_metric.replace('BESG ', '')
                    for fin_metric, corr in correlations.items():
                        if abs(corr) > 0.2:  # Only show meaningful correlations
                            direction = "positive" if corr > 0 else "negative"
                            strength = "strong" if abs(corr) > 0.5 else "moderate"
                            insights.append(f"- {name} has a {strength} {direction} correlation ({corr:.2f}) with {fin_metric}")

            # 6. NEW: Add insights about improvement patterns
            if self.improvement_factors:
                insights.append("\nAnalysis of ESG Score Improvement Patterns:")
                sorted_factors = sorted(self.improvement_factors.items(), key=lambda x: x[1], reverse=True)

                # Find the most and least rewarding score ranges
                best_range = sorted_factors[0]
                worst_range = sorted_factors[-1]

                insights.append(f"- Companies with ESG scores between {best_range[0][0]}-{best_range[0][1]} show {best_range[1]:.1f}x faster improvements")
                insights.append(f"- Companies with ESG scores between {worst_range[0][0]}-{worst_range[0][1]} show {worst_range[1]:.1f}x slower improvements")

            # 7. NEW: Add insights about pillar balance
            insights.append("\nAnalysis of Pillar Balance Importance:")
            if self.pillar_balance_analysis['recommendation'] == 'balanced':
                insights.append(f"- Data shows BALANCED approach is {self.pillar_balance_analysis['balance_advantage']:.2f} points more effective")
                insights.append(f"- Recommendation: Focus on improving all pillars together")
            else:
                insights.append(f"- Data shows FOCUSED approach is {-self.pillar_balance_analysis['balance_advantage']:.2f} points more effective")
                insights.append(f"- Recommendation: Prioritize improvements in weakest pillar first")

        return insights


# Enhanced ESG Reinforcement Learning Environment
# This code contains the updated action space for the DataDrivenESGEnvironment class

class DataDrivenESGEnvironment(gym.Env):
    """Data-driven ESG Environment for ESG Score Optimization using real data-driven insights"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, company_idx=0, initial_year=2018, max_steps=10,
                 action_cost_factor=0.005, max_reward=10.0, scale_factor=0.1,
                 diversity_weight=3.0, balance_weight=2.5, verbose=False):
        super(DataDrivenESGEnvironment, self).__init__()

        self.data = data
        self.max_reward = max_reward
        self.scale_factor = scale_factor
        self.diversity_weight = diversity_weight  # Weight for action diversity rewards
        self.balance_weight = balance_weight  # Weight for pillar balance rewards
        self.verbose = verbose  # Add verbose flag to control debugging output

        # Create data analyzer
        self.analyzer = ESGDataAnalyzer(data, scale_factor=scale_factor)

        # Get list of unique companies
        self.companies = data['Company'].unique()

        # Make sure the company_idx is valid
        if company_idx >= len(self.companies):
            company_idx = 0
            print(f"Warning: company_idx out of range. Using company {self.companies[0]} instead.")

        self.company_idx = company_idx
        self.company = self.companies[company_idx]

        # Get company industry
        if 'Industry' in data.columns:
            self.industry = data[data['Company'] == self.company]['Industry'].iloc[0]
        else:
            self.industry = 'Unknown'

        # Get available years for this company
        company_years = data[data['Company'] == self.company]['Year'].unique()

        # Set initial year to the earliest available year if specified year not available
        if initial_year not in company_years:
            initial_year = min(company_years)
            print(f"Warning: initial_year not available for {self.company}. Using {initial_year} instead.")

        self.initial_year = initial_year
        self.current_year = initial_year
        self.max_steps = max_steps
        self.current_step = 0
        self.action_cost_factor = action_cost_factor

        # Enhanced tracking for strategy development
        self.action_counts = {i: 0 for i in range(60)}  # Updated to 60 actions
        self.pillar_counts = {'environmental': 0, 'social': 0, 'governance': 0, 'all': 0}
        self.last_actions = deque(maxlen=3)  # Track last 3 actions for pattern detection
        self.pillar_improvements = {'environmental': 0, 'social': 0, 'governance': 0}
        self.last_pillar_improved = None

        # STRICT action cooldown system - actions can't be used again for N steps
        self.action_cooldown = {i: 0 for i in range(60)}  # Updated to 60 actions
        self.cooldown_duration = 3  # Actions can't be used again for 3 steps

        # NEW: Track future financial effects
        self.future_financial_effects = []

        # Get initial state
        self.initial_state = self._get_state(self.company, self.initial_year)
        self.current_state = self.initial_state.copy()

        # Store the initial values for key metrics
        self.initial_values = {}
        for i, col in enumerate(self.state_cols):
            self.initial_values[col] = self.initial_state[i]

        # Define expanded action space (based on finalized actions)
        self.action_space = spaces.Discrete(60)  # Updated from 21 to 60 actions

        # Define observation space
        num_features = len(self.initial_state)
        self.observation_space = spaces.Box(
            low=np.zeros(num_features),
            high=np.ones(num_features) * float('inf'),
            dtype=np.float32
        )

        # Get data-driven calibrated action effects
        calibrated_effects = self.analyzer.calibrate_action_effects()

        # NEW: Get data-driven pillar balance recommendation
        pillar_balance_rec = self.analyzer.get_pillar_balance_recommendation()
        # Adjust balance weight based on analysis
        self.balance_weight = pillar_balance_rec['optimal_weight']
        print(f"Data-driven balance weight: {self.balance_weight:.2f} ({pillar_balance_rec['recommendation']} approach recommended)")

        # UPDATED: Finalized action definitions based on research
        self.actions = {
            # --- Foundational Action ---
            0: {'name': 'No Action',
                'effects': {},
                'cost': 0,
                'pillar': None,
                'initiative_type': 'no_action'},

            # --- Retained Original Environmental Actions ---
            3: {'name': 'Water Conservation Program',
                # Original effects assumed reasonable until specific research provided
                'effects': {'Total Water Withdrawal': -8000, 'BESG Environmental Pillar Score': 0.2},
                'cost': 70, # Original cost
                'pillar': 'environmental',
                'initiative_type': 'water_conservation'},
            4: {'name': 'Waste Reduction Initiative',
                # Original effects assumed reasonable until specific research provided
                'effects': {'Total Waste': -4000, 'Waste Recycled': 8, 'BESG Environmental Pillar Score': 0.18},
                'cost': 65, # Original cost
                'pillar': 'environmental',
                'initiative_type': 'waste_reduction'},
            6: {'name': 'Supply Chain Emissions Program (Scope 3)',
                 # Original effects assumed reasonable until specific research provided
                'effects': {'GHG Scope 3': -80000, 'BESG Environmental Pillar Score': 0.25},
                'cost': 85, # Original cost
                'pillar': 'environmental',
                'initiative_type': 'supply_chain_emissions'}, # Updated type

            # --- Retained Original Social Actions (Refined where applicable) ---
            9: {'name': 'Employee Wellbeing Initiative',
                # Original effects assumed reasonable (focus on turnover/wellbeing)
                'effects': {'Employee Turnover Pct': -2.5, 'Total Recordable Incident Rate - Employees': -0.3, 'BESG Social Pillar Score': 0.26},
                'cost': 45, # Original cost
                'pillar': 'social',
                'initiative_type': 'employee_wellbeing'},
           10: {'name': 'Community Engagement Project',
                # Original effects assumed reasonable (focus on community spending)
                'effects': {'Community Spending': 600, 'BESG Social Pillar Score': 0.31},
                'cost': 60, # Original cost
                'pillar': 'social',
                'initiative_type': 'community_engagement'}, # Updated type
           12: {'name': 'Workforce Development Training',
                # Original effects assumed reasonable (focus on skills/turnover)
                'effects': {'BESG Social Pillar Score': 0.24, 'Employee Turnover Pct': -1.5},
                'cost': 40, # Original cost
                'pillar': 'social',
                'initiative_type': 'employee_training'}, # Updated type

            # --- Retained Original Governance Actions (Refined based on research) ---
            14: {'name': 'Board: Improve Gender Diversity (Recruit Woman Director)',
                 'effects': {'Number of Women on Board': 1,
                             'Board Size': 1, # Assumes adding a director
                             'BESG Governance Pillar Score': 0.15}, # Adjusted bonus
                 'cost': 65, # Increased cost reflecting search fees (from 30)
                 'pillar': 'governance',
                 'initiative_type': 'board_diversity_recruitment'}, # Updated type
            16: {'name': 'Board: Improve Effectiveness (Meetings/Attendance)',
                 'effects': {'Number of Board Meetings for the Year': 1, # Adjusted effect
                             'Compensation Committee Meeting Attendance %': 5, # Kept original effect
                             'BESG Governance Pillar Score': 0.15}, # Adjusted bonus
                 'cost': 30, # Kept original cost
                 'pillar': 'governance',
                 'initiative_type': 'board_effectiveness'},
            17: {'name': 'Board: Executive Compensation Reform (Align Pay/Performance)',
                 'effects': {'Compensation Committee Meeting Attendance %': 5, # Adjusted effect
                             'BESG Governance Pillar Score': 0.16}, # Adjusted bonus
                 'cost': 30, # Kept original cost
                 'pillar': 'governance',
                 'initiative_type': 'compensation_reform'},
            18: {'name': 'Implement Company-Wide Ethics & Compliance Program',
                 'effects': {'BESG Governance Pillar Score': 0.15}, # Adjusted bonus
                 'cost': 38, # Kept original cost
                 'pillar': 'governance',
                 'initiative_type': 'ethics_compliance'},

            # --- Retained Original Integrated Actions ---
            19: {'name': 'Comprehensive ESG Disclosure Enhancement', # Renamed slightly
                 # Original effects assumed reasonable
                 'effects': {'ESG Disclosure Score': 0.5, 'Environmental Disclosure Score': 0.5, 'Social Disclosure Score': 0.5, 'Governance Disclosure Score': 0.5,
                             'BESG ESG Score': 0.12, 'BESG Environmental Pillar Score': 0.06, 'BESG Social Pillar Score': 0.06, 'BESG Governance Pillar Score': 0.06},
                 'cost': 75, # Original cost
                 'pillar': 'all',
                 'initiative_type': 'esg_disclosure'},
            20: {'name': 'Develop Integrated Sustainability Strategy', # Renamed slightly
                 # Original effects assumed reasonable
                 'effects': {'BESG Environmental Pillar Score': 0.09, 'BESG Social Pillar Score': 0.09, 'BESG Governance Pillar Score': 0.09, 'BESG ESG Score': 0.14},
                 'cost': 100, # Original cost
                 'pillar': 'all',
                 'initiative_type': 'sustainability_strategy'},

            # --- NEW/REVISED Environmental Actions (from Research 1, 2) ---
            # NOx Reduction Actions
            21: {'name': 'NOx Reduction: Low Excess Air (LEA) Tuning',
                 'effects': {'Nitrogen Oxide Emissions': -0.15, 'BESG Environmental Pillar Score': 0.05},
                 'cost': 5, 'pillar': 'environmental', 'initiative_type': 'nox_operational_tuning'},
            22: {'name': 'NOx Reduction: Install Low-NOx Burners (LNBs)',
                 'effects': {'Nitrogen Oxide Emissions': -0.40, 'BESG Environmental Pillar Score': 0.15},
                 'cost': 40, 'pillar': 'environmental', 'initiative_type': 'nox_combustion_retrofit'},
            23: {'name': 'NOx Reduction: Implement Staged Air Combustion',
                 'effects': {'Nitrogen Oxide Emissions': -0.35, 'BESG Environmental Pillar Score': 0.12},
                 'cost': 30, 'pillar': 'environmental', 'initiative_type': 'nox_combustion_modification'},
            24: {'name': 'NOx Reduction: Implement Staged Fuel Combustion (Fuel Reburning)',
                 'effects': {'Nitrogen Oxide Emissions': -0.60, 'BESG Environmental Pillar Score': 0.20},
                 'cost': 50, 'pillar': 'environmental', 'initiative_type': 'nox_combustion_modification'},
            25: {'name': 'NOx Reduction: Implement Flue Gas Recirculation (FGR)',
                 'effects': {'Nitrogen Oxide Emissions': -0.45, 'BESG Environmental Pillar Score': 0.15},
                 'cost': 20, 'pillar': 'environmental', 'initiative_type': 'nox_combustion_modification'},
            26: {'name': 'NOx Reduction: Install Selective Catalytic Reduction (SCR)',
                 'effects': {'Nitrogen Oxide Emissions': -0.85, 'BESG Environmental Pillar Score': 0.30},
                 'cost': 85, 'pillar': 'environmental', 'initiative_type': 'nox_post_combustion_treatment'},
            27: {'name': 'NOx Reduction: Install Selective Non-Catalytic Reduction (SNCR)',
                 'effects': {'Nitrogen Oxide Emissions': -0.55, 'BESG Environmental Pillar Score': 0.20},
                 'cost': 45, 'pillar': 'environmental', 'initiative_type': 'nox_post_combustion_treatment'},
            28: {'name': 'NOx Reduction: Fuel Switching (e.g., to Natural Gas)',
                 'effects': {'Nitrogen Oxide Emissions': -0.35, 'BESG Environmental Pillar Score': 0.15}, # Note: May impact GHG Scope 1/2 too
                 'cost': 35, 'pillar': 'environmental', 'initiative_type': 'nox_fuel_switch'},
            29: {'name': 'NOx Reduction: Water/Steam Injection (Gas Turbines/Boilers)',
                 'effects': {'Nitrogen Oxide Emissions': -0.75, 'BESG Environmental Pillar Score': 0.25},
                 'cost': 40, 'pillar': 'environmental', 'initiative_type': 'nox_combustion_modification'},
            # GHG Scope 2 Reduction Actions
            30: {'name': 'Scope 2 Reduction: Purchase Voluntary RECs (25% of Consumption)',
                 'effects': {'GHG Scope 2 Location-Based': -0.25, 'Renewable Energy Use': 0.25, 'BESG Environmental Pillar Score': 0.05},
                 'cost': 10, 'pillar': 'environmental', 'initiative_type': 'renewable_procurement_market'},
            31: {'name': 'Scope 2 Reduction: Purchase Compliance RECs (25% of Consumption)',
                 'effects': {'GHG Scope 2 Location-Based': -0.25, 'Renewable Energy Use': 0.25, 'BESG Environmental Pillar Score': 0.05},
                 'cost': 45, 'pillar': 'environmental', 'initiative_type': 'renewable_procurement_market'},
            32: {'name': 'Scope 2 Reduction: Enter Utility Green Tariff (100% of Consumption)',
                 'effects': {'GHG Scope 2 Location-Based': -1.0, 'Renewable Energy Use': 1.0, 'BESG Environmental Pillar Score': 0.15},
                 'cost': 30, 'pillar': 'environmental', 'initiative_type': 'renewable_procurement_utility'},
            33: {'name': 'Scope 2 Reduction: Sign Long-Term PPA (50% of Consumption)',
                 'effects': {'GHG Scope 2 Location-Based': -0.50, 'Renewable Energy Use': 0.50, 'BESG Environmental Pillar Score': 0.18},
                 'cost': 70, 'pillar': 'environmental', 'initiative_type': 'renewable_procurement_ppa'},
            34: {'name': 'Scope 2 Reduction: Install On-site Solar PV (20% of Consumption)',
                 'effects': {'GHG Scope 2 Location-Based': -0.20, 'Renewable Energy Use': 0.20, 'BESG Environmental Pillar Score': 0.15},
                 'cost': 80, 'pillar': 'environmental', 'initiative_type': 'renewable_generation_onsite'},
            # Energy Efficiency Actions
            35: {'name': 'Energy Efficiency: LED Lighting Retrofit',
                 'effects': {'GHG Scope 2 Location-Based': -0.08, 'BESG Environmental Pillar Score': 0.07},
                 'cost': 20, 'pillar': 'environmental', 'initiative_type': 'energy_efficiency_lighting'},
            36: {'name': 'Energy Efficiency: HVAC System Upgrade',
                 'effects': {'GHG Scope 2 Location-Based': -0.12, 'BESG Environmental Pillar Score': 0.10},
                 'cost': 55, 'pillar': 'environmental', 'initiative_type': 'energy_efficiency_hvac'},
            37: {'name': 'Energy Efficiency: Building Insulation Improvement',
                 'effects': {'GHG Scope 2 Location-Based': -0.05, 'BESG Environmental Pillar Score': 0.05},
                 'cost': 30, 'pillar': 'environmental', 'initiative_type': 'energy_efficiency_envelope'},
            38: {'name': 'Energy Efficiency: Implement Energy Management System (EMS)',
                 'effects': {'GHG Scope 2 Location-Based': -0.10, 'BESG Environmental Pillar Score': 0.08},
                 'cost': 40, 'pillar': 'environmental', 'initiative_type': 'energy_efficiency_monitoring'},
            39: {'name': 'Energy Efficiency: Industrial Process Optimization',
                 'effects': {'GHG Scope 2 Location-Based': -0.15, 'BESG Environmental Pillar Score': 0.12}, # Assumes Scope 2 impact
                 'cost': 60, 'pillar': 'environmental', 'initiative_type': 'energy_efficiency_process'},
            # Green Building Actions
            40: {'name': 'Pursue Green Building Certification (e.g., LEED/BREEAM)',
                 'effects': {'BESG Environmental Pillar Score': 0.10, 'ESG Disclosure Score': 0.05},
                 'cost': 50, 'pillar': 'environmental', 'initiative_type': 'green_building_certification'},

            # --- NEW/REVISED Social Actions (from Research 3, 4, 5, 6) ---
            # Refined ID 8
            8:  {'name': 'D&I Program: Increase Women in Workforce (Recruitment/Inclusion)',
                 'effects': {'Pct Women in Workforce': 0.02, 'BESG Social Pillar Score': 0.15},
                 'cost': 40, 'pillar': 'social', 'initiative_type': 'diversity_inclusion_workforce'},
            # Safety Actions
            41: {'name': 'Safety: Implement Behavior-Based Safety (BBS)',
                 'effects': {'Total Recordable Incident Rate - Employees': -0.50, 'BESG Social Pillar Score': 0.20},
                 'cost': 25, 'pillar': 'social', 'initiative_type': 'workplace_safety_bbs'},
            42: {'name': 'Safety: Implement Safety Management System (SMS - e.g., ISO 45001)',
                 'effects': {'Total Recordable Incident Rate - Employees': -0.50, 'Fatalities - Employees': -0.30, 'BESG Social Pillar Score': 0.25},
                 'cost': 60, 'pillar': 'social', 'initiative_type': 'workplace_safety_sms'},
            43: {'name': 'Safety: Implement Hazard Control & Prevention Program',
                 'effects': {'Total Recordable Incident Rate - Employees': -0.40, 'Fatalities - Employees': -0.20, 'BESG Social Pillar Score': 0.15},
                 'cost': 45, 'pillar': 'social', 'initiative_type': 'workplace_safety_hazard_control'},
            44: {'name': 'Safety: Enhanced Fatality Prevention Program (Targeted CAPEX/OPEX)',
                 'effects': {'Fatalities - Employees': -0.50, 'Total Recordable Incident Rate - Employees': -0.25, 'BESG Social Pillar Score': 0.30},
                 'cost': 75, 'pillar': 'social', 'initiative_type': 'workplace_safety_fatality_focus'},
            # Female Leadership Actions
            45: {'name': 'D&I: Implement Leadership Development Programs (LDPs) for Women',
                 'effects': {'Pct Women in Senior Management': 0.015, 'Number of Female Executives': 0.5, 'BESG Social Pillar Score': 0.10},
                 'cost': 35, 'pillar': 'social', 'initiative_type': 'diversity_inclusion_leadership_ldp'},
            46: {'name': 'D&I: Implement Formal Succession Planning for Gender Diversity',
                 'effects': {'Pct Women in Senior Management': 0.02, 'Number of Female Executives': 0.7, 'BESG Social Pillar Score': 0.12},
                 'cost': 40, 'pillar': 'social', 'initiative_type': 'diversity_inclusion_leadership_succession'},
            47: {'name': 'D&I: Establish Formal Mentorship Programs for Women',
                 'effects': {'Pct Women in Senior Management': 0.01, 'Number of Female Executives': 0.4, 'BESG Social Pillar Score': 0.08},
                 'cost': 20, 'pillar': 'social', 'initiative_type': 'diversity_inclusion_leadership_mentorship'},

            # --- NEW/REVISED Governance Actions (from Research 7, 8, 9, 10, 11, 12) ---
            # Refined ID 13
            13: {'name': 'Board: Increase Independence (Recruit Independent Director)',
                 'effects': {'Number of Independent Directors': 1, 'Number of Non Executive Directors on Board': 1, 'Board Size': 1, 'BESG Governance Pillar Score': 0.20},
                 'cost': 70, 'pillar': 'governance', 'initiative_type': 'board_independence_recruitment'},
            # Refined ID 15
            15: {'name': 'Board: Implement Auditor Rotation Policy (Partner/Firm)', # Refocused
                 'effects': {'Years Auditor Employed': -5, 'BESG Governance Pillar Score': 0.10}, # Reset/reduce tenure, score bonus for policy
                 'cost': 50, # Reflects transition cost/disruption
                 'pillar': 'governance',
                 'initiative_type': 'audit_rotation_policy'}, # Updated type
            # Board Structure/Policy Actions
            48: {'name': 'Board: Separate CEO & Chair Roles (or Appoint Lead Independent Director)',
                 'effects': {'BESG Governance Pillar Score': 0.15},
                 'cost': 10, 'pillar': 'governance', 'initiative_type': 'board_structure_role_separation'},
            49: {'name': 'Board: Implement Director Term Limits Policy', # Added 'Policy'
                 'effects': {'BESG Governance Pillar Score': 0.10},
                 'cost': 5, 'pillar': 'governance', 'initiative_type': 'board_tenure_term_limits'},
            # Board Size Actions
            50: {'name': 'Board: Increase Size Strategically (+1 Independent Director)', # Clarified who is added
                 'effects': {'Board Size': 1, 'Number of Independent Directors': 1, 'Number of Non Executive Directors on Board': 1, 'BESG Governance Pillar Score': 0.05},
                 'cost': 70, 'pillar': 'governance', 'initiative_type': 'board_size_increase'},
            51: {'name': 'Board: Decrease Size Strategically (-1 Director)',
                 'effects': {'Board Size': -1, 'BESG Governance Pillar Score': 0.02}, # Assumes efficiency gain, actual impact state-dependent
                 'cost': 0, # Savings handled by reward function
                 'pillar': 'governance', 'initiative_type': 'board_size_decrease'},
            # New Diversity Policy Actions (Research 9)
            52: {'name': 'Board: Implement Formal Board Diversity Policy (Targets)',
                 'effects': {'BESG Governance Pillar Score': 0.12, 'ESG Disclosure Score': 0.05},
                 'cost': 30, 'pillar': 'governance', 'initiative_type': 'board_diversity_policy'},
            53: {'name': 'Board: Engage Stakeholders/Advocacy Groups on Diversity',
                 'effects': {'BESG Governance Pillar Score': 0.08},
                 'cost': 20, 'pillar': 'governance', 'initiative_type': 'board_diversity_stakeholder'},
            # New Audit Committee Action (Research 10)
            54: {'name': 'Board: Enhance Audit Committee Oversight (Expertise/AQIs)',
                 'effects': {'Audit Committee Meetings': 1, 'BESG Governance Pillar Score': 0.15},
                 'cost': 35, 'pillar': 'governance', 'initiative_type': 'audit_committee_enhancement'},
            # New Board Process Actions (Research 11)
            55: {'name': 'Board: Implement Skills-Based Recruitment Policy',
                 'effects': {'BESG Governance Pillar Score': 0.05},
                 'cost': 15, 'pillar': 'governance', 'initiative_type': 'board_recruitment_skills_policy'},
            56: {'name': 'Board: Empower Nomination Committee (Composition/Evaluation)',
                 'effects': {'BESG Governance Pillar Score': 0.10},
                 'cost': 10, 'pillar': 'governance', 'initiative_type': 'board_committee_nomination_focus'},
            # New Tenure/Succession Actions (Research 12)
            57: {'name': 'Board: Implement Mandatory Retirement Age Policy',
                 'effects': {'BESG Governance Pillar Score': 0.08},
                 'cost': 5, 'pillar': 'governance', 'initiative_type': 'board_tenure_retirement_policy'},
            58: {'name': 'Board: Implement Overboarding Policy',
                 'effects': {'BESG Governance Pillar Score': 0.07},
                 'cost': 5, 'pillar': 'governance', 'initiative_type': 'board_policy_overboarding'},
            59: {'name': 'Board: Develop Formal Board Succession Plan',
                 'effects': {'BESG Governance Pillar Score': 0.15},
                 'cost': 25, 'pillar': 'governance', 'initiative_type': 'board_succession_planning'},
        }

        # Get industry benchmarks for the company
        self.industry_benchmarks = {}
        for metric in ['BESG ESG Score', 'BESG Environmental Pillar Score',
                      'BESG Social Pillar Score', 'BESG Governance Pillar Score']:
            benchmark = self.analyzer.get_industry_benchmark(self.company, metric)
            if benchmark:
                self.industry_benchmarks[metric] = benchmark

        for i in range(60):
            if i not in self.actions:
                self.actions[i] = {
                    'name': f'No Action (Placeholder {i})',
                    'effects': {},
                    'cost': 0,
                    'pillar': None,
                    'initiative_type': 'no_action'
                }

        # Generate insights about the company
        self.company_insights = self.analyzer.generate_esg_insights(self.company)

        # Save similar companies for reference
        self.similar_companies = self.analyzer.get_similar_companies(self.company)

    def reset(self):
        """Reset the environment to the initial state"""
        self.current_state = self.initial_state.copy()
        self.current_step = 0
        # Reset action counts and tracking
        self.action_counts = {i: 0 for i in range(60)}  # All 60 possible actions
        self.pillar_counts = {'environmental': 0, 'social': 0, 'governance': 0, 'all': 0}
        self.last_actions = deque(maxlen=3)
        self.pillar_improvements = {'environmental': 0, 'social': 0, 'governance': 0}
        self.last_pillar_improved = None
        self.action_cooldown = {i: 0 for i in range(60)}  # All 60 possible actions
        self.future_financial_effects = []
        return self.current_state

    def _get_state(self, company, year):
        """Get the state for a specific company and year with enhanced metrics"""
        # Filter data for the specific company and year
        company_data = self.data[(self.data['Company'] == company) & (self.data['Year'] == year)]

        if company_data.empty:
            # If no data for this combination, use the latest available data
            company_data = self.data[self.data['Company'] == company].sort_values('Year', ascending=False).iloc[0:1]
            if company_data.empty:
                # If still no data, use a default set of values
                print(f"Warning: No data found for {company}. Using default values.")
                # Create a default row with zeros
                company_data = pd.DataFrame({col: [0] for col in self.data.columns})
                company_data['Company'] = company
                company_data['Year'] = year

        # Updated state columns based on feature importance
        state_cols = [
            # Core ESG metrics
            'BESG ESG Score', 'BESG Environmental Pillar Score', 'BESG Social Pillar Score',
            'BESG Governance Pillar Score', 'ESG Disclosure Score', 'Environmental Disclosure Score',
            'Social Disclosure Score', 'Governance Disclosure Score',

            # Top environmental metrics
            'Nitrogen Oxide Emissions', 'GHG Scope 1', 'GHG Scope 2 Location-Based',
            'GHG Scope 3', 'Total Water Withdrawal', 'Total Waste', 'Waste Recycled',
            'Renewable Energy Use',

            # Top social metrics
            'Total Recordable Incident Rate - Employees', 'Fatalities - Employees',
            'Pct Women in Workforce', 'Pct Women in Senior Management', 'Employee Turnover Pct',
            'Community Spending', 'Number of Female Executives',

            # Top governance metrics
            'Number of Independent Directors', 'Board Size', 'Number of Women on Board',
            'Years Auditor Employed', 'Number of Non Executive Directors on Board',
            'Board Duration (Years)', 'Number of Executives / Company Managers',
            'Age of the Oldest Director', 'Number of Board Meetings for the Year',
            'Audit Committee Meetings', 'Compensation Committee Meeting Attendance %',
            'Age of the Youngest Director',

            # Financial metrics
            'Revenue, Adj', 'Net Income, Adj', 'Market Cap ($M)'
        ]

        # Ensure all columns exist in the data and have numeric values
        available_cols = []
        values = []

        for col in state_cols:
            if col in company_data.columns:
                val = company_data[col].values[0]
                # Check if value is numeric
                try:
                    val = float(val)
                    # Replace NaN values with zeros
                    if np.isnan(val):
                        val = 0.0

                    # Apply scaling to ESG-related scores to convert from 0-100 to 0-10
                    if col in ['BESG ESG Score', 'BESG Environmental Pillar Score', 'BESG Social Pillar Score',
                              'BESG Governance Pillar Score', 'ESG Disclosure Score',
                              'Environmental Disclosure Score', 'Social Disclosure Score', 'Governance Disclosure Score']:
                        if val > 10:  # Only scale if it looks like it's on a 0-100 scale
                            val = val * self.scale_factor  # Scale to 0-10 range

                    available_cols.append(col)
                    values.append(val)
                except (ValueError, TypeError):
                    # Skip columns with non-numeric values
                    pass

        # Create state array with available numeric columns
        state = np.array(values)

        # Store the column names for reference
        self.state_cols = available_cols

        return state

    def _load_feature_importance(self):
        """Load feature importance from CSV files for each pillar"""
        # Initialize dictionaries to store feature importance for each pillar
        self.env_feature_importance = {}
        self.social_feature_importance = {}
        self.gov_feature_importance = {}

        # Define file paths
        env_file = "var_impt/feature_importances_BESG Environmental Pillar Score.csv"
        social_file = "var_impt/feature_importances_BESG Social Pillar Score.csv"
        gov_file = "var_impt/feature_importances_BESG Governance Pillar Score.csv"

        # Load Environmental pillar feature importance
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        feature = parts[0].replace('num__', '')  # Remove 'num__' prefix
                        importance = float(parts[1])
                        self.env_feature_importance[feature] = importance
        except FileNotFoundError:
            print(f"Warning: Environmental feature importance file {env_file} not found")

        # Load Social pillar feature importance
        try:
            with open(social_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        feature = parts[0].replace('num__', '')
                        importance = float(parts[1])
                        self.social_feature_importance[feature] = importance
        except FileNotFoundError:
            print(f"Warning: Social feature importance file {social_file} not found")

        # Load Governance pillar feature importance
        try:
            with open(gov_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        feature = parts[0].replace('num__', '')
                        importance = float(parts[1])
                        self.gov_feature_importance[feature] = importance
        except FileNotFoundError:
            print(f"Warning: Governance feature importance file {gov_file} not found")


    def _calculate_esg_score(self, state_dict):
        """Calculate ESG score based on various metrics - tailored to industry with feature importance"""
        # If ESG score is directly provided and not being modified, use it
        if 'BESG ESG Score' in state_dict and state_dict.get('_esg_modified', False) is False:
            return state_dict['BESG ESG Score']

        # Updated weights based on new data
        env_weight = 0.485  # Environmental weight
        soc_weight = 0.292  # Social weight
        gov_weight = 0.237  # Governance weight

        # Environmental component with feature importance
        env_factors = {}
        if hasattr(self.analyzer, 'feature_importance') and 'environmental' in self.analyzer.feature_importance:
            for feature, importance in self.analyzer.feature_importance['environmental'].items():
                if feature in state_dict:
                    # Determine if feature has negative impact (like emissions)
                    is_negative = feature.startswith('GHG') or 'Emissions' in feature or 'Waste' in feature or 'Water' in feature
                    env_factors[feature] = -importance if is_negative else importance

        # Add base weight for pillar score if not already included
        if 'BESG Environmental Pillar Score' not in env_factors and 'BESG Environmental Pillar Score' in state_dict:
            env_factors['BESG Environmental Pillar Score'] = 0.5

        # Social component with feature importance
        social_factors = {}
        if hasattr(self.analyzer, 'feature_importance') and 'social' in self.analyzer.feature_importance:
            for feature, importance in self.analyzer.feature_importance['social'].items():
                if feature in state_dict:
                    # Determine if feature has negative impact
                    is_negative = feature == 'Employee Turnover Pct' or 'Incident Rate' in feature or 'Fatalities' in feature
                    social_factors[feature] = -importance if is_negative else importance

        # Add base weight for pillar score if not already included
        if 'BESG Social Pillar Score' not in social_factors and 'BESG Social Pillar Score' in state_dict:
            social_factors['BESG Social Pillar Score'] = 0.6

        # Governance component with feature importance
        gov_factors = {}
        if hasattr(self.analyzer, 'feature_importance') and 'governance' in self.analyzer.feature_importance:
            for feature, importance in self.analyzer.feature_importance['governance'].items():
                if feature in state_dict:
                    # Most governance features are positive except for auditor employed years (want independence)
                    is_negative = feature == 'Years Auditor Employed'
                    gov_factors[feature] = -importance if is_negative else importance

        # Add base weight for pillar score if not already included
        if 'BESG Governance Pillar Score' not in gov_factors and 'BESG Governance Pillar Score' in state_dict:
            gov_factors['BESG Governance Pillar Score'] = 0.7

        # Fallback to original factors if feature importance dictionaries are empty
        if not env_factors:
            env_factors = {
                'BESG Environmental Pillar Score': 0.5,
                'Renewable Energy Use': 0.15,
                'GHG Scope 1': -0.1,
                'GHG Scope 2 Location-Based': -0.1,
                'GHG Scope 3': -0.05,
                'Total Water Withdrawal': -0.05,
                'Total Waste': -0.03,
                'Waste Recycled': 0.02
            }

        if not social_factors:
            social_factors = {
                'BESG Social Pillar Score': 0.6,
                'Pct Women in Workforce': 0.15,
                'Pct Women in Senior Management': 0.15,
                'Employee Turnover Pct': -0.1,
                'Community Spending': 0.1
            }

        if not gov_factors:
            gov_factors = {
                'BESG Governance Pillar Score': 0.7,
                'Board Size': 0.1,
                'Number of Independent Directors': 0.2
            }

        # Normalize each component
        env_score = 0
        env_weight_sum = 0
        social_score = 0
        social_weight_sum = 0
        gov_score = 0
        gov_weight_sum = 0

        # Calculate environmental score using feature importance
        for factor, weight in env_factors.items():
            if factor in state_dict:
                value = state_dict[factor]
                abs_weight = abs(weight)
                env_weight_sum += abs_weight

                # Normalize negative factors (lower is better)
                if weight < 0:
                    # Normalize based on typical ranges
                    if factor == 'Nitrogen Oxide Emissions':
                        value = max(0, min(1, 1 - value / 20000))
                    elif factor == 'GHG Scope 1':
                        value = max(0, min(1, 1 - value / 100000))
                    elif factor == 'GHG Scope 2 Location-Based':
                        value = max(0, min(1, 1 - value / 50000))
                    elif factor == 'GHG Scope 3':
                        value = max(0, min(1, 1 - value / 1000000))
                    elif factor == 'Total Water Withdrawal':
                        value = max(0, min(1, 1 - value / 1000000))
                    elif factor == 'Total Waste':
                        value = max(0, min(1, 1 - value / 100000))
                    env_score += value * abs_weight
                else:
                    # Normalize positive factors (higher is better)
                    if factor == 'BESG Environmental Pillar Score':
                        value = value / 10  # Adjusted for 0-10 scale
                    elif factor == 'Renewable Energy Use':
                        value = value / 100
                    elif factor == 'Waste Recycled':
                        value = value / 100
                    env_score += value * weight

        # Calculate social score using feature importance
        for factor, weight in social_factors.items():
            if factor in state_dict:
                value = state_dict[factor]
                abs_weight = abs(weight)
                social_weight_sum += abs_weight

                # Normalize negative factors
                if weight < 0:
                    if factor == 'Employee Turnover Pct':
                        value = max(0, min(1, 1 - value / 50))
                    elif factor == 'Total Recordable Incident Rate - Employees':
                        value = max(0, min(1, 1 - value / 5))
                    elif factor == 'Fatalities - Employees':
                        value = max(0, min(1, 1 - value / 3))
                    social_score += value * abs_weight
                else:
                    # Normalize positive factors
                    if factor == 'BESG Social Pillar Score':
                        value = value / 10  # Adjusted for 0-10 scale
                    elif factor in ['Pct Women in Workforce', 'Pct Women in Senior Management', 'Compensation Committee Meeting Attendance %']:
                        value = value / 100
                    elif factor == 'Community Spending':
                        value = min(1, value / 5000)
                    elif factor == 'Number of Female Executives':
                        value = min(1, value / 5)
                    social_score += value * weight

        # Calculate governance score using feature importance
        for factor, weight in gov_factors.items():
            if factor in state_dict:
                value = state_dict[factor]
                abs_weight = abs(weight)
                gov_weight_sum += abs_weight

                # Normalize factors
                if weight < 0:
                    if factor == 'Years Auditor Employed':
                        value = max(0, min(1, 1 - value / 20))  # Lower is better for auditor tenure
                    gov_score += value * abs_weight
                else:
                    if factor == 'BESG Governance Pillar Score':
                        value = value / 10  # Adjusted for 0-10 scale
                    elif factor == 'Board Size':
                        value = min(1, value / 20)  # Assuming max board size of 20
                    elif factor == 'Number of Independent Directors':
                        board_size = state_dict.get('Board Size', 10)
                        value = value / board_size if board_size > 0 else 0
                    elif factor == 'Number of Women on Board':
                        board_size = state_dict.get('Board Size', 10)
                        value = value / board_size if board_size > 0 else 0
                    elif factor == 'Number of Non Executive Directors on Board':
                        board_size = state_dict.get('Board Size', 10)
                        value = value / board_size if board_size > 0 else 0
                    elif factor == 'Compensation Committee Meeting Attendance %':
                        value = value / 100
                    elif factor == 'Number of Board Meetings for the Year':
                        value = min(1, value / 12)
                    elif factor == 'Audit Committee Meetings':
                        value = min(1, value / 8)
                    elif factor == 'Number of Executives / Company Managers':
                        value = min(1, value / 20)
                    gov_score += value * weight

        # Normalize component scores
        if env_weight_sum > 0:
            env_score = env_score / env_weight_sum
        if social_weight_sum > 0:
            social_score = social_score / social_weight_sum
        if gov_weight_sum > 0:
            gov_score = gov_score / gov_weight_sum

        # FIX: First calculate a simplified ESG score based directly on pillar scores for validation
        env_pillar = state_dict.get('BESG Environmental Pillar Score', 0) / 10  # Normalize to 0-1
        soc_pillar = state_dict.get('BESG Social Pillar Score', 0) / 10
        gov_pillar = state_dict.get('BESG Governance Pillar Score', 0) / 10

        # Direct weighted average of pillar scores
        simplified_esg = (env_weight * env_pillar + soc_weight * soc_pillar + gov_weight * gov_pillar) * 10  # Scale back to 0-10

        # Calculate final ESG score (0-10) with updated pillar weightings
        esg_score = (env_weight * env_score + soc_weight * social_score + gov_weight * gov_score) * 10  # Scaled to 0-10

        # FIX: Ensure the ESG score doesn't deviate too far from the simplified direct calculation
        # This guards against complex feature calculations resulting in unrealistic ESG scores
        max_deviation = 0.2  # Maximum 20% deviation allowed
        if esg_score > simplified_esg * (1 + max_deviation):
            esg_score = simplified_esg * (1 + max_deviation)
        elif esg_score < simplified_esg * (1 - max_deviation):
            esg_score = simplified_esg * (1 - max_deviation)

        # Debug info if verbose mode is enabled
        if hasattr(self, 'verbose') and self.verbose:
            print(f"ESG Calculation - Env: {env_pillar*10:.2f}, Soc: {soc_pillar*10:.2f}, Gov: {gov_pillar*10:.2f}")
            print(f"Simplified ESG: {simplified_esg:.2f}, Full ESG: {esg_score:.2f}")

        return max(0, min(10, esg_score))  # Ensure score is in 0-10 range


    def _calculate_pillar_imbalance(self, state_dict):
        """Calculate the imbalance between ESG pillars"""
        env_score = state_dict.get('BESG Environmental Pillar Score', 0)
        soc_score = state_dict.get('BESG Social Pillar Score', 0)
        gov_score = state_dict.get('BESG Governance Pillar Score', 0)

        # Calculate the standard deviation as a measure of imbalance
        scores = [env_score, soc_score, gov_score]
        mean_score = sum(scores) / len(scores)
        variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
        std_dev = variance ** 0.5

        # Return imbalance score (higher means more imbalanced)
        return std_dev

    def _identify_weakest_pillar(self, state_dict):
        """Identify the weakest pillar that needs improvement"""
        pillar_scores = {
            'environmental': state_dict.get('BESG Environmental Pillar Score', 0),
            'social': state_dict.get('BESG Social Pillar Score', 0),
            'governance': state_dict.get('BESG Governance Pillar Score', 0)
        }

        # Return the pillar with the lowest score
        return min(pillar_scores.items(), key=lambda x: x[1])[0]

    def _compare_to_benchmark(self, metric, value):
        """Compare a metric value to industry benchmarks"""
        if metric in self.industry_benchmarks:
            benchmark = self.industry_benchmarks[metric]

            if value < benchmark['p25']:
                return 'bottom_25'
            elif value > benchmark['p75']:
                return 'top_25'
            elif value > benchmark['p50']:
                return 'above_average'
            else:
                return 'below_average'

        return None

    def _is_action_on_cooldown(self, action):
        """Check if an action is on cooldown and cannot be used"""
        return self.action_cooldown[action] > 0

    def _apply_action_effects(self, action):
        """Apply the effects of an action to the current state with data-driven calibration"""
        # Convert current state to dictionary for easier manipulation
        state_dict = {self.state_cols[i]: self.current_state[i] for i in range(len(self.state_cols))}

        # Flag to track if ESG score needs recalculation
        state_dict['_esg_modified'] = False

        # Get action effects
        action_info = self.actions[action]
        effects = action_info['effects']
        base_cost = action_info.get('cost', 0)
        pillar = action_info.get('pillar', None)
        initiative_type = action_info.get('initiative_type', 'renewable_investment')

        # Track for enhanced diminishing returns
        action_count = self.action_counts.get(action, 0)

        # FIXED: Much stronger diminishing returns calculation
        diminishing_factor = 1.0
        if action_count > 0:
            # Exponential diminishing returns for repeated actions (now much stronger)
            diminishing_factor = max(0.05, 0.5 ** action_count)

        # NEW: Apply data-driven improvement factor based on current ESG score
        if 'BESG ESG Score' in state_dict:
            current_esg = state_dict['BESG ESG Score']
            # FIXED: Get improvement factor with the safety bounds
            improvement_factor = self.analyzer.get_improvement_factor(current_esg)
        else:
            improvement_factor = 1.0

        # FIXED: Add a debug message with reasonable formatting
        if self.verbose:
            print(f"Action: {action_info['name']}, Factor: {improvement_factor:.2f}, Current ESG: {current_esg:.2f}")

        # Apply effects to state with diminishing returns and improvement factor
        applied_effects = {}
        for metric, change in effects.items():
            if metric in state_dict:
                # Skip initiative type metadata
                if metric.startswith('_'):
                    continue

                # FIXED: Apply diminishing returns to the effect
                adjusted_change = change * diminishing_factor

                # FIXED: More reasonable improvement factor application
                if metric in ['BESG ESG Score', 'BESG Environmental Pillar Score',
                           'BESG Social Pillar Score', 'BESG Governance Pillar Score']:
                    # Limit the effect of the improvement factor
                    adjusted_change = adjusted_change * min(1.5, improvement_factor)

                # Store the original value for tracking improvement
                original_value = state_dict[metric]

                # Apply the effect
                state_dict[metric] += adjusted_change
                applied_effects[metric] = (original_value, state_dict[metric], adjusted_change)

                # If we've modified a key metric, flag for ESG recalculation
                if metric != 'BESG ESG Score':
                    state_dict['_esg_modified'] = True

                # Apply constraints to ensure realistic values
                if metric in ['BESG Environmental Pillar Score', 'BESG Social Pillar Score', 'BESG Governance Pillar Score']:
                    # FIXED: Incremental approach for pillar scores - max change of 0.25 per action
                    max_change = 0.25
                    if adjusted_change > max_change:
                        state_dict[metric] = original_value + max_change
                    elif adjusted_change < -max_change:
                        state_dict[metric] = original_value - max_change
                    # Constrain to 0-10 range
                    state_dict[metric] = max(0, min(10, state_dict[metric]))

        # Calculate financial impact with data-driven calibration
        adjusted_cost = base_cost * diminishing_factor

        # Apply data-driven financial impact
        financial_impact = 0
        if action != 0:  # No financial impact for "No Action"
            try:
                # Get the initiative type with fallback
                initiative_type = action_info.get('initiative_type', 'no_action')

                # Get financial impact data from analyzer with validation
                impact = self.analyzer.get_financial_impact(initiative_type, 'immediate')

                # Extensive error checking for impact value
                if impact is None or np.isnan(impact) or np.isinf(impact):
                    # Use action cost to calculate a default percentage impact
                    default_pct = -0.01  # Default 1% negative impact
                    impact = default_pct
                    if self.verbose:
                        print(f"Warning: Invalid financial impact for {initiative_type}, using default {default_pct}")

                if 'Net Income, Adj' in state_dict:
                    net_income = state_dict['Net Income, Adj']

                    # Calculate impact, handling various edge cases
                    if net_income != 0:
                        # Apply the financial impact based on percentage change
                        # Cap extreme impact values for realism
                        capped_impact = max(-0.25, min(0.1, impact))  # Limit between -25% and +10%
                        financial_impact = net_income * capped_impact
                    else:
                        # If net income is 0, use the action cost as direct impact
                        financial_impact = -adjusted_cost

                    # Apply the impact to the state
                    state_dict['Net Income, Adj'] += financial_impact

                    # Add future effects to track (with validation)
                    try:
                        year1_impact = self.analyzer.get_financial_impact(initiative_type, 'year1')
                        year2_impact = self.analyzer.get_financial_impact(initiative_type, 'year2')

                        # Validate year1 impact
                        if year1_impact is not None and not (np.isnan(year1_impact) or np.isinf(year1_impact)):
                            self.future_financial_effects.append({
                                'year': self.current_year + self.current_step + 1,
                                'metric': 'Net Income, Adj',
                                'impact': max(-0.15, min(0.1, year1_impact)),  # Cap between -15% and +10%
                                'base_value': net_income
                            })

                        # Validate year2 impact
                        if year2_impact is not None and not (np.isnan(year2_impact) or np.isinf(year2_impact)):
                            self.future_financial_effects.append({
                                'year': self.current_year + self.current_step + 2,
                                'metric': 'Net Income, Adj',
                                'impact': max(-0.1, min(0.15, year2_impact)),  # Cap between -10% and +15%
                                'base_value': net_income
                            })
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: Error adding future financial effects: {e}")
                else:
                    # If Net Income isn't in the state, we still return the cost as impact
                    financial_impact = -adjusted_cost
                    if self.verbose:
                        print("Warning: Net Income not found in state, using action cost as impact")
            except Exception as e:
                # Fallback if any error occurs in financial impact calculation
                if self.verbose:
                    print(f"Error in financial impact calculation: {e}")
                financial_impact = -adjusted_cost  # Default to basic cost
                if 'Net Income, Adj' in state_dict:
                    state_dict['Net Income, Adj'] -= adjusted_cost

            # Final validation - ensure financial_impact is a valid number
            if np.isnan(financial_impact) or np.isinf(financial_impact):
                financial_impact = -adjusted_cost
                if 'Net Income, Adj' in state_dict:
                    # Fix the state as well
                    state_dict['Net Income, Adj'] = state_dict['Net Income, Adj'] - adjusted_cost + financial_impact

        # Apply any future financial effects that are due this year
        current_sim_year = self.current_year + self.current_step
        valid_future_effects = []

        for effect in self.future_financial_effects:
            if effect['year'] == current_sim_year:
                try:
                    if effect['metric'] in state_dict:
                        base_value = effect['base_value']
                        impact = effect['impact']

                        # Validate impact value
                        if np.isnan(impact) or np.isinf(impact):
                            continue

                        if base_value != 0:
                            impact_value = base_value * impact
                            state_dict[effect['metric']] += impact_value

                            # Add to financial_impact for reporting
                            if effect['metric'] == 'Net Income, Adj':
                                financial_impact += impact_value
                except Exception as e:
                    if self.verbose:
                        print(f"Error applying future financial effect: {e}")
            else:
                # Keep effects for future years
                valid_future_effects.append(effect)

        # Update the future effects list with only valid future entries
        self.future_financial_effects = valid_future_effects

        # Track pillar improvements
        if pillar in ['environmental', 'social', 'governance']:
            # Store which pillar was last improved for diversity rewards
            self.last_pillar_improved = pillar
            self.pillar_counts[pillar] += 1

            # Track specific improvements by pillar
            if pillar == 'environmental' and 'BESG Environmental Pillar Score' in applied_effects:
                _, new_val, change = applied_effects['BESG Environmental Pillar Score']
                self.pillar_improvements['environmental'] += change
            elif pillar == 'social' and 'BESG Social Pillar Score' in applied_effects:
                _, new_val, change = applied_effects['BESG Social Pillar Score']
                self.pillar_improvements['social'] += change
            elif pillar == 'governance' and 'BESG Governance Pillar Score' in applied_effects:
                _, new_val, change = applied_effects['BESG Governance Pillar Score']
                self.pillar_improvements['governance'] += change
        elif pillar == 'all':
            # Track improvements across all pillars
            self.pillar_counts['all'] += 1
            for p in ['environmental', 'social', 'governance']:
                pillar_metric = f'BESG {p.capitalize()} Pillar Score'
                if pillar_metric in applied_effects:
                    _, new_val, change = applied_effects[pillar_metric]
                    self.pillar_improvements[p] += change

        # Recalculate ESG score if we modified key metrics
        old_esg_score = state_dict.get('BESG ESG Score', 0)

        # Force recalculation of ESG score if we took an action other than 'No Action'
        if action != 0:
            state_dict['_esg_modified'] = True

        if state_dict['_esg_modified']:
            new_esg_score = self._calculate_esg_score(state_dict)

            # FIXED: Constrain ESG score relative to pillar scores
            pillar_scores = [
                state_dict.get('BESG Environmental Pillar Score', 0),
                state_dict.get('BESG Social Pillar Score', 0),
                state_dict.get('BESG Governance Pillar Score', 0)
            ]

            # ESG score shouldn't be more than 20% higher than the weighted average of pillars
            env_weight = 0.485
            soc_weight = 0.292
            gov_weight = 0.237

            weighted_avg = (pillar_scores[0] * env_weight +
                            pillar_scores[1] * soc_weight +
                            pillar_scores[2] * gov_weight)

            # Ensure ESG score doesn't exceed reasonable bounds relative to pillars
            if new_esg_score > weighted_avg * 1.2:
                new_esg_score = weighted_avg * 1.2

            # Also ensure maximum per-action change
            max_esg_change = 0.3
            if new_esg_score > old_esg_score + max_esg_change:
                new_esg_score = old_esg_score + max_esg_change

            state_dict['BESG ESG Score'] = new_esg_score
        else:
            new_esg_score = old_esg_score

        # Remove the temporary flag
        del state_dict['_esg_modified']

        # Convert dictionary back to numpy array
        new_state = np.array([state_dict.get(col, 0) for col in self.state_cols])

        return new_state, adjusted_cost, old_esg_score, new_esg_score, state_dict, financial_impact


    def _calculate_benchmark_reward(self, new_esg_score, old_esg_score, state_dict=None):
        """Calculate reward based on comparison to industry benchmarks"""
        benchmark_reward = 0

        # Check overall ESG score against benchmark
        old_benchmark = self._compare_to_benchmark('BESG ESG Score', old_esg_score)
        new_benchmark = self._compare_to_benchmark('BESG ESG Score', new_esg_score)

        # Reward for moving to a higher benchmark category
        if old_benchmark and new_benchmark:
            if old_benchmark == 'bottom_25' and new_benchmark != 'bottom_25':
                benchmark_reward += 2.0  # Big reward for moving out of bottom quartile
            elif old_benchmark == 'below_average' and new_benchmark in ['above_average', 'top_25']:
                benchmark_reward += 1.5  # Good reward for moving above average
            elif old_benchmark == 'above_average' and new_benchmark == 'top_25':
                benchmark_reward += 1.0  # Reward for moving to top quartile

        # Additional checks for pillar-specific benchmarks
        for pillar, metric in [('environmental', 'BESG Environmental Pillar Score'),
                              ('social', 'BESG Social Pillar Score'),
                              ('governance', 'BESG Governance Pillar Score')]:
            if metric in self.state_cols:
                pillar_idx = self.state_cols.index(metric)
                old_pillar_score = self.current_state[pillar_idx]

                # Get the new pillar score either from the state dictionary if provided,
                # or estimate it using relative improvement
                if state_dict and metric in state_dict:
                    new_pillar_score = state_dict[metric]
                elif old_esg_score > 0:  # Avoid division by zero
                    relative_improvement = new_esg_score / old_esg_score
                    new_pillar_score = old_pillar_score * relative_improvement
                else:
                    new_pillar_score = old_pillar_score  # No change if old score was 0

                old_pillar_benchmark = self._compare_to_benchmark(metric, old_pillar_score)
                new_pillar_benchmark = self._compare_to_benchmark(metric, new_pillar_score)

                if old_pillar_benchmark and new_pillar_benchmark and old_pillar_benchmark != new_pillar_benchmark:
                    if old_pillar_benchmark == 'bottom_25' and new_pillar_benchmark != 'bottom_25':
                        benchmark_reward += 1.0  # Reward for moving out of bottom quartile
                    elif old_pillar_benchmark == 'below_average' and new_pillar_benchmark in ['above_average', 'top_25']:
                        benchmark_reward += 0.75  # Reward for moving above average
                    elif old_pillar_benchmark == 'above_average' and new_pillar_benchmark == 'top_25':
                        benchmark_reward += 0.5  # Reward for moving to top quartile

        return benchmark_reward

    def step(self, action):
        """
        Execute one step in the environment with data-driven rewards
        """
        # Check if action is on cooldown and invalid
        if self._is_action_on_cooldown(action) and action != 0:
            # If trying to use an action on cooldown, apply a large penalty
            # and return the current state unchanged
            reward = -self.max_reward * 0.5  # FIXED: Reduced penalty to avoid too extreme negative rewards
            info = {
                'company': self.company,
                'year': self.current_year + self.current_step,
                'action': "INVALID - Action on cooldown",
                'pillar': None,
                'old_esg_score': self.current_state[self.state_cols.index('BESG ESG Score')] if 'BESG ESG Score' in self.state_cols else 0,
                'new_esg_score': self.current_state[self.state_cols.index('BESG ESG Score')] if 'BESG ESG Score' in self.state_cols else 0,
                'esg_improvement': 0,
                'initial_to_current_improvement': 0,
                'action_cost': 0,
                'financial_impact': 0,
                'base_improvement_reward': 0,
                'diversity_bonus': 0,
                'imbalance_penalty': 0,
                'financial_penalty': 0,
                'benchmark_reward': 0,
                'net_reward': reward,
                'pillar_improvements': self.pillar_improvements.copy(),
                'action_counts': self.action_counts.copy(),
                'pillar_counts': self.pillar_counts.copy(),
                'weakest_pillar': self._identify_weakest_pillar({self.state_cols[i]: self.current_state[i] for i in range(len(self.state_cols))}),
                'pillar_imbalance': self._calculate_pillar_imbalance({self.state_cols[i]: self.current_state[i] for i in range(len(self.state_cols))})
            }

            self.current_step += 1
            done = self.current_step >= self.max_steps

            # Decrease cooldown counters
            for a in self.action_cooldown:
                self.action_cooldown[a] = max(0, self.action_cooldown[a] - 1)

            return self.current_state, reward, done, info

        # Update action count and tracking
        self.action_counts[action] += 1
        self.last_actions.append(action)

        # Apply action effects with data-driven calibration
        new_state, cost, old_esg_score, new_esg_score, state_dict, financial_impact = self._apply_action_effects(action)

        # Calculate improvement from initial state for better context
        initial_to_new_improvement = new_esg_score - self.initial_values.get('BESG ESG Score', 0)

        # Calculate reward with data-driven components
        esg_improvement = new_esg_score - old_esg_score

        # FIXED: Ensure reward is always finite and not NaN
        if np.isnan(esg_improvement) or np.isinf(esg_improvement):
            esg_improvement = 0.0

        # 1. Basic reward from ESG improvement - data-driven improvement factor
        # Use dynamic factor based on current score rather than fixed thresholds
        if 'BESG ESG Score' in self.state_cols:
            current_esg = old_esg_score  # FIXED: Use old_esg_score instead of self.current_state for consistency
            improvement_factor = self.analyzer.get_improvement_factor(current_esg)
            if self.verbose:
                print(f"Current ESG: {current_esg:.2f}, Improvement: {esg_improvement:.4f}, Factor: {improvement_factor:.2f}")
            base_improvement_reward = esg_improvement * improvement_factor
        else:
            base_improvement_reward = esg_improvement

        # FIXED: Cap the base improvement reward to avoid extreme values
        base_improvement_reward = max(-5.0, min(5.0, base_improvement_reward))

        # 2. Financial sustainability component - data-driven penalties
        financial_penalty = 0
        if 'Net Income, Adj' in self.state_cols:
            net_income_idx = self.state_cols.index('Net Income, Adj')
            net_income = new_state[net_income_idx]
            initial_net_income = self.initial_values.get('Net Income, Adj', 0)

            # Calculate percentage decrease in net income - with data-driven thresholds
            if initial_net_income > 0:
                pct_decrease = max(0, (initial_net_income - net_income) / initial_net_income)

                # FIXED: More moderate financial penalties
                if pct_decrease > 0.5:  # Severe
                    financial_penalty = 2.0
                elif pct_decrease > 0.3:  # High
                    financial_penalty = 1.5
                elif pct_decrease > 0.15:  # Moderate
                    financial_penalty = 1.0
                elif pct_decrease > 0.05:  # Low
                    financial_penalty = 0.5

                # Bonus for financial improvement
                if pct_decrease < 0:
                    financial_penalty = -0.5  # Reduced bonus for financial improvement

        # FIXED: Ensure financial penalty is finite
        if np.isnan(financial_penalty) or np.isinf(financial_penalty):
            financial_penalty = 0.0

        # 3. Pillar balance component - data-driven importance
        pillar_imbalance = self._calculate_pillar_imbalance(state_dict)

        # FIXED: Ensure pillar_imbalance is finite
        if np.isnan(pillar_imbalance) or np.isinf(pillar_imbalance):
            pillar_imbalance = 0.0

        # Use analyzer's recommendation for balance weight
        balance_recommendation = self.analyzer.get_pillar_balance_recommendation()
        if balance_recommendation['recommendation'] == 'balanced':
            # If data shows balanced approach is better
            imbalance_penalty = pillar_imbalance * self.balance_weight

            # Enhanced penalties for high imbalance
            if pillar_imbalance > 0.2:
                imbalance_penalty = pillar_imbalance ** 2 * self.balance_weight * 2
        else:
            # If data shows focused approach is better, minimal balance penalty
            imbalance_penalty = pillar_imbalance * 0.1

        # FIXED: Cap the imbalance penalty
        imbalance_penalty = min(3.0, imbalance_penalty)

        # 4. Action diversity component with data-driven recommendations
        diversity_bonus = 0
        weakest_pillar = self._identify_weakest_pillar(state_dict)

        if action != 0:  # Skip for No Action
            pillar = self.actions[action].get('pillar', '')

            # Prioritize focus on weakest pillars if data shows that's effective
            if balance_recommendation['recommendation'] == 'focused':
                # Higher bonus for focusing on weakest pillar
                if pillar == weakest_pillar:
                    diversity_bonus += 2.0
                elif pillar == 'all':
                    diversity_bonus += 0.5  # Low bonus for "all" when focus is recommended
            else:
                # Standard bonus for balanced approach
                if pillar == weakest_pillar:
                    diversity_bonus += 1.0
                elif pillar == 'all':
                    diversity_bonus += 1.0  # Equal bonus for "all" when balance is recommended

            # FIXED: Reduced check for action patterns and repetition
            is_repeated = False
            if len(self.last_actions) >= 2:
                is_repeated = self.last_actions[-1] == self.last_actions[-2]

            # Apply penalty for repetitive actions
            if is_repeated:
                diversity_bonus -= 1.5  # Reduced from 3.0

            # Check for pillar balance in action selection
            pillar_counts = sum(self.pillar_counts.values())
            if pillar_counts > 0:
                most_used_pillar = max(self.pillar_counts.items(), key=lambda x: x[1] if x[0] != 'all' else -1)[0]
                least_used_pillar = min(self.pillar_counts.items(), key=lambda x: x[1] if x[0] != 'all' else float('inf'))[0]

                # Reward using underutilized pillars
                if pillar == least_used_pillar and least_used_pillar != 'all':
                    diversity_bonus += 1.0  # Reduced from 2.0

                # Penalize overuse of a single pillar if balanced approach is recommended
                if balance_recommendation['recommendation'] == 'balanced':
                    if pillar == most_used_pillar and self.pillar_counts[most_used_pillar] > pillar_counts / 2:
                        diversity_bonus -= 1.0  # Reduced from 2.0

        # FIXED: Ensure diversity_bonus is finite
        if np.isnan(diversity_bonus) or np.isinf(diversity_bonus):
            diversity_bonus = 0.0

        # 5. Cost efficiency component - data-driven
        cost_efficiency = 0
        if cost > 0 and esg_improvement > 0:
            # Calculate ROI: improvement per unit cost
            cost_efficiency = min(1.0, esg_improvement / cost * 10)  # Reduced max value

            # Add bonus for actions with positive financial impact
            if financial_impact > 0:
                cost_efficiency += 0.5  # Reduced from 1.0

        # FIXED: Ensure cost_efficiency is finite
        if np.isnan(cost_efficiency) or np.isinf(cost_efficiency):
            cost_efficiency = 0.0

        # 6. Benchmark comparison component
        benchmark_reward = self._calculate_benchmark_reward(new_esg_score, old_esg_score, state_dict)

        # FIXED: Ensure benchmark_reward is finite
        if np.isnan(benchmark_reward) or np.isinf(benchmark_reward):
            benchmark_reward = 0.0

        # Combine all reward components with data-driven weights
        # FIXED: Adjusted weights to be more balanced
        reward = (
            base_improvement_reward * 2.0 +      # Base ESG improvement (reduced from 3.0)
            diversity_bonus * self.diversity_weight * 0.5 -  # Strategy diversity (added 0.5 factor)
            imbalance_penalty * 0.8 -           # Pillar balance penalty (added 0.8 factor)
            financial_penalty * 0.7 +           # Financial sustainability (added 0.7 factor)
            cost_efficiency * 0.6 +             # Cost efficiency (added 0.6 factor)
            benchmark_reward * 0.9              # Industry benchmark comparison (added 0.9 factor)
        )

        # FIXED: Apply special cases with more moderate impacts
        # Penalty for "No Action" when ESG score is low
        if action == 0 and new_esg_score < 3:  # Adjusted for 0-10 scale
            reward -= 1.5  # Reduced from 3.0

        # Bonus for achieving balanced pillars
        if pillar_imbalance < 0.15 and new_esg_score > 2:
            if balance_recommendation['recommendation'] == 'balanced':
                reward += 1.0  # Reduced from 2.0
            else:
                reward += 0.3  # Reduced from 0.5

        # FIXED: Hard cap on reward values to prevent extreme values
        reward = np.clip(reward, -self.max_reward * 0.8, self.max_reward * 0.8)

        # FIXED: Ensure reward is a valid, finite number
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        # Update cooldown for actions - STRICT COOLDOWN IMPLEMENTATION
        if action != 0:  # No cooldown for "No Action"
            # Set the cooldown for the used action
            self.action_cooldown[action] = self.cooldown_duration

        # Decrease cooldown for all other actions
        for a in self.action_cooldown:
            if a != action:  # Don't decrease the one we just set
                self.action_cooldown[a] = max(0, self.action_cooldown[a] - 1)

        # Update state and step counter
        self.current_state = new_state
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= self.max_steps

        # Additional info for analysis and visualization
        info = {
            'company': self.company,
            'year': self.current_year + self.current_step,
            'action': self.actions[action]['name'],
            'pillar': self.actions[action].get('pillar', ''),
            'old_esg_score': old_esg_score,
            'new_esg_score': new_esg_score,
            'esg_improvement': esg_improvement,
            'initial_to_current_improvement': initial_to_new_improvement,
            'action_cost': cost,
            'financial_impact': financial_impact,
            'base_improvement_reward': base_improvement_reward * 2.0,  # Match the weight used above
            'diversity_bonus': diversity_bonus * self.diversity_weight * 0.5,  # Match the weight used above
            'imbalance_penalty': imbalance_penalty * 0.8,  # Match the weight used above
            'financial_penalty': financial_penalty * 0.7,  # Match the weight used above
            'cost_efficiency': cost_efficiency * 0.6,  # Match the weight used above
            'benchmark_reward': benchmark_reward * 0.9,  # Match the weight used above
            'net_reward': reward,
            'pillar_improvements': self.pillar_improvements.copy(),
            'action_counts': self.action_counts.copy(),
            'pillar_counts': self.pillar_counts.copy(),
            'weakest_pillar': weakest_pillar,
            'pillar_imbalance': pillar_imbalance,
            'action_cooldowns': self.action_cooldown.copy(),
            'balance_recommendation': balance_recommendation['recommendation']
        }

        return self.current_state, reward, done, info


    def render(self, mode='human'):
        """Render the environment with data-driven insights"""
        if mode == 'human':
            # Print current state information
            print(f"Step: {self.current_step}, Company: {self.company}, Year: {self.current_year + self.current_step}")

            # Display key metrics
            for i, col in enumerate(self.state_cols):
                if col in ['BESG ESG Score', 'BESG Environmental Pillar Score', 'BESG Social Pillar Score',
                         'BESG Governance Pillar Score', 'Revenue, Adj', 'Net Income, Adj',
                         'Market Cap ($M)', 'Renewable Energy Use']:
                    print(f"{col}: {self.current_state[i]:.2f}")

            # Display pillar balance
            env_idx = self.state_cols.index('BESG Environmental Pillar Score') if 'BESG Environmental Pillar Score' in self.state_cols else None
            soc_idx = self.state_cols.index('BESG Social Pillar Score') if 'BESG Social Pillar Score' in self.state_cols else None
            gov_idx = self.state_cols.index('BESG Governance Pillar Score') if 'BESG Governance Pillar Score' in self.state_cols else None

            if None not in [env_idx, soc_idx, gov_idx]:
                env_score = self.current_state[env_idx]
                soc_score = self.current_state[soc_idx]
                gov_score = self.current_state[gov_idx]

                imbalance = self._calculate_pillar_imbalance({
                    'BESG Environmental Pillar Score': env_score,
                    'BESG Social Pillar Score': soc_score,
                    'BESG Governance Pillar Score': gov_score
                })

                print(f"Pillar Balance - Env: {env_score:.2f}, Social: {soc_score:.2f}, Gov: {gov_score:.2f}")
                print(f"Pillar Imbalance Score: {imbalance:.2f}")

                # Show benchmark comparisons
                esg_idx = self.state_cols.index('BESG ESG Score') if 'BESG ESG Score' in self.state_cols else None
                if esg_idx is not None:
                    esg_score = self.current_state[esg_idx]
                    benchmark = self._compare_to_benchmark('BESG ESG Score', esg_score)
                    if benchmark:
                        print(f"ESG Score Benchmark: {benchmark.replace('_', ' ').upper()}")

                # Show actions on cooldown
                cooldown_actions = [i for i, count in self.action_cooldown.items() if count > 0]
                if cooldown_actions:
                    print("Actions on cooldown:", ", ".join([f"{i} ({self.actions[i]['name']}): {self.action_cooldown[i]} steps" for i in cooldown_actions]))

                # NEW: Show data-driven strategy recommendation
                balance_rec = self.analyzer.get_pillar_balance_recommendation()['recommendation']
                print(f"Data recommends {balance_rec.upper()} improvement strategy")

    def get_available_actions(self):
        """Get list of available actions with descriptions"""
        return [(i, action['name'], action.get('cost', 0), action.get('pillar', ''))
                for i, action in self.actions.items()]

    def get_valid_actions(self):
        """Get list of valid actions (not on cooldown)"""
        return [i for i, cooldown in self.action_cooldown.items() if cooldown == 0]

    def get_company_insights(self):
        """Get data-driven insights about the company"""
        return self.company_insights

    def get_similar_companies(self):
        """Get similar companies to the current one"""
        return self.similar_companies

    def get_effective_actions(self):
        """Get historically effective actions based on data analysis"""
        effective = {}
        for pillar in ['environmental', 'social', 'governance']:
            effective[pillar] = self.analyzer.get_effective_actions_for_pillar(pillar)
        return effective

    def get_improvement_factors(self):
        """Get data-driven improvement factors at different ESG score levels"""
        return self.analyzer.improvement_factors

    def get_financial_impacts(self):
        """Get data-driven financial impacts of different initiative types"""
        return self.analyzer.financial_impacts

    def get_pillar_balance_recommendation(self):
        """Get data-driven recommendation for pillar balance approach"""
        return self.analyzer.pillar_balance_analysis


# Enhanced Prioritized Experience Replay
class EnhancedPrioritizedReplayBuffer:
    """Enhanced Prioritized Experience Replay with fixed-size numpy arrays for stability"""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """Initialize an EnhancedPrioritizedReplayBuffer object."""
        self.action_size = action_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

        # Use numpy arrays for better memory management and stability
        self.priorities = np.ones(buffer_size, dtype=np.float32)
        self.memory = []

        self.pos = 0  # Position to add new experiences
        self.size = 0  # Current size of memory

        # Prioritization parameters
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta = beta_start  # Importance sampling correction
        self.beta_frames = beta_frames  # Frames over which to anneal beta
        self.frame = 1
        self.max_priority = 1.0  # Initial max priority

    def add(self, state, action, reward, next_state, done, error=None):
        """Add a new experience to memory with priority."""
        e = self.experience(state, action, reward, next_state, done)

        # Add experience to memory
        if len(self.memory) < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.pos] = e

        # Set priority
        if error is not None:
            # Handle potential NaN error
            if np.isnan(error):
                priority = self.max_priority
            else:
                priority = (float(abs(error)) + 1e-5) ** self.alpha
        else:
            priority = self.max_priority  # Set to max observed priority if no error provided

        # Ensure valid priority
        if np.isnan(priority) or priority <= 0:
            priority = self.max_priority

        self.priorities[self.pos] = priority

        # Update position and size
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        """Sample a batch of experiences using prioritization."""
        if self.size == 0:
            return None

        # Update beta for importance sampling
        self.beta = min(1.0, self.beta + self.frame * (1.0 - self.beta) / self.beta_frames)
        self.frame += 1

        # Get valid priorities (only for filled memory slots)
        valid_priorities = self.priorities[:self.size]

        # Handle potential NaN values or zeros in priorities
        valid_priorities = np.nan_to_num(valid_priorities, nan=self.max_priority)
        valid_priorities = np.maximum(valid_priorities, 1e-8)  # Ensure no zeros

        # Calculate sampling probabilities
        priority_sum = valid_priorities.sum()
        if priority_sum > 0:
            probs = valid_priorities / priority_sum
        else:
            # If all priorities are 0, use uniform distribution
            probs = np.ones_like(valid_priorities) / self.size

        # Ensure probabilities sum to 1 (prevent numerical errors)
        probs = probs / np.sum(probs)

        # Sample indices based on priorities
        indices = np.random.choice(self.size, size=min(self.size, self.batch_size), p=probs, replace=False)

        # Calculate importance-sampling weights (with safety checks)
        weights = np.power(self.size * probs[indices], -self.beta)
        # Handle potential infinities or NaNs
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)

        if np.max(weights) > 0:
            weights = weights / np.max(weights)  # Normalize weights
        else:
            weights = np.ones_like(weights)

        # Get selected experiences
        experiences = [self.memory[i] for i in indices]

        # Convert to tensor format for neural network
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(weights.astype(np.float32)).unsqueeze(1).to(device)

        return (states, actions, rewards, next_states, dones, weights, indices)

    def update_priorities(self, indices, errors):
        """Update priorities for sampled transitions."""
        for i, error in zip(indices, errors):
            if i < self.size:  # Ensure index is valid
                # Handle NaN errors
                if np.isnan(error):
                    priority = self.max_priority
                else:
                    priority = (float(abs(error)) + 1e-5) ** self.alpha

                # Ensure priority is valid
                if np.isnan(priority) or priority <= 0:
                    priority = self.max_priority

                self.priorities[i] = priority
                self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size


# Enhanced Dueling DQN Architecture with Noisy Networks for better exploration
class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration"""

    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class DataDrivenDuelingDQN(nn.Module):
    """Enhanced Dueling DQN with data-driven architecture for better ESG modeling"""

    def __init__(self, state_size, action_size, seed=42, fc1_units=256, fc2_units=256):
        super(DataDrivenDuelingDQN, self).__init__()
        self.seed = torch.manual_seed(seed)

        # FIXED: Use smaller initial layer sizes for better stability
        # Feature extraction layers - shared for all metrics
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)  # Batch normalization for stability
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

        # FIXED: Initialize weights with smaller values for stability
        self._initialize_weights()

        # FIXED: Use standard linear layers instead of NoisyLinear for initial stabilization
        # Later we can reintroduce NoisyLinear when the network is stable
        # Advantage stream with domain-specific layers
        self.advantage_hidden = nn.Linear(fc2_units, 128)
        self.advantage = nn.Linear(128, action_size)

        # Value stream with domain-specific layers
        self.value_hidden = nn.Linear(fc2_units, 128)
        self.value = nn.Linear(128, 1)

        # Dropout for regularization - helps prevent overfitting
        # FIXED: Increase dropout to prevent overfitting
        self.dropout = nn.Dropout(0.3)

    def _initialize_weights(self):
        """Initialize weights with smaller values for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # FIXED: Use a more conservative initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # FIXED: Use small constant bias initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, state):
        """Forward pass with dueling architecture for better value estimation"""
        # FIXED: Add gradient clipping at runtime to prevent exploding gradients
        with torch.no_grad():
            if torch.isnan(state).any() or torch.isinf(state).any():
                # Replace problematic values
                state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        # Feature extraction
        x = F.relu(self.fc1(state))
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.bn1(x)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.dropout(x)

        # FIXED: Add gradient checks between layers
        with torch.no_grad():
            if torch.isnan(x).any() or torch.isinf(x).any():
                # If we still have NaNs after main layers, replace them
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # Value stream
        val_hidden = F.relu(self.value_hidden(x))
        val = self.value(val_hidden)

        # Advantage stream
        adv_hidden = F.relu(self.advantage_hidden(x))
        adv = self.advantage(adv_hidden)

        # FIXED: Add more checks before final calculation
        with torch.no_grad():
            if torch.isnan(val).any() or torch.isinf(val).any():
                val = torch.nan_to_num(val, nan=0.0, posinf=1.0, neginf=-1.0)
            if torch.isnan(adv).any() or torch.isinf(adv).any():
                adv = torch.nan_to_num(adv, nan=0.0, posinf=1.0, neginf=-1.0)

        # Combine value and advantage using dueling technique
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum(A(s,a')))
        avg_adv = adv.mean(dim=1, keepdim=True)
        q_values = val + adv - avg_adv

        # FIXED: Final safety check
        with torch.no_grad():
            if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                q_values = torch.nan_to_num(q_values, nan=0.0, posinf=1.0, neginf=-1.0)

        return q_values

    def reset_noise(self):
        """Dummy method to maintain compatibility with previous code."""
        pass


class DataDrivenDQNAgent:
    """
    Data-Driven DQN Agent with ESG-specific enhancements
    """
    def __init__(self, state_size, action_size, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Use enhanced network architecture
        self.qnetwork_local = DataDrivenDuelingDQN(state_size, action_size, seed).to(device)
        self.qnetwork_target = DataDrivenDuelingDQN(state_size, action_size, seed).to(device)

        # FIXED: Use a much lower learning rate to prevent NaN issues
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-5)  # Changed from 2e-4

        # FIXED: Slower learning rate decay
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.95)  # Changed from step_size=100, gamma=0.9

        # Enhanced Prioritized Replay memory
        self.memory = EnhancedPrioritizedReplayBuffer(
            action_size, buffer_size=100000, batch_size=64, seed=seed,
            alpha=0.6, beta_start=0.4, beta_frames=100000
        )

        # Learning parameters
        self.gamma = 0.99  # Discount factor
        self.tau = 1e-3    # Soft update parameter
        self.update_every = 4  # How often to update the network
        self.t_step = 0

        # Double DQN
        self.use_double_dqn = True

        # Multi-step learning parameters
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)

        # Exploration parameters - for epsilon-greedy (used alongside noisy networks)
        self.eps = 1.0
        # FIXED: More aggressive epsilon decay for better exploration
        self.eps_end = 0.02   # Changed from 0.01
        self.eps_decay = 0.99  # Changed from 0.995

    def step(self, state, action, reward, next_state, done):
        try:
            # Store experience in n-step buffer for multi-step learning
            self.n_step_buffer.append((state, action, reward, next_state, done))

            # Only add to replay buffer if we have enough steps
            if len(self.n_step_buffer) == self.n_step:
                try:
                    # Get the n-step discounted reward
                    n_step_reward = 0
                    for i in range(self.n_step):
                        n_step_reward += (self.gamma ** i) * self.n_step_buffer[i][2]

                    # Get the initial state and action
                    initial_state = self.n_step_buffer[0][0]
                    initial_action = self.n_step_buffer[0][1]

                    # Get the final next state and done flag
                    final_next_state = self.n_step_buffer[-1][3]
                    final_done = self.n_step_buffer[-1][4]

                    # Guard against invalid states or actions
                    if (initial_state is None or
                        final_next_state is None or
                        not isinstance(initial_action, (int, np.integer)) or
                        initial_action < 0 or
                        initial_action >= self.action_size):
                        # Skip this experience
                        self.n_step_buffer.pop(0)
                        return

                    # Compute estimated TD error for prioritization
                    try:
                        with torch.no_grad():
                            state_tensor = torch.from_numpy(initial_state).float().unsqueeze(0).to(device)
                            next_state_tensor = torch.from_numpy(final_next_state).float().unsqueeze(0).to(device)

                            # Get current Q value
                            q_values = self.qnetwork_local(state_tensor).data.cpu().numpy()[0]

                            # Check for NaN values
                            if np.isnan(q_values).any():
                                # Use default error if we have NaN values
                                td_error = 1.0
                            else:
                                q_current = q_values[initial_action]

                                # Get next Q value using target network (Double DQN if enabled)
                                if self.use_double_dqn:
                                    # Double DQN: use local network to select action, target network to evaluate
                                    next_q_values = self.qnetwork_local(next_state_tensor).data.cpu().numpy()[0]
                                    if np.isnan(next_q_values).any():
                                        next_action = 0  # Use default action if NaN
                                    else:
                                        next_action = np.argmax(next_q_values)

                                    target_q_values = self.qnetwork_target(next_state_tensor).data.cpu().numpy()[0]
                                    if np.isnan(target_q_values).any():
                                        q_next = 0  # Use default value if NaN
                                    else:
                                        q_next = target_q_values[next_action]
                                else:
                                    # Standard DQN
                                    target_q_values = self.qnetwork_target(next_state_tensor).data.cpu().numpy()[0]
                                    if np.isnan(target_q_values).any():
                                        q_next = 0  # Use default value if NaN
                                    else:
                                        q_next = np.max(target_q_values)

                                # Calculate target Q value with n-step return
                                target = n_step_reward + (self.gamma ** self.n_step) * q_next * (1 - final_done)

                                # TD error for prioritization
                                td_error = abs(target - q_current)

                                # Handle NaN TD error
                                if np.isnan(td_error):
                                    td_error = 1.0  # Default priority
                    except Exception as e:
                        print(f"Error calculating TD error: {e}")
                        td_error = 1.0  # Default priority

                    # Add to replay buffer with TD error for prioritization
                    self.memory.add(initial_state, initial_action, n_step_reward, final_next_state, final_done, td_error)

                except Exception as e:
                    print(f"Error processing n-step experience: {e}")
                    # Remove oldest experience and continue
                    if len(self.n_step_buffer) > 0:
                        self.n_step_buffer.pop(0)

            # Learn every UPDATE_EVERY time steps
            self.t_step = (self.t_step + 1) % self.update_every
            if self.t_step == 0:
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > self.memory.batch_size:
                    try:
                        experiences = self.memory.sample()
                        if experiences is not None:
                            self.learn(experiences, self.gamma)
                    except Exception as e:
                        print(f"Error during learning: {e}")

        except Exception as e:
            print(f"Error in step function: {e}")

    def act(self, state, eps=None, env=None):
        """Returns actions for given state using current policy with exploration.

        Args:
            state: Current state
            eps: Epsilon for exploration
            env: Environment (to check for valid actions)
        """
        if eps is None:
            eps = self.eps

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Reset noise for next forward pass
        self.qnetwork_local.reset_noise()

        # Get valid actions if environment is provided
        valid_actions = list(range(self.action_size))
        if env is not None:
            valid_actions = env.get_valid_actions()

            # ADDED: Give preference to actions targeting weakest pillar
            if random.random() < 0.4 and 'BESG ESG Score' in env.state_cols:
                # Identify weakest pillar
                state_dict = {env.state_cols[i]: state[i] for i in range(len(state))}
                weakest_pillar = env._identify_weakest_pillar(state_dict)

                # Filter for actions targeting weakest pillar
                weakest_pillar_actions = [a for a in valid_actions if a != 0 and
                                         env.actions[a].get('pillar', '') == weakest_pillar]

                if weakest_pillar_actions:
                    # 40% chance to focus on weakest pillar
                    valid_actions = weakest_pillar_actions

        # Epsilon-greedy action selection (used alongside noisy networks)
        if random.random() > eps:
            # Choose best valid action
            if len(valid_actions) < self.action_size:  # We have restricted actions
                # Extract values only for valid actions and find the best one
                valid_values = action_values[0, valid_actions].cpu().data.numpy()
                best_valid_idx = np.argmax(valid_values)
                return valid_actions[best_valid_idx]
            else:
                # All actions are valid, just take the best one
                return np.argmax(action_values.cpu().data.numpy())
        else:
            # Random selection from valid actions
            return random.choice(valid_actions)

    def learn(self, experiences, gamma):
        """Update value parameters using batch of experience tuples."""
        states, actions, rewards, next_states, dones, weights, indices = experiences

        try:
            # Get expected Q values from local model
            q_expected = self.qnetwork_local(states).gather(1, actions)

            if self.use_double_dqn:
                # Double DQN: Use local network to select best actions for next states
                next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
                # Use target network to evaluate those actions
                q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_actions)
            else:
                # Standard DQN
                q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

            # Compute Q targets for current states with n-step return
            q_targets = rewards + (gamma ** self.n_step) * q_targets_next * (1 - dones)

            # FIXED: More robust NaN checking with replacements instead of skipping
            if torch.isnan(q_expected).any():
                # Replace NaN values with zeros instead of skipping
                q_expected = torch.nan_to_num(q_expected, nan=0.0)
                print("Warning: NaN values in q_expected replaced with zeros")

            if torch.isnan(q_targets).any():
                # Replace NaN values with zeros
                q_targets = torch.nan_to_num(q_targets, nan=0.0)
                print("Warning: NaN values in q_targets replaced with zeros")

            # Compute loss (weighted MSE with importance sampling)
            element_wise_loss = F.mse_loss(q_expected, q_targets, reduction='none')

            # FIXED: More robust NaN checking for weights
            if torch.isnan(element_wise_loss).any() or torch.isnan(weights).any():
                # Replace NaN values in loss and weights
                element_wise_loss = torch.nan_to_num(element_wise_loss, nan=0.0)
                weights = torch.nan_to_num(weights, nan=1.0)
                print("Warning: NaN values in loss or weights replaced")

            loss = (weights * element_wise_loss).mean()

            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()

            # FIXED: More aggressive gradient clipping to prevent NaN issues
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 0.5)  # Changed from 1.0 to 0.5

            self.optimizer.step()

            # Update target network
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

            # FIXED: More robust TD error calculation
            with torch.no_grad():
                td_errors = abs(q_expected - q_targets).detach().cpu().numpy()
                # Replace any remaining NaN values
                td_errors = np.nan_to_num(td_errors, nan=1.0)

            self.memory.update_priorities(indices, td_errors)

            # Update learning rate
            self.scheduler.step()

            # Update epsilon for exploration
            self.eps = max(self.eps_end, self.eps_decay * self.eps)

        except Exception as e:
            print(f"Error in learn function: {e}")
            # Continue with epsilon decay even if learning fails
            self.eps = max(self.eps_end, self.eps_decay * self.eps)

    def soft_update(self, local_model, target_model, tau):
        """Soft update target network parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, filepath):
        """Save the agent's learning model."""
        torch.save({
            'model_state_dict': self.qnetwork_local.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.eps
        }, filepath)

    def load(self, filepath):
        """Load a saved model."""
        if device.type == 'cuda':
            checkpoint = torch.load(filepath)
        else:
            # Load to CPU if CUDA is not available
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

        self.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.eps = checkpoint['epsilon']


# Advanced training with data-driven curriculum learning
def data_driven_training(env, n_episodes=500, early_stop_threshold=None, checkpoint_dir='./checkpoints',
                         evaluate_every=50, curriculum_learning=True):
    """Data-driven training with curriculum learning and adaptive exploration."""
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Initialize agent with state size from environment
    state_size = len(env.reset())
    action_size = env.action_space.n
    agent = DataDrivenDQNAgent(state_size=state_size, action_size=action_size)

    # Track scores and performance metrics
    scores = []
    recent_scores = deque(maxlen=100)
    best_avg_score = -np.inf

    # Track pillar improvements for analysis
    env_improvements = []
    social_improvements = []
    gov_improvements = []

    # Curriculum learning stages - dynamically adjusted based on company data
    if curriculum_learning:
        # Get data-driven recommendation for balance vs. focus approach
        balance_rec = env.get_pillar_balance_recommendation()
        balance_weight = balance_rec['optimal_weight']

        # Design curriculum based on company's current ESG score
        # to better target appropriate improvement level
        initial_esg = env.initial_values.get('BESG ESG Score', 0) if hasattr(env, 'initial_values') else 0

        # FIXED: More moderate curriculum progression
        curriculum_stages = [
            {'max_steps': 3, 'episodes': 100, 'diversity_weight': 2.0, 'balance_weight': balance_weight, 'cooldown': 1},  # Intro
            {'max_steps': 5, 'episodes': 150, 'diversity_weight': 2.5, 'balance_weight': balance_weight, 'cooldown': 2},  # Intermediate
            {'max_steps': 8, 'episodes': 250, 'diversity_weight': 3.0, 'balance_weight': balance_weight, 'cooldown': 3}   # Advanced
        ]

        current_stage = 0
        episode_counter = 0
        total_episodes = 0

        # Set initial curriculum parameters
        env.max_steps = curriculum_stages[current_stage]['max_steps']
        env.diversity_weight = curriculum_stages[current_stage]['diversity_weight']
        env.balance_weight = curriculum_stages[current_stage]['balance_weight']
        env.cooldown_duration = curriculum_stages[current_stage]['cooldown']

    # Create evaluation environment
    eval_env = copy.deepcopy(env)

    # FIXED: Enable verbose mode during evaluation
    eval_env.verbose = True

    # Main training loop
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        pillar_improvements = {'environmental': 0, 'social': 0, 'governance': 0}

        # Episode loop
        done = False
        while not done:
            # Use environment-aware action selection
            action = agent.act(state, env=env)
            next_state, reward, done, info = env.step(action)

            # FIXED: Add debug output for checking reward calculation
            if i_episode == 1 and env.current_step < 3:
                print(f"Step {env.current_step}: Action={env.actions[action]['name']}, Reward={reward:.2f}")
                if 'esg_improvement' in info:
                    print(f"  ESG Improvement: {info['esg_improvement']:.4f}")

            agent.step(state, action, reward, next_state, done)

            # Track improvements for analysis
            for pillar in ['environmental', 'social', 'governance']:
                if pillar in info['pillar_improvements']:
                    pillar_improvements[pillar] = info['pillar_improvements'][pillar]

            state = next_state
            score += reward

        # Store score and pillar improvements
        scores.append(score)
        recent_scores.append(score)
        env_improvements.append(pillar_improvements['environmental'])
        social_improvements.append(pillar_improvements['social'])
        gov_improvements.append(pillar_improvements['governance'])

        # Print progress
        if i_episode % 10 == 0:
            avg_score = np.mean(recent_scores) if recent_scores else 0
            print(f'Episode {i_episode}\tAverage Score: {avg_score:.2f}\tEpsilon: {agent.eps:.3f}')
            print(f'Pillar Improvements - Env: {np.mean(env_improvements[-10:]):.3f}, '
                  f'Social: {np.mean(social_improvements[-10:]):.3f}, '
                  f'Gov: {np.mean(gov_improvements[-10:]):.3f}')

        # Evaluate agent periodically
        if i_episode % evaluate_every == 0:
            print("\nStarting evaluation...")
            eval_score, eval_pillars = evaluate_agent(eval_env, agent, num_episodes=3)
            print(f'Evaluation at episode {i_episode}: Avg Score: {eval_score:.2f}')
            print(f'Eval Pillar Balance - Env: {eval_pillars["env"]:.3f}, '
                  f'Social: {eval_pillars["social"]:.3f}, Gov: {eval_pillars["gov"]:.3f}')

            # Save if this is the best model so far based on evaluation
            if eval_score > best_avg_score:
                best_avg_score = eval_score
                agent.save(f'{checkpoint_dir}/best_model.pth')
                print(f'New best model saved with eval score: {best_avg_score:.2f}')

        # Curriculum learning stage progression
        if curriculum_learning:
            episode_counter += 1
            total_episodes += 1

            if current_stage < len(curriculum_stages) - 1 and episode_counter >= curriculum_stages[current_stage]['episodes']:
                current_stage += 1
                episode_counter = 0

                # Update environment parameters
                env.max_steps = curriculum_stages[current_stage]['max_steps']
                env.diversity_weight = curriculum_stages[current_stage]['diversity_weight']
                env.balance_weight = curriculum_stages[current_stage]['balance_weight']
                env.cooldown_duration = curriculum_stages[current_stage]['cooldown']

                # Also update eval environment
                eval_env.max_steps = curriculum_stages[current_stage]['max_steps']
                eval_env.diversity_weight = curriculum_stages[current_stage]['diversity_weight']
                eval_env.balance_weight = curriculum_stages[current_stage]['balance_weight']
                eval_env.cooldown_duration = curriculum_stages[current_stage]['cooldown']

                print(f"\nAdvancing to curriculum stage {current_stage+1}: "
                     f"max_steps = {env.max_steps}, "
                     f"diversity_weight = {env.diversity_weight}, "
                     f"balance_weight = {env.balance_weight}, "
                     f"cooldown = {env.cooldown_duration}")

        # Early stopping
        if early_stop_threshold is not None and len(recent_scores) >= 100:
            if np.mean(recent_scores) >= early_stop_threshold:
                print(f'\nEnvironment solved in {i_episode} episodes! '
                      f'Average Score: {np.mean(recent_scores):.2f}')
                agent.save(f'{checkpoint_dir}/final_model.pth')
                break

    # Save final model regardless
    agent.save(f'{checkpoint_dir}/final_model.pth')

    return scores, agent, (env_improvements, social_improvements, gov_improvements)


def evaluate_agent(env, agent, num_episodes=3):
    """Evaluate the agent's performance without exploration."""
    total_score = 0
    env_scores = []
    social_scores = []
    gov_scores = []

    # FIXED: Create a fresh copy of the environment for evaluation
    eval_env = copy.deepcopy(env)

    # FIXED: Reset all episode-specific variables to ensure fresh evaluation
    for i in range(num_episodes):
        state = eval_env.reset()
        score = 0
        done = False
        step = 0

        # FIXED: Set a max step limit to prevent infinite loops
        max_steps = eval_env.max_steps

        while not done and step < max_steps:
            # Act without exploration (epsilon=0) and respect action cooldowns
            action = agent.act(state, eps=0.0, env=eval_env)
            next_state, reward, done, info = eval_env.step(action)

            # FIXED: Add detailed logging for debugging
            if i == 0 and step < 3:  # Log first few steps of first episode
                print(f"Eval Step {step}: Action={eval_env.actions[action]['name']}, Reward={reward:.2f}")
                if 'esg_improvement' in info:
                    print(f"  ESG Improvement: {info['esg_improvement']:.4f}")

            state = next_state
            score += reward
            step += 1

        # FIXED: Add more detailed output for final state
        if i == 0:
            print(f"Evaluation episode ended after {step} steps with score {score:.2f}")

        total_score += score

        # Track final pillar scores
        env_idx = eval_env.state_cols.index('BESG Environmental Pillar Score') if 'BESG Environmental Pillar Score' in eval_env.state_cols else None
        soc_idx = eval_env.state_cols.index('BESG Social Pillar Score') if 'BESG Social Pillar Score' in eval_env.state_cols else None
        gov_idx = eval_env.state_cols.index('BESG Governance Pillar Score') if 'BESG Governance Pillar Score' in eval_env.state_cols else None

        if None not in [env_idx, soc_idx, gov_idx]:
            env_scores.append(state[env_idx])
            social_scores.append(state[soc_idx])
            gov_scores.append(state[gov_idx])

    avg_score = total_score / num_episodes
    avg_pillars = {
        'env': np.mean(env_scores) if env_scores else 0,
        'social': np.mean(social_scores) if social_scores else 0,
        'gov': np.mean(gov_scores) if gov_scores else 0
    }

    return avg_score, avg_pillars



# Enhanced visualization with data-driven insights
def visualize_esg_strategy(env, agent, num_episodes=1, save_dir='.'):
    """Test the trained agent and visualize the ESG strategy with data-driven insights"""

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Store results for visualization
    results = []

    for i in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        episode_results = {
            'step': [], 'action': [], 'pillar': [], 'esg_score': [],
            'env_score': [], 'social_score': [], 'gov_score': [],
            'net_income': [], 'reward': [], 'pillar_imbalance': [],
            'improvement': [], 'cost': [], 'weakest_pillar': [],
            'benchmark_category': [], 'financial_impact': []
        }

        # Record initial state
        env_idx = env.state_cols.index('BESG Environmental Pillar Score') if 'BESG Environmental Pillar Score' in env.state_cols else None
        soc_idx = env.state_cols.index('BESG Social Pillar Score') if 'BESG Social Pillar Score' in env.state_cols else None
        gov_idx = env.state_cols.index('BESG Governance Pillar Score') if 'BESG Governance Pillar Score' in env.state_cols else None
        esg_idx = env.state_cols.index('BESG ESG Score') if 'BESG ESG Score' in env.state_cols else None
        net_income_idx = env.state_cols.index('Net Income, Adj') if 'Net Income, Adj' in env.state_cols else None

        # Record initial state
        episode_results['step'].append(0)
        episode_results['action'].append('Initial State')
        episode_results['pillar'].append(None)
        episode_results['esg_score'].append(state[esg_idx] if esg_idx is not None else 0)
        episode_results['env_score'].append(state[env_idx] if env_idx is not None else 0)
        episode_results['social_score'].append(state[soc_idx] if soc_idx is not None else 0)
        episode_results['gov_score'].append(state[gov_idx] if gov_idx is not None else 0)
        episode_results['net_income'].append(state[net_income_idx] if net_income_idx is not None else 0)
        episode_results['reward'].append(0)
        episode_results['improvement'].append(0)
        episode_results['cost'].append(0)
        episode_results['financial_impact'].append(0)

        # Calculate initial pillar imbalance
        state_dict = {env.state_cols[i]: state[i] for i in range(len(state))}
        pillar_imbalance = env._calculate_pillar_imbalance(state_dict)
        weakest_pillar = env._identify_weakest_pillar(state_dict)

        episode_results['pillar_imbalance'].append(pillar_imbalance)
        episode_results['weakest_pillar'].append(weakest_pillar)

        # Get benchmark category if available
        if hasattr(env, '_compare_to_benchmark'):
            benchmark = env._compare_to_benchmark('BESG ESG Score', state[esg_idx] if esg_idx is not None else 0)
            episode_results['benchmark_category'].append(benchmark)
        else:
            episode_results['benchmark_category'].append(None)

        print(f"\nTesting Episode {i+1}")
        print("Initial state:")
        env.render()

        # NEW: Show data-driven insights
        if hasattr(env, 'get_company_insights'):
            company_insights = env.get_company_insights()
            print("\nCompany ESG Insights:")
            for insight in company_insights[:5]:  # Show first 5 insights
                print(f"- {insight}")
            print("...")

        # NEW: Show ESG score improvement patterns
        if hasattr(env, 'get_improvement_factors'):
            improvement_factors = env.get_improvement_factors()
            print("\nData-Driven ESG Improvement Patterns:")
            if improvement_factors:
                best_range = max(improvement_factors.items(), key=lambda x: x[1])
                print(f"- Fastest improvements happen for ESG scores between {best_range[0][0]}-{best_range[0][1]} (factor: {best_range[1]:.2f}x)")
                print(f"- Current ESG score: {state[esg_idx] if esg_idx is not None else 0:.2f}")
                current_factor = env.analyzer.get_improvement_factor(state[esg_idx] if esg_idx is not None else 0)
                print(f"- Current improvement potential: {current_factor:.2f}x")

        # NEW: Show pillar balance approach recommendation
        if hasattr(env, 'get_pillar_balance_recommendation'):
            balance_rec = env.get_pillar_balance_recommendation()
            print(f"\nData recommends {balance_rec['recommendation'].upper()} improvement approach")
            if balance_rec['recommendation'] == 'focused':
                print(f"- Focus on improving {weakest_pillar.upper()} pillar first")
            else:
                print(f"- Balance improvements across all pillars")

        while not done:
            # Use environment-aware action selection
            action = agent.act(state, eps=0.0, env=env)
            next_state, reward, done, info = env.step(action)

            # Record results
            episode_results['step'].append(step + 1)
            episode_results['action'].append(env.actions[action]['name'])
            episode_results['pillar'].append(env.actions[action].get('pillar', None))
            episode_results['esg_score'].append(next_state[esg_idx] if esg_idx is not None else 0)
            episode_results['env_score'].append(next_state[env_idx] if env_idx is not None else 0)
            episode_results['social_score'].append(next_state[soc_idx] if soc_idx is not None else 0)
            episode_results['gov_score'].append(next_state[gov_idx] if gov_idx is not None else 0)
            episode_results['net_income'].append(next_state[net_income_idx] if net_income_idx is not None else 0)
            episode_results['reward'].append(reward)
            episode_results['improvement'].append(info['esg_improvement'])
            episode_results['cost'].append(info['action_cost'])
            episode_results['pillar_imbalance'].append(info['pillar_imbalance'])
            episode_results['weakest_pillar'].append(info['weakest_pillar'])
            episode_results['financial_impact'].append(info.get('financial_impact', 0))

            # Get benchmark category if available
            if hasattr(env, '_compare_to_benchmark'):
                benchmark = env._compare_to_benchmark('BESG ESG Score', next_state[esg_idx] if esg_idx is not None else 0)
                episode_results['benchmark_category'].append(benchmark)
            else:
                episode_results['benchmark_category'].append(None)

            print(f"\nStep {step+1}: Action = {env.actions[action]['name']} (Pillar: {env.actions[action].get('pillar', 'None')})")
            print(f"Reward = {reward:.2f}, ESG Score Change: {info['esg_improvement']:.2f}")
            if 'financial_impact' in info and info['financial_impact'] != 0:
                print(f"Financial Impact: {info['financial_impact']:.2f}")
            env.render()

            state = next_state
            total_reward += reward
            step += 1

        results.append(episode_results)
        print(f"\nEpisode {i+1} - Total Reward: {total_reward:.2f}")
        print(f"Final ESG Score: {state[esg_idx] if esg_idx is not None else 0:.2f} "
              f"(Started at {env.initial_state[esg_idx] if esg_idx is not None else 0:.2f})")

        # Calculate final pillar balance
        env_final = state[env_idx] if env_idx is not None else 0
        soc_final = state[soc_idx] if soc_idx is not None else 0
        gov_final = state[gov_idx] if gov_idx is not None else 0

        print(f"Final Pillar Scores - Env: {env_final:.2f}, Social: {soc_final:.2f}, Gov: {gov_final:.2f}")
        print(f"Final Pillar Imbalance: {episode_results['pillar_imbalance'][-1]:.3f}")

        # Show final benchmark position if available
        if episode_results['benchmark_category'][-1]:
            benchmark = episode_results['benchmark_category'][-1].replace('_', ' ').upper()
            print(f"Industry Position: {benchmark}")

    # Create enhanced visualizations
    for i, episode_results in enumerate(results):
        # Create a DataFrame for easier plotting
        results_df = pd.DataFrame(episode_results)

        # Enhanced visualization with 6 subplots
        plt.figure(figsize=(18, 12))

        # 1. Overall ESG Score Progression
        plt.subplot(2, 3, 1)
        plt.plot(results_df['step'], results_df['esg_score'], marker='o', linestyle='-', linewidth=2)
        plt.title('Overall ESG Score Progression', fontsize=14)
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('ESG Score', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 2. ESG Pillar Scores
        plt.subplot(2, 3, 2)
        plt.plot(results_df['step'], results_df['env_score'], marker='o', color='green', label='Environmental')
        plt.plot(results_df['step'], results_df['social_score'], marker='o', color='blue', label='Social')
        plt.plot(results_df['step'], results_df['gov_score'], marker='o', color='purple', label='Governance')
        plt.title('ESG Pillar Scores', fontsize=14)
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Pillar Score', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 3. Financial Impact (Net Income)
        plt.subplot(2, 3, 3)
        plt.plot(results_df['step'], results_df['net_income'], marker='o', color='red')
        plt.title('Financial Impact (Net Income)', fontsize=14)
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Net Income', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 4. Rewards per Action
        plt.subplot(2, 3, 4)
        plt.plot(results_df['step'], results_df['reward'], marker='o', color='orange')
        plt.title('Rewards per Action', fontsize=14)
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 5. Pillar Imbalance Over Time
        plt.subplot(2, 3, 5)
        plt.plot(results_df['step'], results_df['pillar_imbalance'], marker='o', color='brown')
        plt.title('Pillar Imbalance Over Time', fontsize=14)
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Imbalance Score', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 6. Action Distribution by Pillar
        plt.subplot(2, 3, 6)
        pillar_counts = results_df['pillar'].value_counts()
        if None in pillar_counts:
            pillar_counts = pillar_counts.drop(None)
        colors = {'environmental': 'green', 'social': 'blue', 'governance': 'purple', 'all': 'orange'}
        pillar_colors = [colors.get(p, 'gray') for p in pillar_counts.index]
        plt.bar(pillar_counts.index, pillar_counts.values, color=pillar_colors)
        plt.title('Action Distribution by Pillar', fontsize=14)
        plt.xlabel('Pillar', fontsize=12)
        plt.ylabel('Number of Actions', fontsize=12)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/episode_{i+1}_results.png')
        plt.show()

        # Create additional plot for action effects
        plt.figure(figsize=(12, 10))

        # ESG improvement per action
        plt.subplot(2, 2, 1)
        action_df = results_df[1:].copy()  # Skip initial state
        plt.bar(action_df['action'], action_df['improvement'], color='blue')
        plt.title('ESG Improvement per Action', fontsize=14)
        plt.xlabel('Action', fontsize=12)
        plt.ylabel('ESG Score Improvement', fontsize=12)
        plt.xticks(rotation=90)

        # Financial impact comparison (NEW)
        plt.subplot(2, 2, 2)
        plt.bar(action_df['action'], action_df['financial_impact'], color='red')
        plt.title('Financial Impact per Action', fontsize=14)
        plt.xlabel('Action', fontsize=12)
        plt.ylabel('Financial Impact', fontsize=12)
        plt.xticks(rotation=90)

        # Efficiency (improvement/cost) with financial adjustment (NEW)
        plt.subplot(2, 2, 3)
        # Adjust for positive financial impacts
        adjusted_cost = action_df['cost'] - action_df['financial_impact']
        # Avoid division by zero
        adjusted_cost = adjusted_cost.replace(0, 1)
        efficiency = action_df['improvement'] / adjusted_cost
        plt.bar(action_df['action'], efficiency, color='green')
        plt.title('Efficiency (Improvement per Adjusted Cost)', fontsize=14)
        plt.xlabel('Action', fontsize=12)
        plt.ylabel('Efficiency', fontsize=12)
        plt.xticks(rotation=90)

        # Weakest pillar over time
        plt.subplot(2, 2, 4)
        weakest_counts = pd.Series(results_df['weakest_pillar']).value_counts()
        colors = {'environmental': 'green', 'social': 'blue', 'governance': 'purple'}
        weakest_colors = [colors.get(p, 'gray') for p in weakest_counts.index]
        plt.pie(weakest_counts, labels=weakest_counts.index, autopct='%1.1f%%', colors=weakest_colors)
        plt.title('Distribution of Weakest Pillar', fontsize=14)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/episode_{i+1}_action_analysis.png')
        plt.show()

        # Create a table showing actions taken
        action_table = results_df.copy()
        action_table = action_table.rename(columns={
            'step': 'Step',
            'action': 'Action',
            'pillar': 'Pillar',
            'esg_score': 'ESG Score',
            'env_score': 'Env Score',
            'social_score': 'Social Score',
            'gov_score': 'Gov Score',
            'reward': 'Reward',
            'improvement': 'Improvement',
            'cost': 'Cost',
            'financial_impact': 'Financial Impact',
            'benchmark_category': 'Industry Position'
        })

        # Print the action table
        print("\nActions Taken:")
        print(action_table[['Step', 'Action', 'Pillar', 'ESG Score', 'Env Score', 'Social Score', 'Gov Score', 'Reward', 'Financial Impact']])

        # Generate enhanced strategic insights with data-driven recommendations
        print("\n=== DATA-DRIVEN ESG STRATEGY INSIGHTS ===")

        # 1. Analyze the most effective actions
        if len(action_df) > 0:
            # Include financial impacts in efficiency calculation
            action_df['adjusted_cost'] = action_df['cost'] - action_df['financial_impact']
            action_df['adjusted_cost'] = action_df['adjusted_cost'].replace(0, 1)  # Avoid division by zero
            action_df['efficiency'] = action_df['improvement'] / action_df['adjusted_cost']
            top_actions = action_df.sort_values('efficiency', ascending=False).head(3)

            print("\nMost Effective Actions:")
            for _, row in top_actions.iterrows():
                print(f"- {row['action']} (Pillar: {row['pillar']})")
                print(f"  • ESG Improvement: {row['improvement']:.3f}")
                print(f"  • Cost: {row['cost']:.2f}")
                print(f"  • Financial Impact: {row['financial_impact']:.2f}")
                print(f"  • Efficiency: {row['efficiency']:.4f}")

        # 2. Analyze pillar balance impact with data-driven recommendation
        initial_imbalance = results_df['pillar_imbalance'].iloc[0]
        final_imbalance = results_df['pillar_imbalance'].iloc[-1]

        # Get pillar balance recommendation
        if hasattr(env, 'get_pillar_balance_recommendation'):
            balance_rec = env.get_pillar_balance_recommendation()
            balance_approach = balance_rec['recommendation']

            if balance_approach == 'balanced':
                if final_imbalance < initial_imbalance:
                    print(f"\nPillar Balance: IMPROVED from {initial_imbalance:.3f} to {final_imbalance:.3f} (aligned with data recommendation)")
                else:
                    print(f"\nPillar Balance: WORSENED from {initial_imbalance:.3f} to {final_imbalance:.3f} (data recommends balanced approach)")
            else:  # focused approach recommended
                if final_imbalance > initial_imbalance:
                    print(f"\nPillar Focus: Increased focus from {initial_imbalance:.3f} to {final_imbalance:.3f} (aligned with data recommendation)")
                else:
                    print(f"\nPillar Focus: Decreased focus from {initial_imbalance:.3f} to {final_imbalance:.3f} (data recommends focused approach)")
        else:
            # Generic message without data-driven recommendation
            if final_imbalance < initial_imbalance:
                print(f"\nPillar Balance: IMPROVED from {initial_imbalance:.3f} to {final_imbalance:.3f}")
            elif final_imbalance > initial_imbalance:
                print(f"\nPillar Balance: WORSENED from {initial_imbalance:.3f} to {final_imbalance:.3f}")
            else:
                print(f"\nPillar Balance: UNCHANGED at {final_imbalance:.3f}")

        # 3. Analyze financial impact
        initial_income = results_df['net_income'].iloc[0]
        final_income = results_df['net_income'].iloc[-1]
        income_change = final_income - initial_income

        if income_change < 0:
            print(f"Financial Impact: Cost of {abs(income_change):.2f} (ROI: {(results_df['esg_score'].iloc[-1] - results_df['esg_score'].iloc[0]) / abs(income_change):.4f} ESG points per unit)")
        else:
            print(f"Financial Impact: POSITIVE change of {income_change:.2f}")

        # 4. Benchmark progress
        if 'benchmark_category' in results_df.columns and results_df['benchmark_category'].iloc[0] is not None:
            initial_benchmark = results_df['benchmark_category'].iloc[0]
            final_benchmark = results_df['benchmark_category'].iloc[-1]

            if initial_benchmark != final_benchmark:
                print(f"Industry Position: Moved from {initial_benchmark.replace('_', ' ').upper()} to {final_benchmark.replace('_', ' ').upper()}")
            else:
                print(f"Industry Position: Remained at {final_benchmark.replace('_', ' ').upper()}")

        # 5. Data-driven recommendations for further improvement
        print("\nFurther Recommendations based on Data Analysis:")

        # NEW: Get recommendation based on current ESG score
        current_esg = results_df['esg_score'].iloc[-1]
        if hasattr(env, 'get_improvement_factors'):
            improvement_factors = env.get_improvement_factors()
            if improvement_factors:
                best_range = max(improvement_factors.items(), key=lambda x: x[1])
                best_min, best_max = best_range[0]

                if current_esg < best_min:
                    print(f"- Focus on reaching the high-improvement ESG range ({best_min}-{best_max})")
                elif current_esg > best_max:
                    print(f"- Consider more ambitious ESG targets as you've surpassed the highest-improvement range")
                else:
                    print(f"- Current ESG score is in the optimal improvement range ({best_min}-{best_max})")

        # Identify areas that still need work
        final_state = {
            'env': results_df['env_score'].iloc[-1],
            'social': results_df['social_score'].iloc[-1],
            'governance': results_df['gov_score'].iloc[-1]
        }

        weakest_final = min(final_state.items(), key=lambda x: x[1])

        print(f"- Focus next on {weakest_final[0].upper()} pillar (current score: {weakest_final[1]:.2f})")

        # Recommend specific actions based on what wasn't used
        if hasattr(env, 'actions'):
            used_actions = set(results_df['action'].tolist()[1:])  # Skip initial state
            unused_actions = []

            for i, action in env.actions.items():
                if i != 0 and action['name'] not in used_actions:
                    if action.get('pillar', '') == weakest_final[0]:
                        unused_actions.append((i, action['name'], action.get('initiative_type', '')))

            if unused_actions:
                # NEW: Check financial impacts to recommend most financially advantageous actions
                if hasattr(env, 'get_financial_impacts'):
                    financial_impacts = env.get_financial_impacts()
                    if financial_impacts:
                        # Sort unused actions by financial impact
                        financial_sorted = []
                        for i, name, initiative in unused_actions:
                            if initiative in financial_impacts:
                                impact = financial_impacts[initiative].get('immediate', -0.05)
                                financial_sorted.append((i, name, initiative, impact))

                        if financial_sorted:
                            # Sort by least negative/most positive financial impact
                            financial_sorted.sort(key=lambda x: x[3], reverse=True)
                            print("- Consider these financially optimal actions for this pillar:")
                            for i, name, initiative, impact in financial_sorted[:3]:  # Show top 3
                                impact_desc = "Positive" if impact >= 0 else "Negative"
                                print(f"  • {name} (Financial Impact: {impact_desc})")
                        else:
                            print("- Consider these unused actions for this pillar:")
                            for i, name, _ in unused_actions[:3]:  # Show top 3
                                print(f"  • {name}")
                    else:
                        print("- Consider these unused actions for this pillar:")
                        for i, name, _ in unused_actions[:3]:  # Show top 3
                            print(f"  • {name}")
                else:
                    print("- Consider these unused actions for this pillar:")
                    for i, name, _ in unused_actions[:3]:  # Show top 3
                        print(f"  • {name}")

        # Compare with similar companies if available
        if hasattr(env, 'get_similar_companies'):
            similar_companies = env.get_similar_companies()
            if similar_companies:
                print("\nLearn from similar companies:")
                for company in similar_companies[:2]:  # Show top 2
                    print(f"- {company}")

    return results


# Example usage
def run_data_driven_esg_optimization(data, company_idx=0, max_steps=8):
    """Run the complete data-driven ESG optimization pipeline"""
    # Create results directory
    results_dir = './esg_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create the environment
    env = DataDrivenESGEnvironment(
        data,
        company_idx=company_idx,
        max_steps=max_steps,
        scale_factor=0.1  # Only applied to scores > 10
    )

    # Display company information and insights
    company = env.company
    print(f"Analyzing company: {company}")
    print("\nInitial ESG Profile:")
    env.render()

    # Show data-driven insights
    insights = env.get_company_insights()
    print("\nData-Driven ESG Insights:")
    for insight in insights:
        print(f"- {insight}")

    # NEW: Show data-driven improvement patterns
    if hasattr(env, 'get_improvement_factors'):
        improvement_factors = env.get_improvement_factors()
        if improvement_factors:
            print("\nData-Driven ESG Improvement Patterns:")
            sorted_factors = sorted(improvement_factors.items(), key=lambda x: x[1], reverse=True)
            for (low, high), factor in sorted_factors:
                print(f"- ESG Score Range {low}-{high}: Improvement factor {factor:.2f}x")

    # NEW: Show pillar balance recommendation
    if hasattr(env, 'get_pillar_balance_recommendation'):
        balance_rec = env.get_pillar_balance_recommendation()
        print(f"\nData-Driven Strategy Recommendation: {balance_rec['recommendation'].upper()} approach")
        print(f"- Balance Advantage: {balance_rec['balance_advantage']:.3f}")
        print(f"- Optimal Balance Weight: {balance_rec['optimal_weight']:.2f}")

    # Train the agent
    print("\nTraining ESG Optimization Agent...")
    scores, agent, pillar_improvements = data_driven_training(
        env,
        n_episodes=300,
        checkpoint_dir=f'{results_dir}/checkpoints',
        curriculum_learning=True
    )

    # Visualize training progress
    plt.figure(figsize=(15, 5))
    plt.plot(scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig(f'{results_dir}/training_progress.png')
    plt.show()

    # Run and visualize the optimized ESG strategy
    print("\nVisualizing Optimized ESG Strategy...")
    eval_env = DataDrivenESGEnvironment(
        data,
        company_idx=company_idx,
        max_steps=max_steps,
        scale_factor=0.1  # Only applied to scores > 10
    )

    results = visualize_esg_strategy(
        eval_env,
        agent,
        save_dir=f'{results_dir}/visualizations'
    )

    return results, agent, env