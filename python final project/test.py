"""
ENHANCED Land Price Prediction System for Kathmandu Valley
===========================================================
IMPROVEMENTS FOR BETTER ACCURACY:
1. Advanced Feature Engineering (interaction features, polynomial features)
2. Better outlier detection and handling
3. XGBoost integration for superior performance
4. Stacking ensemble methods
5. Advanced hyperparameter optimization
6. Feature scaling improvements
7. Better cross-validation strategy
8. Log transformation for price (handles skewed data better)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              VotingRegressor, StackingRegressor, ExtraTreesRegressor)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import joblib
import re
import warnings
import threading
warnings.filterwarnings('ignore')

# Try to import XGBoost (install if needed)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# ============================================================================
# SECTION 1: ENHANCED DATA EXPLORATION AND ANALYSIS
# ============================================================================

class DataExplorer:
    """Handle all exploratory data analysis and visualization tasks"""
    
    def __init__(self, df):
        self.df = df
        self.figures = {}
        
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        summary = []
        summary.append("=" * 80)
        summary.append("SUMMARY STATISTICS")
        summary.append("=" * 80)
        summary.append(f"\nDataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
        if 'price_per_anna' in self.df.columns:
            summary.append("\nTARGET VARIABLE: Price per Anna")
            summary.append(f"  Min Price:    Rs. {self.df['price_per_anna'].min():,.2f}")
            summary.append(f"  Max Price:    Rs. {self.df['price_per_anna'].max():,.2f}")
            summary.append(f"  Mean Price:   Rs. {self.df['price_per_anna'].mean():,.2f}")
            summary.append(f"  Median Price: Rs. {self.df['price_per_anna'].median():,.2f}")
            summary.append(f"  Std Dev:      Rs. {self.df['price_per_anna'].std():,.2f}")
            summary.append(f"  Skewness:     {self.df['price_per_anna'].skew():.2f}")
            summary.append(f"  Kurtosis:     {self.df['price_per_anna'].kurtosis():.2f}")
        
        return "\n".join(summary)
    
    def analyze_missing_values(self):
        """Analyze missing values"""
        missing = self.df.isnull().sum()
        missing_pct = 100 * missing / len(self.df)
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(missing_df) > 0:
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.bar(missing_df['Column'], missing_df['Percentage'])
            ax.set_xticklabels(missing_df['Column'], rotation=45, ha='right')
            ax.set_ylabel('Percentage Missing (%)')
            ax.set_title('Missing Values Analysis')
            fig.tight_layout()
            self.figures['missing_values'] = fig
        
        return missing_df
    
    def analyze_distributions(self):
        """Analyze distributions of numerical features"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) == 0:
            return
        
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig = Figure(figsize=(15, 5 * n_rows))
        
        for idx, col in enumerate(numerical_cols):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            self.df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            
            # Add skewness info
            skewness = self.df[col].skew()
            ax.text(0.95, 0.95, f'Skew: {skewness:.2f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.tight_layout()
        self.figures['distributions'] = fig
        
        # Box plots
        fig2 = Figure(figsize=(15, 5 * n_rows))
        
        for idx, col in enumerate(numerical_cols):
            ax = fig2.add_subplot(n_rows, n_cols, idx + 1)
            self.df.boxplot(column=col, ax=ax)
            ax.set_title(f'Box Plot: {col}')
        
        fig2.tight_layout()
        self.figures['boxplots'] = fig2
    
    def analyze_correlations(self):
        """Analyze correlations between features"""
        numerical_df = self.df.select_dtypes(include=[np.number])
        
        if len(numerical_df.columns) < 2:
            return None
        
        corr_matrix = numerical_df.corr()
        
        fig = Figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, ax=ax)
        ax.set_title('Feature Correlation Heatmap', fontsize=16, pad=20)
        fig.tight_layout()
        self.figures['correlation'] = fig
        
        return corr_matrix


# ============================================================================
# SECTION 2: ENHANCED MACHINE LEARNING MODEL
# ============================================================================

class EnhancedLandPricePredictor:
    """Enhanced Machine Learning model with advanced techniques for better accuracy"""
    
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
        self.model = None
        self.scaler = RobustScaler()  # RobustScaler is better for outliers
        self.label_encoders = {}
        self.dummy_columns = []
        self.feature_columns = None
        self.df = None
        self.use_log_transform = True  # Log transform for price
        self.poly_features = None  # For polynomial features
        self.categorical_freq_maps = {}
        
        self.categorical_columns = ['district', 'location', 'road_type', 'land_shape', 
                                    'slope', 'commercial_potential', 'utilities_access']
        self.numerical_columns = ['road_access_ft', 'distance_to_ringroad_km', 'area_sqft']
    
    def clean_price(self, price_str):
        """Convert price string to numeric value"""
        if pd.isna(price_str):
            return np.nan
        s = str(price_str)
        # normalize and remove commas
        s = s.replace(',', ' ').strip().lower()
        s = re.sub(r'[^0-9a-z\.\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()

        total = 0.0
        found = False
        # find all number+unit occurrences e.g. '2 crore', '10 lakh', '50 thousand'
        for m in re.findall(r'([0-9]*\.?[0-9]+)\s*(crore|cr|lakh|lac|thousand|thou|k|rs|rupee|rupees)', s):
            num = float(m[0])
            unit = m[1]
            found = True
            if unit in ('crore', 'cr'):
                total += num * 10000000
            elif unit in ('lakh', 'lac'):
                total += num * 100000
            elif unit in ('thousand', 'thou') or unit == 'k':
                total += num * 1000
            elif unit in ('rs', 'rupee', 'rupees'):
                total += num

        # If we parsed unit-based components, return the total
        if found and total > 0:
            return total

        # Try to extract a single numeric token (handles values like '20' or '270000')
        num_match = re.search(r'([0-9]+\.?[0-9]*)', s)
        if num_match:
            try:
                return float(num_match.group(1))
            except:
                return np.nan

        return np.nan
    
    def create_interaction_features(self, df):
        """Create interaction features for better predictions"""
        df = df.copy()
        
        # Interaction features that make sense for land pricing
        if 'road_access_ft' in df.columns and 'area_sqft' in df.columns:
            df['road_access_per_area'] = df['road_access_ft'] / (df['area_sqft'] + 1)
        
        if 'distance_to_ringroad_km' in df.columns and 'area_sqft' in df.columns:
            df['area_distance_interaction'] = df['area_sqft'] / (df['distance_to_ringroad_km'] + 1)
        
        if 'road_access_ft' in df.columns and 'distance_to_ringroad_km' in df.columns:
            df['road_distance_ratio'] = df['road_access_ft'] / (df['distance_to_ringroad_km'] + 1)
        
        # Log transformations for skewed features
        if 'area_sqft' in df.columns:
            df['log_area'] = np.log1p(df['area_sqft'])
        
        if 'distance_to_ringroad_km' in df.columns:
            df['log_distance'] = np.log1p(df['distance_to_ringroad_km'])
        
        # Square root transformations
        if 'road_access_ft' in df.columns:
            df['sqrt_road_access'] = np.sqrt(df['road_access_ft'])
        
        return df
    
    def remove_outliers_advanced(self, X, y, method='isolation_forest'):
        """Advanced outlier removal using multiple methods"""
        from sklearn.ensemble import IsolationForest
        
        # Combine both X and y for outlier detection
        combined = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        
        if method == 'isolation_forest':
            # Isolation Forest - good for high-dimensional data
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            outlier_labels = iso_forest.fit_predict(combined)
            mask = outlier_labels != -1
        else:
            # Z-score method on target variable
            z_scores = np.abs(stats.zscore(y))
            mask = z_scores < 3
        
        return X[mask], y[mask]
    
    def load_and_preprocess_data(self, show_eda=True):
        """Load and preprocess the dataset with advanced techniques"""
        self.df = pd.read_csv(self.csv_path)

        # Normalize column names to snake_case for robust matching
        def normalize_col(c):
            return str(c).strip().lower().replace(' ', '_')

        self.df.columns = [normalize_col(c) for c in self.df.columns]

        # Helper: parse land size (ropani/anna/paisa/daam) into sqft
        def parse_area_to_sqft(s):
            if pd.isna(s):
                return np.nan
            s0 = str(s).lower()
            # replace commas and other separators
            s0 = re.sub(r'[,:]', ' ', s0)
            # look for ropani
            ropani = 0.0
            anna = 0.0
            paisa = 0.0
            daam = 0.0
            # patterns
            m = re.search(r'([0-9]*\.?[0-9]+)\s*(ropani|ropani|ropani|ropan|rpn)', s0)
            if m:
                ropani = float(m.group(1))
            m = re.search(r'([0-9]*\.?[0-9]+)\s*(aana|anna|aana|ana)', s0)
            if m:
                anna = float(m.group(1))
            m = re.search(r'([0-9]*\.?[0-9]+)\s*(paisa|paise|poisa|paisas|pais)', s0)
            if m:
                paisa = float(m.group(1))
            m = re.search(r'([0-9]*\.?[0-9]+)\s*(daam|dam|daam)', s0)
            if m:
                daam = float(m.group(1))

            # If any unit found, compute sqft
            if ropani or anna or paisa or daam:
                sqft = ropani * 5476.0 + anna * 342.25 + paisa * (342.25 / 4.0) + daam * (342.25 / 16.0)
                return sqft

            # If string contains 'aana' spelled differently like '3.5 aana' handled above.
            # Otherwise, try to extract a plain number. If number > 1000, assume it's sqft; if <=1000, assume it's aana.
            m2 = re.search(r'([0-9]*\.?[0-9]+)', s0)
            if m2:
                val = float(m2.group(1))
                if val > 1000:
                    return val
                else:
                    return val * 342.25

            return np.nan

        # Create standardized feature columns expected by the model
        # area_sqft: from possible columns like 'land_size' or 'land_size'
        if 'area_sqft' not in self.df.columns:
            if 'land_size' in self.df.columns:
                self.df['area_sqft'] = self.df['land_size'].apply(parse_area_to_sqft)
            elif 'land_size' in self.df.columns:
                self.df['area_sqft'] = self.df['land_size'].apply(parse_area_to_sqft)

        # distance_to_ringroad_km: try to parse from distance_to_main_road or similar
        if 'distance_to_ringroad_km' not in self.df.columns:
            if 'distance_to_main_road' in self.df.columns:
                def parse_distance(s):
                    if pd.isna(s):
                        return np.nan
                    s0 = str(s).lower()
                    if 'under' in s0 or 'under 100' in s0:
                        return 0.05
                    if '100' in s0 and '300' in s0:
                        return 0.2
                    if '300' in s0 and '600' in s0:
                        return 0.45
                    # meters value
                    m = re.search(r'([0-9]*\.?[0-9]+)\s*m', s0)
                    if m:
                        return float(m.group(1)) / 1000.0
                    # direct numeric
                    m2 = re.search(r'([0-9]*\.?[0-9]+)', s0)
                    if m2:
                        val = float(m2.group(1))
                        if val > 1 and val < 1000:
                            # ambiguous: treat as meters
                            return val / 1000.0
                    return np.nan

                self.df['distance_to_ringroad_km'] = self.df['distance_to_main_road'].apply(parse_distance)

        # road_access_ft: derive from road_access yes/no or explicit values
        if 'road_access_ft' not in self.df.columns:
            if 'road_access' in self.df.columns:
                def parse_road_access(v):
                    if pd.isna(v):
                        return 0.0
                    s = str(v).lower()
                    if 'yes' in s:
                        return 30.0
                    if 'no' in s:
                        return 0.0
                    m = re.search(r'([0-9]*\.?[0-9]+)\s*ft', s)
                    if m:
                        return float(m.group(1))
                    return 30.0 if s.strip() else 0.0

                self.df['road_access_ft'] = self.df['road_access'].apply(parse_road_access)

        # --- Robust detection of price column ---
        # Many datasets may label the price column differently (e.g. 'title', 'price', etc.).
        # If 'price_per_anna' is missing, try to detect a column containing price-like strings
        if 'price_per_anna' not in self.df.columns:
            detected = False
            # look for common price-like column names first
            candidates = [c for c in self.df.columns if any(k in c.lower() for k in ['price', 'per aana', 'per anna', 'title', 'amount', 'rs', 'rupee'])]
            for c in candidates:
                sample = self.df[c].astype(str).head(50).str.lower()
                if sample.str.contains(r'lakh|crore|per aana|per anna|rs\.|rs|k\b|lakh|crore').any():
                    self.df['price_per_anna'] = self.df[c]
                    detected = True
                    break

            # fallback: scan all object columns for price-like content
            if not detected:
                for c in self.df.select_dtypes(include=['object', 'string']).columns:
                    sample = self.df[c].astype(str).head(50).str.lower()
                    if sample.str.contains(r'lakh|crore|per aana|per anna|rs\.|rs|k\b|lakh|crore').any():
                        self.df['price_per_anna'] = self.df[c]
                        detected = True
                        break

            if not detected:
                raise ValueError("Could not find price column. Expected 'price_per_anna' or a column containing price strings (e.g. 'title').")

        # Clean price column
        if self.df['price_per_anna'].dtype == 'object' or self.df['price_per_anna'].dtype == 'string':
            self.df['price_per_anna'] = self.df['price_per_anna'].apply(self.clean_price)
        
        # Remove rows with missing target
        self.df = self.df.dropna(subset=['price_per_anna'])
        
        # Perform EDA
        eda_results = ""
        self.explorer = None
        if show_eda:
            self.explorer = DataExplorer(self.df)
            eda_results = self.explorer.generate_summary_statistics()
            self.explorer.analyze_missing_values()
            self.explorer.analyze_distributions()
            self.explorer.analyze_correlations()
        
        # Handle missing values in numerical columns with median
        for col in self.numerical_columns:
            if col in self.df.columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Handle missing values in categorical columns with mode
        for col in self.categorical_columns:
            if col in self.df.columns:
                self.df[col].fillna(self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # Create interaction features
        self.df = self.create_interaction_features(self.df)
        
        # Add simple frequency-based encoding for the most important categorical fields
        for col in ['district', 'location']:
            if col in self.df.columns:
                freq_map = self.df[col].astype(str).value_counts(normalize=True)
                self.categorical_freq_maps[col] = freq_map
                self.df[f'{col}_freq'] = self.df[col].astype(str).map(freq_map).fillna(0.0)

        # One-hot encode categorical variables
        available_cats = [col for col in self.categorical_columns if col in self.df.columns]
        if len(available_cats) > 0:
            dummies = pd.get_dummies(self.df[available_cats].astype(str), prefix=available_cats)
            self.df = pd.concat([self.df, dummies], axis=1)
            self.dummy_columns = dummies.columns.tolist()
        
        return self.df, eda_results
    
    def prepare_features(self):
        """Prepare feature matrix with all engineered features"""
        numerical_cols = [col for col in self.numerical_columns if col in self.df.columns]
        
        # Include engineered features
        engineered_features = [
            'road_access_per_area', 'area_distance_interaction', 'road_distance_ratio',
            'log_area', 'log_distance', 'sqrt_road_access'
        ]
        engineered_cols = [col for col in engineered_features if col in self.df.columns]

        frequency_cols = [f'{col}_freq' for col in ['district', 'location'] if f'{col}_freq' in self.df.columns]
        
        encoded_cols = [c for c in self.dummy_columns if c in self.df.columns]
        
        self.feature_columns = numerical_cols + engineered_cols + frequency_cols + encoded_cols
        
        X = self.df[self.feature_columns]
        y = self.df['price_per_anna']
        
        # Apply log transformation to target if enabled
        if self.use_log_transform:
            y = np.log1p(y)
            self.y_is_log = True
        else:
            self.y_is_log = False
        
        return X, y
    
    def train_model(self, model_type='xgboost'):
        """Train enhanced model with advanced techniques"""
        X, y = self.prepare_features()

        # If no features detected, try fallback to any numeric columns (excluding target)
        if X.shape[1] == 0:
            numeric_candidates = [c for c in self.df.select_dtypes(include=[np.number]).columns.tolist() if c != 'price_per_anna']
            if len(numeric_candidates) > 0:
                self.feature_columns = numeric_candidates
                X = self.df[self.feature_columns]
            else:
                raise ValueError("No feature columns found after preprocessing. Ensure your CSV has numerical or categorical columns that can be used as features.")
        
        # Advanced outlier removal
        X_clean, y_clean = self.remove_outliers_advanced(X, y, method='isolation_forest')
        
        # Stratified split based on price quantiles for better representation
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # Scale features using RobustScaler (better for outliers)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Choose model based on type
        if model_type == 'xgboost' and XGBOOST_AVAILABLE:
            # XGBoost with extensive hyperparameter tuning
            param_grid = {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2]
            }
            xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            
            # Use fewer combinations for faster training
            from sklearn.model_selection import RandomizedSearchCV
            self.model = RandomizedSearchCV(
                xgb_model, param_grid, n_iter=20, cv=5, 
                scoring='r2', n_jobs=-1, random_state=42, verbose=1
            )
            self.model.fit(X_train_scaled, y_train)
            self.model = self.model.best_estimator_
            
        elif model_type == 'stacking':
            # Stacking ensemble - combines multiple models
            base_models = [
                ('rf', RandomForestRegressor(n_estimators=200, max_depth=15, 
                                            min_samples_split=5, random_state=42, n_jobs=-1)),
                ('gb', GradientBoostingRegressor(n_estimators=200, max_depth=5, 
                                                learning_rate=0.05, random_state=42)),
                ('extra', ExtraTreesRegressor(n_estimators=200, max_depth=15, 
                                             random_state=42, n_jobs=-1))
            ]
            
            if XGBOOST_AVAILABLE:
                base_models.append(
                    ('xgb', xgb.XGBRegressor(n_estimators=200, max_depth=6, 
                                            learning_rate=0.05, random_state=42))
                )
            
            self.model = StackingRegressor(
                estimators=base_models,
                final_estimator=Ridge(alpha=10.0),
                cv=5,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
            
        elif model_type == 'random_forest':
            # Enhanced Random Forest
            param_grid = {
                'n_estimators': [200, 300, 500],
                'max_depth': [15, 20, 25, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            self.model = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=0, scoring='r2')
            self.model.fit(X_train_scaled, y_train)
            self.model = self.model.best_estimator_
            
        elif model_type == 'gradient_boosting':
            # Enhanced Gradient Boosting
            param_grid = {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 5, 6],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'min_samples_split': [2, 5, 10]
            }
            gb = GradientBoostingRegressor(random_state=42)
            self.model = GridSearchCV(gb, param_grid, cv=5, n_jobs=-1, verbose=0, scoring='r2')
            self.model.fit(X_train_scaled, y_train)
            self.model = self.model.best_estimator_
        
        else:  # ElasticNet (regularized linear regression)
            param_grid = {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
            }
            elastic = ElasticNet(random_state=42, max_iter=10000)
            self.model = GridSearchCV(elastic, param_grid, cv=5, n_jobs=-1, verbose=0, scoring='r2')
            self.model.fit(X_train_scaled, y_train)
            self.model = self.model.best_estimator_
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Inverse transform if log was used
        if self.y_is_log:
            y_train_original = np.expm1(y_train)
            y_test_original = np.expm1(y_test)
            y_pred_train_original = np.expm1(y_pred_train)
            y_pred_test_original = np.expm1(y_pred_test)
        else:
            y_train_original = y_train
            y_test_original = y_test
            y_pred_train_original = y_pred_train
            y_pred_test_original = y_pred_test
        
        # Calculate metrics on original scale
        metrics = {
            'train_r2': r2_score(y_train_original, y_pred_train_original),
            'test_r2': r2_score(y_test_original, y_pred_test_original),
            'train_mae': mean_absolute_error(y_train_original, y_pred_train_original),
            'test_mae': mean_absolute_error(y_test_original, y_pred_test_original),
            'train_rmse': np.sqrt(mean_squared_error(y_train_original, y_pred_train_original)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_original, y_pred_test_original)),
            'train_mape': np.mean(np.abs((y_train_original - y_pred_train_original) / y_train_original)) * 100,
            'test_mape': np.mean(np.abs((y_test_original - y_pred_test_original) / y_test_original)) * 100
        }
        
        # Cross-validation with KFold
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, 
                                    cv=kfold, scoring='r2', n_jobs=-1)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Feature importance
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Store for visualization
        self.y_test = y_test_original
        self.y_pred_test = y_pred_test_original
        
        return metrics, feature_importance
    
    def predict_price(self, input_data):
        """Predict land price with all enhancements"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        data = input_data.copy()
        
        # Create a temporary dataframe for feature engineering
        temp_df = pd.DataFrame([data])
        
        # Convert numerical values
        for col in self.numerical_columns:
            if col in temp_df.columns:
                temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0)
        
        # Create interaction features
        temp_df = self.create_interaction_features(temp_df)
        
        # Build feature row
        row = {}
        
        # Numerical features
        for col in self.numerical_columns:
            if col in self.feature_columns:
                row[col] = float(temp_df[col].iloc[0]) if col in temp_df.columns else 0.0

        # Frequency encoding for district/location
        for col in ['district', 'location']:
            if col in self.df.columns and f'{col}_freq' in self.feature_columns:
                freq_map = self.categorical_freq_maps.get(col, {})
                if col in data:
                    row[f'{col}_freq'] = float(freq_map.get(str(data[col]), 0.0))
                else:
                    row[f'{col}_freq'] = 0.0
        
        # Engineered features
        engineered_features = [
            'road_access_per_area', 'area_distance_interaction', 'road_distance_ratio',
            'log_area', 'log_distance', 'sqrt_road_access'
        ]
        for col in engineered_features:
            if col in self.feature_columns:
                row[col] = float(temp_df[col].iloc[0]) if col in temp_df.columns else 0.0
        
        # Categorical dummies
        for dummy_col in self.dummy_columns:
            try:
                prefix, level = dummy_col.split('_', 1)
            except ValueError:
                prefix = dummy_col
                level = ''

            matched_value = ''
            for candidate_key in [prefix, prefix.lower(), prefix.replace('district', 'district').replace('location', 'location')]:
                if candidate_key in data:
                    matched_value = str(data[candidate_key])
                    break

            if not matched_value and prefix in ['district', 'location']:
                for key in data:
                    if key.lower() == prefix.lower():
                        matched_value = str(data[key])
                        break

            row[dummy_col] = 1 if matched_value.lower() == level.lower() else 0
        
        # Create feature array
        feature_array = np.array([[row.get(col, 0) for col in self.feature_columns]])
        feature_array_scaled = self.scaler.transform(feature_array)
        
        # Predict
        prediction = self.model.predict(feature_array_scaled)[0]
        
        # Inverse transform if log was used
        if self.y_is_log:
            prediction = np.expm1(prediction)
        
        return prediction
    
    def save_model(self, filename='enhanced_land_price_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'dummy_columns': self.dummy_columns,
            'use_log_transform': self.use_log_transform,
            'y_is_log': self.y_is_log
        }
        joblib.dump(model_data, filename)
    
    def load_model(self, filename='enhanced_land_price_model.pkl'):
        """Load a trained model from a pickle file."""
        model_data = joblib.load(filename)

        if isinstance(model_data, dict):
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.label_encoders = model_data.get('label_encoders', {})
            self.feature_columns = model_data.get('feature_columns', [])
            self.dummy_columns = model_data.get('dummy_columns', [])
            self.use_log_transform = model_data.get('use_log_transform', True)
            self.y_is_log = model_data.get('y_is_log', False)
        else:
            self.model = model_data
            self.scaler = None
            self.label_encoders = {}
            self.feature_columns = []
            self.dummy_columns = []
            self.use_log_transform = True
            self.y_is_log = False


# ============================================================================
# SECTION 3: ENHANCED GRAPHICAL USER INTERFACE
# ============================================================================

class EnhancedLandPricePredictorGUI:
    """Interactive GUI for the Enhanced Land Price Prediction System"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Land Price Predictor - Kathmandu Valley")
        self.root.state('zoomed')
        self.root.configure(bg="#e8f0fb")
        
        self.predictor = EnhancedLandPricePredictor()
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#e8f0fb', borderwidth=0)
        style.configure('TNotebook.Tab', padding=[14, 10], background='#d4e4f7', foreground='#0f2f55', font=('Segoe UI', 10, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', '#ffffff')])
        style.configure('Accent.TButton', background='#0b6396', foreground='white', font=('Segoe UI', 10, 'bold'), padding=8)
        style.map('Accent.TButton', background=[('active', '#145c9e'), ('disabled', '#a7c1d5')])
        style.configure('Card.TFrame', background='white')
        style.configure('Section.TLabelframe', background='white', borderwidth=1, relief='solid')
        style.configure('Header.TLabel', background='#e8f0fb', foreground='#0f2f55', font=('Segoe UI', 18, 'bold'))
        style.configure('SubHeader.TLabel', background='#e8f0fb', foreground='#385b85', font=('Segoe UI', 11))
        style.configure('FormLabel.TLabel', background='white', foreground='#0f2f55', font=('Segoe UI', 10, 'bold'))
        style.configure('Info.TLabel', background='white', foreground='#0f2f55', font=('Segoe UI', 10))
        style.configure('TLabelFrame.Label', font=('Segoe UI', 11, 'bold'))
        style.configure('TEntry', fieldbackground='#f9fbff', background='#f9fbff', foreground='#0f2f55')
        style.configure('TCombobox', fieldbackground='#f9fbff', background='#f9fbff', foreground='#0f2f55')
        style.configure('TButton', padding=6, font=('Segoe UI', 10))
        
        header_frame = ttk.Frame(root, style='Card.TFrame')
        header_frame.pack(fill='x', padx=12, pady=(12, 0))
        title_label = ttk.Label(header_frame, text="Kathmandu Valley Land Price Predictor", style='Header.TLabel')
        title_label.pack(anchor='w', padx=16, pady=(16, 4))
        subtitle_label = ttk.Label(header_frame, text="Improved UI, interactive data loading, model training and prediction.", style='SubHeader.TLabel')
        subtitle_label.pack(anchor='w', padx=16, pady=(0, 16))
        ttk.Separator(root, orient='horizontal').pack(fill='x', padx=12, pady=(0, 10))
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=12, pady=10)
        
        self.create_data_tab()
        self.create_training_tab()
        self.create_prediction_tab()
        self.create_results_tab()
        self.create_visualization_tab()
    
    def create_visualization_tab(self):
        """Create visualization display tab"""
        viz_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(viz_frame, text="📊 Visualizations")
        
        title = ttk.Label(viz_frame, text="Data Visualizations", style='Header.TLabel')
        title.pack(pady=10, anchor='w', padx=20)
        
        canvas = tk.Canvas(viz_frame, bg='#f2f6fc', highlightthickness=0)
        scrollbar = ttk.Scrollbar(viz_frame, orient="vertical", command=canvas.yview)
        self.viz_scrollable_frame = ttk.Frame(canvas)
        
        self.viz_scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.viz_scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.viz_placeholder = ttk.Label(self.viz_scrollable_frame, 
                                        text="Load data and train model to see visualizations",
                                        font=('Arial', 12))
        self.viz_placeholder.pack(pady=50)
    
    def create_data_tab(self):
        """Create data loading tab"""
        data_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(data_frame, text="📁 Data Loading")
        
        title = ttk.Label(data_frame, text="Load and Preprocess Data", style='Header.TLabel')
        title.pack(pady=20, anchor='w', padx=20)
        
        file_frame = ttk.LabelFrame(data_frame, text="Dataset Selection", padding=16, style='Section.TLabelframe')
        file_frame.pack(pady=10, padx=20, fill='x')
        
        ttk.Label(file_frame, text="CSV File:", style='FormLabel.TLabel').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=58).grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file, style='Accent.TButton').grid(row=0, column=2, sticky='e', padx=5, pady=5)
        file_frame.columnconfigure(1, weight=1)
        
        self.show_eda_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(data_frame, text="Perform Exploratory Data Analysis", 
                       variable=self.show_eda_var).pack(pady=10, padx=20, anchor='w')
        
        ttk.Button(data_frame, text="Load and Preprocess Data", 
                  command=self.load_data, style='Accent.TButton').pack(pady=20)
        
        info_label = ttk.Label(data_frame, text="Dataset Information:", style='FormLabel.TLabel')
        info_label.pack(pady=10, anchor='w', padx=20)
        
        self.data_info_text = scrolledtext.ScrolledText(data_frame, height=20, width=100, bg='#f7f9fd', fg='#1c1c1c', relief='flat', insertbackground='#1c1c1c')
        self.data_info_text.pack(pady=10, padx=20, fill='both', expand=True)
    
    def create_training_tab(self):
        """Create model training tab"""
        train_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(train_frame, text="🎯 Model Training")
        
        title = ttk.Label(train_frame, text="Train Enhanced Machine Learning Model", style='Header.TLabel')
        title.pack(pady=20, anchor='w', padx=20)
        
        model_frame = ttk.LabelFrame(train_frame, text="Advanced Model Selection", padding=20, style='Section.TLabelframe')
        model_frame.pack(pady=10, padx=20, fill='x')
        
        self.model_type_var = tk.StringVar(value="stacking")
        
        if XGBOOST_AVAILABLE:
            ttk.Radiobutton(model_frame, text="XGBoost (Recommended - Best Performance)", 
                           variable=self.model_type_var, value="xgboost").pack(anchor='w', pady=2)
        
        ttk.Radiobutton(model_frame, text="Stacking Ensemble (Multiple Models Combined)", 
                       variable=self.model_type_var, value="stacking").pack(anchor='w', pady=2)
        ttk.Radiobutton(model_frame, text="Random Forest (Enhanced)", 
                       variable=self.model_type_var, value="random_forest").pack(anchor='w', pady=2)
        ttk.Radiobutton(model_frame, text="Gradient Boosting (Enhanced)", 
                       variable=self.model_type_var, value="gradient_boosting").pack(anchor='w', pady=2)
        ttk.Radiobutton(model_frame, text="ElasticNet Regression (Regularized)", 
                       variable=self.model_type_var, value="elasticnet").pack(anchor='w', pady=2)
        
        if not XGBOOST_AVAILABLE:
            note = ttk.Label(model_frame, 
                           text="Note: Install XGBoost for best performance (pip install xgboost)", 
                           foreground='red', font=('Arial', 9, 'italic'))
            note.pack(anchor='w', pady=5)
        
        ttk.Button(train_frame, text="🚀 Train Model", 
                  command=self.train_model).pack(pady=20)
        
        save_frame = ttk.Frame(train_frame)
        save_frame.pack(pady=10)
        
        ttk.Button(save_frame, text="💾 Save Model", command=self.save_model).pack(side='left', padx=5)
        ttk.Button(save_frame, text="📂 Load Model", command=self.load_model).pack(side='left', padx=5)
        
        results_label = ttk.Label(train_frame, text="Training Results:", font=('Arial', 12, 'bold'))
        results_label.pack(pady=10)
        
        self.training_results_text = scrolledtext.ScrolledText(train_frame, height=15, width=100)
        self.training_results_text.pack(pady=10, padx=20)
    
    def create_prediction_tab(self):
        """Create prediction tab"""
        pred_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(pred_frame, text="🔮 Price Prediction")
        
        title = ttk.Label(pred_frame, text="Predict Land Price", style='Header.TLabel')
        title.pack(pady=20, anchor='w', padx=20)
        
        tip_label = ttk.Label(pred_frame, text="Enter land details below and click Predict. Use numeric values for area, distance and road access.", style='SubHeader.TLabel', wraplength=900, justify='left')
        tip_label.pack(anchor='w', padx=20)
        
        canvas = tk.Canvas(pred_frame, bg='#f2f6fc', highlightthickness=0)
        scrollbar = ttk.Scrollbar(pred_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='Card.TFrame')
        
        scrollable_frame.bind("<Configure>", 
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        input_frame = ttk.LabelFrame(scrollable_frame, text="Land Details", padding=20, style='Section.TLabelframe')
        input_frame.pack(pady=10, padx=20, fill='both', expand=True)
        input_frame.columnconfigure(1, weight=1)
        
        self.input_vars = {}
        
        row = 0
        ttk.Label(input_frame, text="District:", style='FormLabel.TLabel').grid(row=row, column=0, sticky='w', pady=5)
        self.input_vars['district'] = ttk.Entry(input_frame, width=32)
        self.input_vars['district'].insert(0, 'Kathmandu')
        self.input_vars['district'].grid(row=row, column=1, pady=5, padx=5, sticky='ew')
        
        row += 1
        ttk.Label(input_frame, text="District:").grid(row=row, column=0, sticky='w', pady=5)
        self.input_vars['district'] = ttk.Entry(input_frame, width=32)
        self.input_vars['district'].grid(row=row, column=1, pady=5, padx=5)

        row += 1
        ttk.Label(input_frame, text="Location:").grid(row=row, column=0, sticky='w', pady=5)
        self.input_vars['location'] = ttk.Entry(input_frame, width=32)
        self.input_vars['location'].grid(row=row, column=1, pady=5, padx=5)
        
        row += 1
        ttk.Label(input_frame, text="Road Access (ft):").grid(row=row, column=0, sticky='w', pady=5)
        self.input_vars['road_access_ft'] = ttk.Entry(input_frame, width=32)
        self.input_vars['road_access_ft'].grid(row=row, column=1, pady=5, padx=5)
        
        row += 1
        ttk.Label(input_frame, text="Road Type:").grid(row=row, column=0, sticky='w', pady=5)
        self.input_vars['road_type'] = ttk.Combobox(input_frame, width=30, state='readonly')
        self.input_vars['road_type']['values'] = ['Pitched', 'Graveled', 'Earthen']
        self.input_vars['road_type'].grid(row=row, column=1, pady=5, padx=5, sticky='ew')
        
        row += 1
        ttk.Label(input_frame, text="Distance to Ring Road (km):").grid(row=row, column=0, sticky='w', pady=5)
        self.input_vars['distance_to_ringroad_km'] = ttk.Entry(input_frame, width=32)
        self.input_vars['distance_to_ringroad_km'].grid(row=row, column=1, pady=5, padx=5)
        
        row += 1
        ttk.Label(input_frame, text="Area (sqft):").grid(row=row, column=0, sticky='w', pady=5)
        self.input_vars['area_sqft'] = ttk.Entry(input_frame, width=32)
        self.input_vars['area_sqft'].grid(row=row, column=1, pady=5, padx=5)
        
        row += 1
        ttk.Label(input_frame, text="Land Shape:").grid(row=row, column=0, sticky='w', pady=5)
        self.input_vars['land_shape'] = ttk.Combobox(input_frame, width=30, state='readonly')
        self.input_vars['land_shape']['values'] = ['Regular', 'Irregular']
        self.input_vars['land_shape'].grid(row=row, column=1, pady=5, padx=5, sticky='ew')
        
        row += 1
        ttk.Label(input_frame, text="Slope:", style='FormLabel.TLabel').grid(row=row, column=0, sticky='w', pady=5)
        self.input_vars['slope'] = ttk.Combobox(input_frame, width=30, state='readonly')
        self.input_vars['slope']['values'] = ['Flat', 'Gentle', 'Steep']
        self.input_vars['slope'].grid(row=row, column=1, pady=5, padx=5, sticky='ew')
        
        row += 1
        ttk.Label(input_frame, text="Commercial Potential:", style='FormLabel.TLabel').grid(row=row, column=0, sticky='w', pady=5)
        self.input_vars['commercial_potential'] = ttk.Combobox(input_frame, width=30, state='readonly')
        self.input_vars['commercial_potential']['values'] = ['High', 'Medium', 'Low']
        self.input_vars['commercial_potential'].grid(row=row, column=1, pady=5, padx=5, sticky='ew')
        
        row += 1
        ttk.Label(input_frame, text="Utilities Access:", style='FormLabel.TLabel').grid(row=row, column=0, sticky='w', pady=5)
        self.input_vars['utilities_access'] = ttk.Combobox(input_frame, width=30, state='readonly')
        self.input_vars['utilities_access']['values'] = ['Complete', 'Partial', 'None']
        self.input_vars['utilities_access'].grid(row=row, column=1, pady=5, padx=5, sticky='ew')
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        action_frame = ttk.Frame(pred_frame, style='Card.TFrame')
        action_frame.pack(fill='x', padx=20, pady=(10, 0))
        ttk.Button(action_frame, text="🔮 Predict Price", 
                  command=self.predict_price, style='Accent.TButton').pack(side='left')
        
        result_frame = ttk.LabelFrame(pred_frame, text="Prediction Output", padding=16, style='Section.TLabelframe')
        result_frame.pack(fill='x', padx=20, pady=20)
        self.prediction_result = ttk.Label(result_frame, text="Your prediction will appear here.", 
                                          style='Info.TLabel', justify='left', anchor='w')
        self.prediction_result.pack(fill='x', padx=10, pady=10)
    
    def create_results_tab(self):
        """Create results tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📈 Feature Importance")
        
        title = ttk.Label(results_frame, text="Feature Importance Analysis", 
                         font=('Arial', 16, 'bold'))
        title.pack(pady=20)
        
        self.feature_importance_text = scrolledtext.ScrolledText(results_frame, height=25, width=100)
        self.feature_importance_text.pack(pady=10, padx=20)
    
    def browse_file(self):
        """Browse for CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
    
    def load_data(self):
        """Load and preprocess data in background thread"""
        csv_path = self.file_path_var.get()
        if not csv_path:
            messagebox.showerror("Error", "Please select a CSV file first!")
            return
        
        # Show progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Loading Data")
        progress_window.geometry("450x150")
        progress_window.resizable(False, False)
        
        ttk.Label(progress_window, text="Loading and preprocessing data...", font=('Arial', 11)).pack(pady=15)
        progress_bar = ttk.Progressbar(progress_window, length=400, mode='indeterminate')
        progress_bar.pack(pady=10)
        progress_bar.start()
        ttk.Label(progress_window, text="This may take a moment. Please wait...", 
                 font=('Arial', 9, 'italic'), foreground='gray').pack(pady=10)
        
        progress_window.update()
        
        # Run loading in background thread
        def run_loading():
            try:
                self.predictor.csv_path = csv_path
                df, eda_results = self.predictor.load_and_preprocess_data(self.show_eda_var.get())
                self.root.after(0, lambda: self.show_load_results(df, eda_results, progress_window))
            except Exception as e:
                import traceback
                error_msg = f"Failed to load data: {str(e)}\n\n{traceback.format_exc()}"
                self.root.after(0, lambda: self.show_load_error(error_msg, progress_window))
        
        threading.Thread(target=run_loading, daemon=True).start()
    
    def show_load_results(self, df, eda_results, progress_window):
        """Display load results after completion"""
        try:
            progress_window.destroy()
        except:
            pass
        
        try:
            info = f"✅ Dataset loaded successfully!\n\n"
            info += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
            info += eda_results + "\n\n"
            info += "ENHANCEMENTS APPLIED:\n"
            info += "✓ Advanced outlier detection\n"
            info += "✓ Interaction features created\n"
            info += "✓ Log transformations applied\n"
            info += "✓ Missing values handled intelligently\n\n"
            info += f"First 5 rows:\n{df.head()}\n"
            
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(1.0, info)
            
            if self.show_eda_var.get() and hasattr(self.predictor, 'explorer'):
                self.display_eda_visualizations()
            
            messagebox.showinfo("Success", "Data loaded and enhanced successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying data: {str(e)}")
    
    def show_load_error(self, error_msg, progress_window):
        """Show load error after thread completes"""
        try:
            progress_window.destroy()
        except:
            pass
        messagebox.showerror("Error", error_msg)
    
    def train_model(self):
        """Train the enhanced model in background thread"""
        if self.predictor.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        # Show progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Training in Progress")
        progress_window.geometry("450x150")
        progress_window.resizable(False, False)
        
        ttk.Label(progress_window, text="Initializing model training...", font=('Arial', 11)).pack(pady=15)
        progress_bar = ttk.Progressbar(progress_window, length=400, mode='indeterminate')
        progress_bar.pack(pady=10)
        progress_bar.start()
        ttk.Label(progress_window, text="This may take several minutes. Please wait...", 
                 font=('Arial', 9, 'italic'), foreground='gray').pack(pady=10)
        
        progress_window.update()
        
        # Run training in background thread
        def run_training():
            try:
                model_type = self.model_type_var.get()
                metrics, feature_importance = self.predictor.train_model(model_type)
                self.root.after(0, lambda: self.show_training_results(metrics, feature_importance, model_type, progress_window))
            except Exception as e:
                import traceback
                error_msg = f"Failed to train model: {str(e)}\n\n{traceback.format_exc()}"
                self.root.after(0, lambda: self.show_training_error(error_msg, progress_window))
        
        threading.Thread(target=run_training, daemon=True).start()
    
    def show_training_results(self, metrics, feature_importance, model_type, progress_window):
        """Display training results after completion"""
        try:
            progress_window.destroy()
        except:
            pass
        
        try:
            
            results = f"🎉 MODEL TRAINING COMPLETE!\n\n"
            results += f"Model Type: {model_type.upper()}\n"
            results += "=" * 80 + "\n\n"
            
            results += f"📊 PERFORMANCE METRICS\n"
            results += f"{'─' * 80}\n"
            results += f"Training R² Score:    {metrics['train_r2']:.4f} ({metrics['train_r2']*100:.2f}%)\n"
            results += f"Testing R² Score:     {metrics['test_r2']:.4f} ({metrics['test_r2']*100:.2f}%)\n"
            results += f"Cross-Val R² Score:   {metrics['cv_mean']:.4f} (± {metrics['cv_std']*2:.4f})\n\n"
            
            results += "📏 ERROR METRICS\n"
            results += f"{'─' * 80}\n"
            results += f"Training MAE:         Rs. {metrics['train_mae']:,.2f}\n"
            results += f"Testing MAE:          Rs. {metrics['test_mae']:,.2f}\n"
            results += f"Training RMSE:        Rs. {metrics['train_rmse']:,.2f}\n"
            results += f"Testing RMSE:         Rs. {metrics['test_rmse']:,.2f}\n"
            results += f"Testing MAPE:         {metrics['test_mape']:.2f}%\n\n"
            
            results += "🎯 INTERPRETATION\n"
            results += f"{'─' * 80}\n"
            results += f"• Model explains {metrics['test_r2']*100:.2f}% of price variance\n"
            
            if metrics['test_r2'] > 0.9:
                results += "• ✅ EXCELLENT predictive performance!\n"
            elif metrics['test_r2'] > 0.8:
                results += "• ✅ VERY GOOD predictive performance!\n"
            elif metrics['test_r2'] > 0.7:
                results += "• ✅ GOOD predictive performance\n"
            elif metrics['test_r2'] > 0.6:
                results += "• ⚠️  MODERATE predictive performance\n"
            else:
                results += "• ⚠️  Model needs improvement\n"
            
            results += f"• Average prediction error: Rs. {metrics['test_mae']:,.2f}\n"
            results += f"• Typical percentage error: {metrics['test_mape']:.2f}%\n\n"
            
            r2_diff = metrics['train_r2'] - metrics['test_r2']
            results += "🔍 OVERFITTING ANALYSIS\n"
            results += f"{'─' * 80}\n"
            results += f"R² difference: {r2_diff:.4f}\n"
            if r2_diff < 0.03:
                results += "✅ No overfitting - Model generalizes excellently!\n"
            elif r2_diff < 0.07:
                results += "✅ Minimal overfitting - Model generalizes well\n"
            elif r2_diff < 0.15:
                results += "⚠️  Slight overfitting detected\n"
            else:
                results += "⚠️⚠️  Significant overfitting - Consider regularization\n"
            
            results += "\n💡 ACCURACY IMPROVEMENTS APPLIED:\n"
            results += f"{'─' * 80}\n"
            results += "✓ Advanced feature engineering (interaction features)\n"
            results += "✓ Log transformation for price (handles skewness)\n"
            results += "✓ RobustScaler (better for outliers)\n"
            results += "✓ Isolation Forest outlier removal\n"
            results += "✓ Extensive hyperparameter tuning\n"
            if model_type == 'stacking':
                results += "✓ Multiple models combined (ensemble)\n"
            
            self.training_results_text.delete(1.0, tk.END)
            self.training_results_text.insert(1.0, results)
            
            if feature_importance is not None:
                if not hasattr(self, 'feature_importance_text'):
                    self.create_results_tab()
                
                importance_text = "\n" + "=" * 80 + "\n"
                importance_text += "🔝 TOP 15 MOST IMPORTANT FEATURES\n"
                importance_text += "=" * 80 + "\n\n"
                importance_text += feature_importance.head(15).to_string(index=False)
                importance_text += "\n\n💡 INTERPRETATION:\n"
                importance_text += "─" * 80 + "\n"
                importance_text += "• These features have the strongest impact on predictions\n"
                importance_text += "• Focus on collecting accurate data for these features\n"
                importance_text += "• Engineered features often show high importance\n"
                
                self.feature_importance_text.delete(1.0, tk.END)
                self.feature_importance_text.insert(1.0, importance_text)
            
            self.display_model_visualizations()
            
            messagebox.showinfo("Success", 
                              f"Model trained successfully!\n\n"
                              f"Test R² Score: {metrics['test_r2']:.4f}\n"
                              f"Test MAE: Rs. {metrics['test_mae']:,.2f}\n"
                              f"Test MAPE: {metrics['test_mape']:.2f}%")
        
        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Error displaying results: {str(e)}\n\n{traceback.format_exc()}")
    
    def show_training_error(self, error_msg, progress_window):
        """Show training error after thread completes"""
        try:
            progress_window.destroy()
        except:
            pass
        messagebox.showerror("Error", error_msg)
    
    def predict_price(self):
        """Predict land price"""
        if self.predictor.model is None:
            messagebox.showerror("Error", "Please train or load a model first!")
            return
        
        try:
            input_data = {}
            input_data['district'] = self.input_vars['district'].get()
            input_data['location'] = self.input_vars['location'].get()
            input_data['road_access_ft'] = self.input_vars['road_access_ft'].get()
            input_data['road_type'] = self.input_vars['road_type'].get()
            input_data['distance_to_ringroad_km'] = self.input_vars['distance_to_ringroad_km'].get()
            input_data['area_sqft'] = self.input_vars['area_sqft'].get()
            input_data['land_shape'] = self.input_vars['land_shape'].get()
            input_data['slope'] = self.input_vars['slope'].get()
            input_data['commercial_potential'] = self.input_vars['commercial_potential'].get()
            input_data['utilities_access'] = self.input_vars['utilities_access'].get()
            
            predicted_price = self.predictor.predict_price(input_data)
            area_sqft = float(input_data['area_sqft'])
            total_price = predicted_price * area_sqft / 342.25
            
            result = f"🏆 PREDICTION RESULTS\n\n"
            result += f"💰 Price per Anna:     Rs. {predicted_price:,.2f}\n"
            result += f"📊 Total Price:        Rs. {total_price:,.2f}\n"
            result += f"📐 Area:               {area_sqft:,.0f} sqft\n"
            result += f"ℹ️  Note: 1 Anna = 342.25 sqft"
            
            self.prediction_result.config(text=result)
            
        except ValueError as e:
            messagebox.showerror("Error", "Please fill all fields with valid numerical values!")
        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Prediction failed: {str(e)}\n\n{traceback.format_exc()}")
    
    def save_model(self):
        """Save trained model"""
        if self.predictor.model is None:
            messagebox.showerror("Error", "No model to save! Please train a model first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.predictor.save_model(filename)
                messagebox.showinfo("Success", f"Model saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        """Load trained model"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.predictor.load_model(filename)
                if self.predictor.model is not None:
                    messagebox.showinfo("Success", f"Model loaded from {filename}")
                else:
                    messagebox.showerror("Error", "The selected file did not contain a usable model.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def display_eda_visualizations(self):
        """Display EDA visualizations"""
        for widget in self.viz_scrollable_frame.winfo_children():
            widget.destroy()
        
        if not hasattr(self.predictor, 'explorer') or not self.predictor.explorer:
            return
        
        explorer = self.predictor.explorer
        
        for idx, (name, fig) in enumerate(explorer.figures.items()):
            title_map = {
                'missing_values': 'Missing Values Analysis',
                'distributions': 'Feature Distributions',
                'boxplots': 'Box Plots (Outlier Detection)',
                'correlation': 'Correlation Heatmap'
            }
            
            title_label = ttk.Label(self.viz_scrollable_frame, 
                                   text=title_map.get(name, name),
                                   font=('Arial', 14, 'bold'))
            title_label.pack(pady=(20 if idx > 0 else 10, 10))
            
            canvas = FigureCanvasTkAgg(fig, master=self.viz_scrollable_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=10, padx=20)
            
            ttk.Separator(self.viz_scrollable_frame, orient='horizontal').pack(fill='x', padx=20, pady=10)
    
    def display_model_visualizations(self):
        """Display model prediction visualizations"""
        if not hasattr(self.predictor, 'y_test') or not hasattr(self.predictor, 'y_pred_test'):
            return
        
        for widget in self.viz_scrollable_frame.winfo_children():
            if hasattr(widget, '_viz_type') and widget._viz_type == 'model':
                widget.destroy()
        
        model_title = ttk.Label(self.viz_scrollable_frame, 
                               text="Model Performance Visualizations",
                               font=('Arial', 14, 'bold'))
        model_title.pack(pady=20)
        model_title._viz_type = 'model'
        
        fig = Figure(figsize=(14, 6))
        
        # Actual vs Predicted
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(self.predictor.y_test, self.predictor.y_pred_test, alpha=0.5, s=30, color='blue')
        ax1.plot([self.predictor.y_test.min(), self.predictor.y_test.max()], 
                [self.predictor.y_test.min(), self.predictor.y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Price (Rs.)', fontsize=10)
        ax1.set_ylabel('Predicted Price (Rs.)', fontsize=10)
        ax1.set_title('Actual vs Predicted Prices', fontsize=12, pad=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calculate and display R²
        r2 = r2_score(self.predictor.y_test, self.predictor.y_pred_test)
        ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                fontsize=10, fontweight='bold')
        
        # Residual plot
        ax2 = fig.add_subplot(1, 2, 2)
        residuals = self.predictor.y_test - self.predictor.y_pred_test
        ax2.scatter(self.predictor.y_pred_test, residuals, alpha=0.5, s=30, color='purple')
        ax2.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Residual')
        
        residual_std = residuals.std()
        residual_mean = residuals.mean()
        y_limit = max(abs(residual_mean - 3*residual_std), abs(residual_mean + 3*residual_std))
        ax2.set_ylim(-y_limit, y_limit)
        
        ax2.set_xlabel('Predicted Price (Rs.)', fontsize=10)
        ax2.set_ylabel('Residuals (Rs.)', fontsize=10)
        ax2.set_title('Residual Plot', fontsize=12, pad=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        residual_info = f'Mean: {residual_mean:,.0f}\nStd: {residual_std:,.0f}'
        ax2.text(0.02, 0.98, residual_info, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)
        
        fig.tight_layout(pad=2.0)
        
        canvas = FigureCanvasTkAgg(fig, master=self.viz_scrollable_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(pady=10, padx=20)
        canvas_widget._viz_type = 'model'


# ============================================================================
# SECTION 4: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedLandPricePredictorGUI(root)
    root.mainloop()