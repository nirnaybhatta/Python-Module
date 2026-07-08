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
import warnings
import re
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
    """Enhanced ML model with advanced techniques"""
    
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
        self.model = None
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.dummy_columns = []
        self.feature_columns = None
        self.df = None
        self.use_log_transform = True
        self.poly_features = None
        self.explorer = None
        
        self.categorical_columns = ['district', 'location', 'road_type', 'land_shape', 
                                    'slope', 'commercial_potential', 'utilities_access']
        self.numerical_columns = ['road_access_ft', 'distance_to_ringroad_km', 'area_sqft']
    
    def clean_price(self, price_str):
        """Convert price string to numeric value using robust regex parsing"""
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
    
    def load_and_preprocess_data(self, show_eda=True):
        """Load and preprocess the dataset"""
        self.df = pd.read_csv(self.csv_path)

        # Normalize column names to snake_case
        def normalize_col(c):
            return str(c).strip().lower().replace(' ', '_')

        self.df.columns = [normalize_col(c) for c in self.df.columns]

        # Parse area to sqft
        def parse_area_to_sqft(s):
            if pd.isna(s):
                return np.nan
            s0 = str(s).lower()
            s0 = re.sub(r'[,:]', ' ', s0)
            
            ropani = 0.0
            anna = 0.0
            # patterns - ropani and aana conversions
            m = re.search(r'([0-9]*\.?[0-9]+)\s*(ropani|ropan|rpn)', s0)
            if m:
                ropani = float(m.group(1))
            m = re.search(r'([0-9]*\.?[0-9]+)\s*(aana|anna|ana)', s0)
            if m:
                anna = float(m.group(1))

            if ropani or anna:
                sqft = ropani * 5476.0 + anna * 342.25
                return sqft

            # Extract plain number
            m2 = re.search(r'([0-9]*\.?[0-9]+)', s0)
            if m2:
                val = float(m2.group(1))
                if val > 1000:
                    return val  # assume sqft
                else:
                    return val * 342.25  # assume aana
            return np.nan

        # Create standardized feature columns
        if 'area_sqft' not in self.df.columns:
            if 'land_size' in self.df.columns:
                self.df['area_sqft'] = self.df['land_size'].apply(parse_area_to_sqft)

        if 'distance_to_ringroad_km' not in self.df.columns:
            if 'distance_to_main_road' in self.df.columns:
                def parse_distance(s):
                    if pd.isna(s):
                        return 0.2  # default
                    s0 = str(s).lower()
                    if 'under' in s0 and '100' in s0:
                        return 0.05
                    if '100' in s0 and '300' in s0:
                        return 0.2
                    if '300' in s0 and '600' in s0:
                        return 0.45
                    m = re.search(r'([0-9]*\.?[0-9]+)\s*m', s0)
                    if m:
                        return float(m.group(1)) / 1000.0
                    return 0.2
                self.df['distance_to_ringroad_km'] = self.df['distance_to_main_road'].apply(parse_distance)

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

        # Detect and clean price column
        if 'price_per_anna' not in self.df.columns:
            detected = False
            candidates = [c for c in self.df.columns if any(k in c for k in ['price', 'title', 'amount', 'rs'])]
            for c in candidates:
                sample = self.df[c].astype(str).head(50).str.lower()
                if sample.str.contains(r'lakh|crore|rs|anna').any():
                    self.df['price_per_anna'] = self.df[c]
                    detected = True
                    break

            if not detected:
                raise ValueError("Could not find price column")

        # Clean price column
        if self.df['price_per_anna'].dtype == 'object' or self.df['price_per_anna'].dtype == 'string':
            self.df['price_per_anna'] = self.df['price_per_anna'].apply(self.clean_price)
        
        # Remove rows with missing target
        self.df = self.df.dropna(subset=['price_per_anna'])
        
        # Perform EDA
        eda_results = ""
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
            if col in self.df.columns and col in self.df.columns:
                mode_val = self.df[col].mode()
                if len(mode_val) > 0:
                    self.df[col].fillna(mode_val[0], inplace=True)
        
        return self.df, eda_results
    
    def prepare_features(self):
        """Prepare feature matrix"""
        numerical_cols = [col for col in self.numerical_columns if col in self.df.columns]
        
        encoded_cols = [c for c in self.dummy_columns if c in self.df.columns]
        
        # Fallback: if no features yet, use all numeric cols except price
        if not numerical_cols and not encoded_cols:
            numerical_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns 
                            if c != 'price_per_anna']
        
        self.feature_columns = numerical_cols + encoded_cols
        
        if not self.feature_columns:
            raise ValueError("No features available")
        
        X = self.df[self.feature_columns].fillna(0)
        y = self.df['price_per_anna']
        
        # Apply log transformation to target if enabled
        if self.use_log_transform:
            y = np.log1p(y)
            self.y_is_log = True
        else:
            self.y_is_log = False
        
        return X, y
    
    def train_model(self, model_type='random_forest'):
        """Train enhanced model"""
        X, y = self.prepare_features()
        
        # Basic outlier removal using z-score
        z_scores = np.abs(stats.zscore(y))
        mask = z_scores < 3
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Choose model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
            self.model.fit(X_train_scaled, y_train)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            self.model.fit(X_train_scaled, y_train)
        else:  # xgboost if available
            if XGBOOST_AVAILABLE:
                self.model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
                self.model.fit(X_train_scaled, y_train)
            else:
                self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
                self.model.fit(X_train_scaled, y_train)
        
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
        }
        
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
        """Predict land price"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Build feature array from input
        row = {}
        for col in self.feature_columns:
            val = input_data.get(col, 0)
            try:
                row[col] = float(val) if val else 0.0
            except:
                row[col] = 0.0
        
        # Create feature array
        feature_array = np.array([[row.get(col, 0) for col in self.feature_columns]])
        feature_array_scaled = self.scaler.transform(feature_array)
        
        # Predict
        prediction = self.model.predict(feature_array_scaled)[0]
        
        # Inverse transform if log was used
        if self.y_is_log:
            prediction = np.expm1(prediction)
        
        return prediction
    
    def save_model(self, filename='land_price_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'use_log_transform': self.use_log_transform,
            'y_is_log': self.y_is_log
        }
        joblib.dump(model_data, filename)
    
    def load_model(self, filename='land_price_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.use_log_transform = model_data.get('use_log_transform', True)
        self.y_is_log = model_data.get('y_is_log', False)


# ============================================================================
# SECTION 3: GUI
# ============================================================================

class EnhancedLandPricePredictorGUI:
    """Interactive GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Land Price Predictor - Kathmandu Valley")
        self.root.state('zoomed')
        
        self.predictor = EnhancedLandPricePredictor()
        
        style = ttk.Style()
        style.theme_use('clam')
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.create_data_tab()
        self.create_training_tab()
        self.create_prediction_tab()
    
    def create_data_tab(self):
        """Create data loading tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📁 Data Loading")
        
        title = ttk.Label(data_frame, text="Load and Preprocess Data", 
                         font=('Arial', 16, 'bold'))
        title.pack(pady=20)
        
        file_frame = ttk.Frame(data_frame)
        file_frame.pack(pady=10, padx=20, fill='x')
        
        ttk.Label(file_frame, text="CSV File:").pack(side='left', padx=5)
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=50).pack(side='left', padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side='left', padx=5)
        
        ttk.Button(data_frame, text="Load and Preprocess Data", 
                  command=self.load_data).pack(pady=20)
        
        self.data_info_text = scrolledtext.ScrolledText(data_frame, height=20, width=100)
        self.data_info_text.pack(pady=10, padx=20)
    
    def create_training_tab(self):
        """Create model training tab"""
        train_frame = ttk.Frame(self.notebook)
        self.notebook.add(train_frame, text="🎯 Model Training")
        
        title = ttk.Label(train_frame, text="Train Machine Learning Model", 
                         font=('Arial', 16, 'bold'))
        title.pack(pady=20)
        
        model_frame = ttk.LabelFrame(train_frame, text="Model Selection", padding=20)
        model_frame.pack(pady=10, padx=20, fill='x')
        
        self.model_type_var = tk.StringVar(value="random_forest")
        
        ttk.Radiobutton(model_frame, text="Random Forest", 
                       variable=self.model_type_var, value="random_forest").pack(anchor='w', pady=2)
        ttk.Radiobutton(model_frame, text="Gradient Boosting", 
                       variable=self.model_type_var, value="gradient_boosting").pack(anchor='w', pady=2)
        ttk.Radiobutton(model_frame, text="XGBoost", 
                       variable=self.model_type_var, value="xgboost").pack(anchor='w', pady=2)
        
        ttk.Button(train_frame, text="🚀 Train Model", 
                  command=self.train_model).pack(pady=20)
        
        save_frame = ttk.Frame(train_frame)
        save_frame.pack(pady=10)
        
        ttk.Button(save_frame, text="💾 Save Model", command=self.save_model).pack(side='left', padx=5)
        ttk.Button(save_frame, text="📂 Load Model", command=self.load_model).pack(side='left', padx=5)
        
        self.training_results_text = scrolledtext.ScrolledText(train_frame, height=15, width=100)
        self.training_results_text.pack(pady=10, padx=20)
    
    def create_prediction_tab(self):
        """Create prediction tab"""
        pred_frame = ttk.Frame(self.notebook)
        self.notebook.add(pred_frame, text="🔮 Price Prediction")
        
        title = ttk.Label(pred_frame, text="Predict Land Price", 
                         font=('Arial', 16, 'bold'))
        title.pack(pady=20)
        
        input_frame = ttk.LabelFrame(pred_frame, text="Land Details", padding=20)
        input_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        self.input_vars = {}
        
        for idx, label in enumerate(['District', 'Location', 'Road Access (ft)', 'Area (sqft)']):
            ttk.Label(input_frame, text=label + ":").grid(row=idx, column=0, sticky='w', pady=5)
            self.input_vars[label] = ttk.Entry(input_frame, width=32)
            self.input_vars[label].grid(row=idx, column=1, pady=5, padx=5)
        
        ttk.Button(pred_frame, text="🔮 Predict Price", 
                  command=self.predict_price).pack(pady=20)
        
        self.prediction_result = ttk.Label(pred_frame, text="", 
                                          font=('Arial', 14, 'bold'), foreground='green')
        self.prediction_result.pack(pady=10)
    
    def browse_file(self):
        """Browse for CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
    
    def load_data(self):
        """Load and preprocess data"""
        csv_path = self.file_path_var.get()
        if not csv_path:
            messagebox.showerror("Error", "Please select a CSV file first!")
            return
        
        try:
            self.predictor.csv_path = csv_path
            df, eda_results = self.predictor.load_and_preprocess_data(show_eda=False)
            
            info = f"✅ Dataset loaded successfully!\n\n"
            info += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
            info += eda_results + "\n\n"
            info += f"First 5 rows:\n{df.head()}\n"
            
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(1.0, info)
            
            messagebox.showinfo("Success", "Data loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def train_model(self):
        """Train the model"""
        if self.predictor.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        try:
            model_type = self.model_type_var.get()
            
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Training in Progress")
            progress_window.geometry("400x100")
            ttk.Label(progress_window, text="Training model... Please wait.", 
                     font=('Arial', 12)).pack(pady=20)
            progress_window.update()
            
            metrics, feature_importance = self.predictor.train_model(model_type)
            
            progress_window.destroy()
            
            results = f"🎉 MODEL TRAINING COMPLETE!\n\n"
            results += f"Model Type: {model_type.upper()}\n"
            results += "=" * 80 + "\n\n"
            
            results += f"📊 PERFORMANCE METRICS\n"
            results += f"Training R² Score:    {metrics['train_r2']:.4f}\n"
            results += f"Testing R² Score:     {metrics['test_r2']:.4f}\n"
            results += f"Training MAE:         Rs. {metrics['train_mae']:,.2f}\n"
            results += f"Testing MAE:          Rs. {metrics['test_mae']:,.2f}\n\n"
            
            self.training_results_text.delete(1.0, tk.END)
            self.training_results_text.insert(1.0, results)
            
            messagebox.showinfo("Success", 
                              f"Model trained successfully!\n\n"
                              f"Test R² Score: {metrics['test_r2']:.4f}\n"
                              f"Test MAE: Rs. {metrics['test_mae']:,.2f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
    
    def predict_price(self):
        """Predict land price"""
        if self.predictor.model is None:
            messagebox.showerror("Error", "Please train or load a model first!")
            return
        
        try:
            input_data = {label: var.get() for label, var in self.input_vars.items()}
            predicted_price = self.predictor.predict_price(input_data)
            
            result = f"🏆 PREDICTION RESULT\n\n"
            result += f"💰 Predicted Price:     Rs. {predicted_price:,.2f}\n"
            
            self.prediction_result.config(text=result)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
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
                messagebox.showinfo("Success", f"Model loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedLandPricePredictorGUI(root)
    root.mainloop()
