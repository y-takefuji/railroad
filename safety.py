import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.cluster import FeatureAgglomeration
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load the data
file_path = 'Dispatchers_Background_Data.xls'
data = pd.read_excel(file_path)

# Check original shape
print(f"Original dataset shape: {data.shape}")

# Remove rows with NaN in target
data = data.dropna(subset=['FRA_report'])
print(f"Dataset shape after removing rows with NaN target: {data.shape}")

# Check target distribution
target_distribution = data['FRA_report'].value_counts()
print(f"Target distribution:\n{target_distribution}")

# Process time variables
for col in data.columns:
    # Check sample values to identify time columns
    sample_vals = data[col].dropna().astype(str).tolist()
    if sample_vals and ':' in str(sample_vals[0]):
        if len(str(sample_vals[0]).split(':')) == 3:  # Check if it looks like HH:MM:SS
            print(f"Converting time column: {col}")
            # Convert time format (HH:MM:SS) to seconds
            data[col] = data[col].astype(str).apply(
                lambda x: sum(int(t) * 60 ** (2 - i) for i, t in enumerate(x.split(':')))
                if ':' in x and len(x.split(':')) == 3 else np.nan
            )

# Clean the dataset by handling non-numeric columns
for col in data.columns:
    if col == 'FRA_report':  # Skip target
        continue
    
    # Check if column is numeric
    if not pd.api.types.is_numeric_dtype(data[col]):
        print(f"Converting column to numeric: {col}")
        # Try to convert to numeric, set errors to coerce to handle non-convertible values
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill missing values with median
for col in data.columns:
    if col != 'FRA_report' and data[col].isna().any():
        data[col] = data[col].fillna(data[col].median())

print(f"Dataset shape after preprocessing: {data.shape}")

# Separate features and target
X = data.drop('FRA_report', axis=1)
y = data['FRA_report']

# Transform target labels from [1, 2] to [0, 1] for models like XGBoost
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Original target values: {np.unique(y.values)}")
print(f"Encoded target values: {np.unique(y_encoded)}")

# Define functions for feature selection methods
def random_forest_importance(X, y, n_features=5):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    return feature_importance.nlargest(n_features).index.tolist()

def xgboost_importance(X, y, n_features=5):
    # Use encoded y for XGBoost which expects [0, 1] for binary classification
    model = XGBClassifier(random_state=42)
    model.fit(X, label_encoder.fit_transform(y))
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    return feature_importance.nlargest(n_features).index.tolist()

def logistic_regression_importance(X, y, n_features=5):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    feature_importance = pd.Series(np.abs(model.coef_[0]), index=X.columns)
    return feature_importance.nlargest(n_features).index.tolist()

def cox_model_importance(X, y, n_features=5):
    # Create a DataFrame with features and target
    df = X.copy()
    df['event'] = y  # Treating the target as an event
    df['time'] = 1  # Setting a constant time since we're not given survival time
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time', event_col='event')
    
    # Extract feature importance (absolute values of coefficients)
    feature_importance = pd.Series(np.abs(cph.params_), index=cph.params_.index)
    return feature_importance.nlargest(n_features).index.tolist()

def feature_agglomeration(X, y, n_features=5):
    # Apply feature agglomeration
    agglo = FeatureAgglomeration(n_clusters=min(30, X.shape[1]))
    agglo.fit(X)
    
    # Get cluster assignments
    labels = agglo.labels_
    
    # Calculate feature importance within each cluster
    feature_importances = {}
    for i, col in enumerate(X.columns):
        cluster = labels[i]
        if cluster not in feature_importances:
            feature_importances[cluster] = []
        importance = abs(np.corrcoef(X[col], y)[0, 1])
        feature_importances[cluster].append((col, importance))
    
    # Select top features across all clusters
    all_importances = []
    for cluster, features in feature_importances.items():
        all_importances.extend(features)
    
    # Sort by importance and select top features
    all_importances.sort(key=lambda x: x[1], reverse=True)
    return [feat[0] for feat in all_importances[:n_features]]

def hvgs_selection(X, y, n_features=5):
    # Calculate variance for each feature
    variances = X.var()
    
    # Select features with highest variance
    return variances.nlargest(n_features).index.tolist()

def spearman_correlation(X, y, n_features=5):
    correlations = []
    for col in X.columns:
        corr, _ = spearmanr(X[col], y)
        correlations.append((col, abs(corr)))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    return [feat[0] for feat in correlations[:n_features]]

# Define function for cross-validation
def cross_validate_rf(X, y, features):
    X_selected = X[features]
    model = RandomForestClassifier(random_state=42)
    scores = cross_val_score(model, X_selected, y, cv=10, scoring='accuracy')
    return scores.mean()

# Dictionary to store results
results = {
    'Method': [],
    'CV5 Accuracy': [],
    'CV5 Feature Rankings': [],
    'CV4 Feature Rankings': []
}

# Apply each feature selection method
methods = {
    'Random Forest': random_forest_importance,
    'XGBoost': xgboost_importance,
    'Logistic Regression': logistic_regression_importance,
    'Cox Model': cox_model_importance,
    'Feature Agglomeration': feature_agglomeration,
    'HVGS': hvgs_selection,
    'Spearman': spearman_correlation
}

for name, method in methods.items():
    print(f"\nProcessing {name}...")
    
    # Select top 5 features from full dataset
    top_5_features = method(X, y, 5)
    
    # Cross-validate with selected features
    if name in ['Random Forest', 'XGBoost', 'Logistic Regression']:
        if name == 'Random Forest':
            model = RandomForestClassifier(random_state=42)
            X_selected = X[top_5_features]
            cv5_accuracy = cross_val_score(model, X_selected, y, cv=10, scoring='accuracy').mean()
        elif name == 'XGBoost':
            model = XGBClassifier(random_state=42)
            X_selected = X[top_5_features]
            # Use encoded y for XGBoost
            cv5_accuracy = cross_val_score(model, X_selected, y_encoded, cv=10, scoring='accuracy').mean()
        else:
            model = LogisticRegression(random_state=42, max_iter=1000)
            X_selected = X[top_5_features]
            cv5_accuracy = cross_val_score(model, X_selected, y, cv=10, scoring='accuracy').mean()
    else:
        # For Cox, FA, HVGS and Spearman, use Random Forest
        X_selected = X[top_5_features]
        cv5_accuracy = cross_validate_rf(X, y, top_5_features)
    
    # Remove the highest ranked feature from the full dataset
    highest_feature = top_5_features[0]
    X_reduced = X.drop(highest_feature, axis=1)
    
    # Re-fit with the reduced dataset and re-select top 4 features
    top_4_features = method(X_reduced, y, 4)
    
    # Store results
    results['Method'].append(name)
    results['CV5 Accuracy'].append(cv5_accuracy)
    results['CV5 Feature Rankings'].append(', '.join(top_5_features))
    results['CV4 Feature Rankings'].append(', '.join(top_4_features))

# Create summary table
summary_table = pd.DataFrame(results)
summary_table['CV5 Accuracy'] = summary_table['CV5 Accuracy'].apply(lambda x: f"{x:.4f}")

# Display summary table
print("\nSummary Table:")
print(summary_table[['Method', 'CV5 Accuracy', 'CV5 Feature Rankings', 'CV4 Feature Rankings']])

# Save result table to CSV
summary_table.to_csv('result.csv', index=False)
print("\nResults saved to 'result.csv'")