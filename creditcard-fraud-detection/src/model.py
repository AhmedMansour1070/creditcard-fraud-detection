import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler



import os

PLOT_DIR = r"C:\Users\Ahmed\Downloads\creditcard-fraud-detection\src\model_plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# ---- Feature Engineering Transformer ---- #
class CustomFeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Time_Diff'] = X['Time'].diff().fillna(0)
        X['Short_Time_Gap'] = (X['Time_Diff'] < 2).astype(int)
        X['High_Risk_Short_Time'] = (
            (X['Short_Time_Gap'] == 1) & 
            (X['Amount_Quartile'].isin(['Q1', 'Q4']))
        ).astype(int)
        return X[['Time_Diff', 'Short_Time_Gap', 'High_Risk_Short_Time']]

# ---- Modeling Functions ---- #


def train_decision_tree(df):
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    tree = DecisionTreeClassifier(random_state=42, max_depth=3)
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    print("=== Decision Tree ===")
    print(classification_report(y_test, y_pred))

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': tree.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(feature_importance)

    # Save tree plot
    plt.figure(figsize=(12, 8))
    plot_tree(tree, feature_names=X.columns, class_names=['Non-Fraud', 'Fraud'], filled=True, rounded=True)
    plot_path = os.path.join(PLOT_DIR, "decision_tree.png")
    if not os.path.exists(plot_path):
        plt.savefig(plot_path)
        print(f"Saved tree plot: {plot_path}")
    else:
        print("Decision tree plot already exists — skipping.")
    plt.close()

def train_random_forest_with_top_features(df):
    top_features = ['V14', 'V10', 'V12', 'V11', 'V17', 'V7', 'Amount']
    X = df[top_features]
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=42)
    
    model = RandomForestClassifier(random_state=42, bootstrap=False)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("=== Random Forest (Top Features) ===")
    print(classification_report(y_test, y_pred))

    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"Best threshold: {best_threshold:.2f}")

def train_with_smote(df):
    top_features = ['V14', 'V10', 'V12', 'V11', 'V17', 'V7', 'Amount']
    X = df[top_features]
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=42)
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_bal, y_bal)

    y_pred = model.predict(X_test)
    print("=== Random Forest with SMOTE ===")
    print(classification_report(y_test, y_pred))

def train_with_undersampling(df):
    top_features = ['V14', 'V10', 'V12', 'V11', 'V17', 'V7', 'Amount']
    X = df[top_features]
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=42)
    rus = RandomUnderSampler(random_state=42)
    X_bal, y_bal = rus.fit_resample(X_train, y_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_bal, y_bal)

    y_pred = model.predict(X_test)
    print("=== Random Forest with Undersampling ===")
    print(classification_report(y_test, y_pred))

def train_pipeline_with_engineering(df):
    df = df.copy()
    df['Amount_Quartile'] = pd.qcut(df['Amount'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    numeric_features = [col for col in X.columns if col.startswith('V') or col in ['Amount', 'Time']]
    categorical_features = ['Amount_Quartile']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features),
        ('eng', CustomFeatureEngineering(), ['Time', 'Amount_Quartile'])
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("=== Pipeline with Feature Engineering ===")
    print(classification_report(y_test, y_pred))

# ---- Entrypoint ---- #

def run_all_models(df):
    train_decision_tree(df)
    train_random_forest_with_top_features(df)
    train_with_smote(df)
    train_with_undersampling(df)
    train_pipeline_with_engineering(df)

    print("All models have been trained and evaluated.")