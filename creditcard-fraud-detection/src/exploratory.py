import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Set output directory for plots
PLOT_DIR = r"C:\Users\Ahmed\Downloads\creditcard-fraud-detection\src\eda_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def save_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, name))
    plt.close()

def class_distribution(df):
    sns.histplot(df['Class'])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    save_plot("class_distribution.png")

def plot_feature_distributions(df):
    for col in df.columns[:-1]:
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        save_plot(f"feature_distribution_{col}.png")

def plot_relative_std_diff(std_comparison_df):
    plt.figure(figsize=(12, 6))
    plt.bar(std_comparison_df['Feature'], std_comparison_df['Relative_Diff'])
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xticks(rotation=90)
    plt.title('Relative Difference in Std Between Fraud and Non-Fraud')
    plt.ylabel('Relative Difference')
    plt.xlabel('Feature')
    save_plot("relative_std_diff.png")

def kde_plot_for_two_df(df_fraud, df_non_fraud, features):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df_fraud[feature], label='Fraud', shade=True)
        sns.kdeplot(df_non_fraud[feature], label='Non-Fraud', shade=True)
        plt.title(f'Distribution of {feature} for Fraud vs Non-Fraud')
        plt.legend()
        save_plot(f"kde_{feature}.png")

def plot_short_gap_analysis(df):
    short_gap_analysis = df.groupby(['Amount_Quartile', 'Class'])['Short_Time_Gap'].mean().reset_index()
    sns.barplot(data=short_gap_analysis, x='Amount_Quartile', y='Short_Time_Gap', hue='Class')
    plt.title('Percentage of Short Gaps by Amount Quartile and Class')
    plt.ylabel('Percentage of Short Gaps')
    save_plot("short_gap_analysis.png")

def run_exploratory_analysis(df):
    print("Saving class distribution plot...")
    class_distribution(df)

    print("Saving feature distribution plots...")
    plot_feature_distributions(df)

    df_fraud = df[df['Class'] == 1]
    df_non_fraud = df[df['Class'] == 0]

    print("Saving KDE plots for selected features...")
    kde_plot_for_two_df(df_fraud, df_non_fraud, ['V14', 'V10', 'V12', 'V11', 'V17', 'V7'])

    print("Saving standard deviation comparison plot...")
    stds = pd.DataFrame({
        "Feature": df.columns[:-1],
        "Fraud_Std": df_fraud.std().values[:-1],
        "NonFraud_Std": df_non_fraud.std().values[:-1]
    })
    stds["Relative_Diff"] = (stds["Fraud_Std"] - stds["NonFraud_Std"]) / stds["NonFraud_Std"]
    plot_relative_std_diff(stds)

    if 'Short_Time_Gap' in df.columns and 'Amount_Quartile' in df.columns:
        print("Saving short gap analysis plot...")
        plot_short_gap_analysis(df)
