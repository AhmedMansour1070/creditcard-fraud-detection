import numpy as np
import pandas as pd
from scipy.stats import levene, ks_2samp

def perform_levene_test(df1, df2, features):
    """
    Performs Levene's test for equal variances on a list of features.
    """
    for feature in features:
        stat, p_value = levene(df1[feature], df2[feature])
        print(f"Levene's Test for {feature}: p-value = {p_value:.4f}")

        if p_value < 0.05:
            print(f"  → Variance is significantly different for {feature}.")
        else:
            print(f"  → No significant difference in variance for {feature}.")
        print()

def ks_score(stat, p_value):
    """
    A custom scoring function that ranks feature significance.
    """
    return stat * (1 / (p_value + 1e-5))  # Avoid division by zero

def perform_ks_tests(df_fraud, df_non_fraud):
    """
    Performs KS test on all features and returns a sorted DataFrame of scores.
    """
    scores = []
    features = df_fraud.columns

    for feature in features:
        stat, p_value = ks_2samp(df_fraud[feature], df_non_fraud[feature])
        score_val = ks_score(stat, p_value)
        print(f"KS Test for {feature}: stat = {stat:.4f}, p = {p_value:.2e}")
        scores.append((feature, score_val))

    score_df = pd.DataFrame(scores, columns=['Feature', 'Score']).sort_values(by='Score', ascending=False)
    return score_df
