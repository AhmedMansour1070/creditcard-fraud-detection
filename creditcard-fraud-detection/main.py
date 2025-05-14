from src.data_loader import load_data
from src.exploratory import run_exploratory_analysis
from src.model import (
    train_decision_tree,
    train_random_forest_with_top_features,
    train_with_smote,
    train_with_undersampling,
    train_pipeline_with_engineering
)
from src.utils import perform_levene_test, perform_ks_tests
import pandas as pd

def main():
    df = load_data()
    df['Amount_Quartile'] = pd.qcut(df['Amount'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    df_fraud = df[df['Class'] == 1]
    df_non_fraud = df[df['Class'] == 0]

    # Statistical tests
    perform_levene_test(df_fraud, df_non_fraud, ['V17', 'V10', 'V12'])
    ks_result_df = perform_ks_tests(
        df_fraud[df_fraud.columns[:-1]],
        df_non_fraud[df_non_fraud.columns[:-1]]
    )
    print(ks_result_df.head())

    # EDA
    # run_exploratory_analysis(df)

    # Run selected model
    # train_decision_tree(df)
    # train_random_forest_with_top_features(df)
    # train_with_smote(df)
    # train_with_undersampling(df)
    # train_pipeline_with_engineering(df)

if __name__ == "__main__":
    main()
