# Credit Card Fraud Detection

This project applies machine learning techniques to detect fraudulent credit card transactions using real-world data. It includes preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## 📂 Project Structure

```
creditcard-fraud-detection/
├── src/
│   ├── data_loader.py           # Load data
│   ├── exploratory.py           # EDA plots saved to disk
│   ├── model.py                 # Model training and evaluation
│   ├── utils.py                 # Statistical tests
│   ├── eda_plots/               # Auto-saved EDA images
│   ├── model_plots/             # Decision tree, etc.
├── data/                        # (Not included in repo, place your CSV here)
├── [creditcard_fraud_detection_notebook](https://colab.research.google.com/drive/1-8Qtbw47T2QZEJEixNPSzvrN33GgRs3W?usp=sharing)
├── main.py                      # Run all analysis
├── requirements.txt             # Dependencies
├── README.md                    # Project description
```

##  Dataset

- Based on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Features are anonymized via PCA (V1–V28) plus `Time`, `Amount`, and `Class`

> The dataset is not included due to size/privacy. Download from Kaggle and place in `src/data/`.

##  Models Used

- Decision Tree
- Random Forest (Top Features)
- Random Forest with SMOTE
- Undersampling
- Feature Engineering Pipeline

##  Features

- Custom plot saving to `eda_plots/` and `model_plots/`
- Levene’s and KS statistical tests for feature selection
- Modularized pipeline for flexibility
- Configurable via `main.py`

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

Edit `main.py` to:
- Run only selected models
- Toggle EDA or statistical tests

## Requirements

Python 3.8+ and the following libraries:
- pandas, numpy, matplotlib, seaborn
- scikit-learn, imbalanced-learn

## Author

**Ahmed Mansour** – [GitHub](https://github.com/AhmedMansour1070)
