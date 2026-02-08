"""
Tourism Prediction Pipeline

This script automates the end-to-end workflow for a tourism product prediction project:
1. Sync raw data to Hugging Face Dataset Hub.
2. Load and clean the dataset (handle missing values, fix typos).
3. Split the dataset into train and test sets and upload them to HF.
4. Encode categorical features using LabelEncoder.
5. Train an XGBoost classifier on the training data.
6. Evaluate the model on the test set using F1 score.
7. Save and register the trained model to the Hugging Face Model Hub.

The pipeline is designed for scalability and reproducibility.
"""

import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from huggingface_hub import HfApi, login

# 1. Setup & Authentication
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token from environment
REPO_ID = "shashankksaxena"
login(token=HF_TOKEN)  # Log in to HF using token
api = HfApi()

# --- STEP A: SYNC RAW DATA (SCALABILITY) ---
# Upload local CSV to Hugging Face Dataset Hub if it exists
if os.path.exists("tourism_project/data/tourism.csv"):
    api.upload_file(
        path_or_fileobj="tourism_project/data/tourism.csv",
        path_in_repo="tourism.csv",
        repo_id=f"{REPO_ID}/tourism-data",
        repo_type="dataset"
    )

# --- STEP B: LOAD & CLEAN ---
# Load dataset from Hugging Face Hub
data_url = f"https://huggingface.co/datasets/{REPO_ID}/tourism-data/raw/main/tourism.csv"
df = pd.read_csv(data_url)

# Drop unnecessary columns and fix known typos
df.drop(columns=['Unnamed: 0', 'CustomerID'], errors='ignore', inplace=True)
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# Impute missing values: mode for categorical, median for numeric
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# --- STEP C: SPLIT & UPLOAD DATASETS (RUBRIC REQUIREMENT) ---
# Stratified train-test split
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['ProdTaken']
)

# Save train/test locally for versioning
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

# Upload train/test splits to Hugging Face Dataset Hub
api.upload_file(path_or_fileobj="train.csv", path_in_repo="train.csv", 
                repo_id=f"{REPO_ID}/tourism-data", repo_type="dataset")
api.upload_file(path_or_fileobj="test.csv", path_in_repo="test.csv", 
                repo_id=f"{REPO_ID}/tourism-data", repo_type="dataset")

print("Train and Test datasets versioned and uploaded to Hugging Face.")

# --- STEP D: ENCODING & TRAINING ---
# Encode categorical columns
le = LabelEncoder()
cat_cols = train_df.select_dtypes(include=['object']).columns

for col in cat_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Prepare features and target
X_train = train_df.drop('ProdTaken', axis=1)
y_train = train_df['ProdTaken']
X_test = test_df.drop('ProdTaken', axis=1)
y_test = test_df['ProdTaken']

# Train XGBoost classifier
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric='logloss')
model.fit(X_train, y_train)

# --- STEP E: EVALUATE & REGISTER MODEL ---
# Evaluate on test set
y_pred = model.predict(X_test)
score = f1_score(y_test, y_pred)
print(f"Model trained. Test F1 Score: {score:.4f}")

# Save trained model locally and upload to HF Model Hub
joblib.dump(model, "model.pkl")
api.upload_file(
    path_or_fileobj="model.pkl",
    path_in_repo="model.pkl",
    repo_id=f"{REPO_ID}/tourism-model",
    repo_type="model"
)
print("Best model registered in the Hugging Face Model Hub.")
