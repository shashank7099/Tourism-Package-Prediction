"""
Tourism Prediction Pipeline

This script automates the end-to-end workflow:
1. Sync raw data to Hugging Face Dataset Hub.
2. Load and clean the dataset.
3. Split the dataset, save artifacts to 'tourism_project/model_building', and upload to HF.
4. Train an XGBoost classifier.
5. Register the model to the Hugging Face Model Hub.
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
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "shashankksaxena"
login(token=HF_TOKEN)
api = HfApi()

# Define the artifact directory and ensure it exists
artifact_path = "tourism_project/model_building"
os.makedirs(artifact_path, exist_ok=True)

# --- STEP A: SYNC RAW DATA ---
if os.path.exists("tourism_project/data/tourism.csv"):
    api.upload_file(
        path_or_fileobj="tourism_project/data/tourism.csv",
        path_in_repo="tourism.csv",
        repo_id=f"{REPO_ID}/tourism-data",
        repo_type="dataset"
    )

# --- STEP B: LOAD & CLEAN ---
data_url = f"https://huggingface.co/datasets/{REPO_ID}/tourism-data/raw/main/tourism.csv"
df = pd.read_csv(data_url)

df.drop(columns=['Unnamed: 0', 'CustomerID'], errors='ignore', inplace=True)
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# Impute missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# --- STEP C: SPLIT & UPLOAD DATASETS ---
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['ProdTaken']
)

# Define full local paths for artifacts
train_file = os.path.join(artifact_path, "train.csv")
test_file = os.path.join(artifact_path, "test.csv")
model_file = os.path.join(artifact_path, "model.pkl")

# Save artifacts locally in the model_building folder
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

# Upload train/test splits to Hugging Face
api.upload_file(path_or_fileobj=train_file, path_in_repo="train.csv", 
                repo_id=f"{REPO_ID}/tourism-data", repo_type="dataset")
api.upload_file(path_or_fileobj=test_file, path_in_repo="test.csv", 
                repo_id=f"{REPO_ID}/tourism-data", repo_type="dataset")

print(f"Artifacts saved to {artifact_path} and uploaded to Hugging Face.")

# --- STEP D: ENCODING & TRAINING ---
le = LabelEncoder()
cat_cols = train_df.select_dtypes(include=['object']).columns

# Ensure encoding is consistent
for col in cat_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

X_train = train_df.drop('ProdTaken', axis=1)
y_train = train_df['ProdTaken']
X_test = test_df.drop('ProdTaken', axis=1)
y_test = test_df['ProdTaken']

model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric='logloss')
model.fit(X_train, y_train)

# --- STEP E: EVALUATE & REGISTER MODEL ---
y_pred = model.predict(X_test)
score = f1_score(y_test, y_pred)
print(f"Model trained. Test F1 Score: {score:.4f}")

# Save the model pickle into the model_building folder
joblib.dump(model, model_file)

# Upload model to HF Model Hub
api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo="model.pkl",
    repo_id=f"{REPO_ID}/tourism-model",
    repo_type="model"
)
print("Best model registered in the Hugging Face Model Hub.")
