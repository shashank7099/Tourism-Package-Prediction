"""
Tourism Prediction Pipeline with SMOTE

This script automates:
1. Data Syncing (HF Dataset Hub)
2. Cleaning & Encoding
3. Stratified Splitting & SMOTE Balancing
4. Training with XGBoost
5. Evaluation & Registration (HF Model Hub)
"""

import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from huggingface_hub import HfApi, login

# =========================
# INITIALIZATION
# =========================
print("\n================ PIPELINE STARTED ================")
print("[INIT] Initializing environment and Hugging Face connection...")

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "shashankksaxena"

login(token=HF_TOKEN)
api = HfApi()

artifact_path = "tourism_project/model_building"
os.makedirs(artifact_path, exist_ok=True)
print(f"[INIT] Artifact directory ready at: {artifact_path}")

# =========================
# STEP A: SYNC RAW DATA
# =========================
print("\n================ STEP A: DATA SYNC =================")
local_raw_path = "tourism_project/data/tourism.csv"

if os.path.exists(local_raw_path):
    print("[STEP A] Raw data found locally. Uploading to Hugging Face...")
    api.upload_file(
        path_or_fileobj=local_raw_path,
        path_in_repo="tourism.csv",
        repo_id=f"{REPO_ID}/tourism-data",
        repo_type="dataset"
    )
    print("[STEP A] Raw data successfully synced.")
else:
    print("[STEP A] WARNING: Raw data not found locally. Skipping upload.")

# =========================
# STEP B: LOAD & CLEAN DATA
# =========================
print("\n================ STEP B: LOAD & CLEAN =================")
data_url = f"https://huggingface.co/datasets/{REPO_ID}/tourism-data/raw/main/tourism.csv"
print(f"[STEP B] Loading data from: {data_url}")

df = pd.read_csv(data_url)
print(f"[STEP B] Data loaded successfully with shape: {df.shape}")

print("[STEP B] Dropping identifier columns (if present)...")
df.drop(columns=['Unnamed: 0', 'CustomerID'], errors='ignore', inplace=True)

print("[STEP B] Fixing known categorical typos...")
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

print("[STEP B] Handling missing values...")
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

print("[STEP B] Missing value handling complete.")

# =========================
# STEP C: ENCODING
# =========================
print("\n================ STEP C: ENCODING =================")
print("[STEP C] Encoding categorical features for SMOTE compatibility...")

le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print(f"[STEP C] Encoded columns: {list(cat_cols)}")
print(f"[STEP C] Dataset shape after encoding: {df.shape}")

# =========================
# STEP D: SPLIT & SMOTE
# =========================
print("\n================ STEP D: SPLIT & SMOTE =================")
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']

print("[STEP D] Performing stratified 80/20 train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[STEP D] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"[STEP D] Before SMOTE - Positive class count: {sum(y_train == 1)}")

print("[STEP D] Applying SMOTE to training data...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"[STEP D] After SMOTE - Positive class count: {sum(y_train_res == 1)}")
print(f"[STEP D] Balanced train shape: {X_train_res.shape}")

# Save artifacts
print("[STEP D] Saving train/test artifacts...")
train_file = os.path.join(artifact_path, "train.csv")
test_file = os.path.join(artifact_path, "test.csv")

pd.concat([X_train_res, y_train_res], axis=1).to_csv(train_file, index=False)
pd.concat([X_test, y_test], axis=1).to_csv(test_file, index=False)

print("[STEP D] Uploading processed datasets to Hugging Face...")
# FIXED: Using explicit keyword arguments
api.upload_file(
    path_or_fileobj=train_file, 
    path_in_repo="train.csv", 
    repo_id=f"{REPO_ID}/tourism-data", 
    repo_type="dataset"
)
api.upload_file(
    path_or_fileobj=test_file, 
    path_in_repo="test.csv", 
    repo_id=f"{REPO_ID}/tourism-data", 
    repo_type="dataset"
)
print("[STEP D] Dataset upload complete.")

# =========================
# STEP E: MODEL TRAINING
# =========================
print("\n================ STEP E: TRAINING =================")
print("[STEP E] Initializing XGBoost model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='logloss'
)

print("[STEP E] Training model on SMOTE-balanced data...")
model.fit(X_train_res, y_train_res)
print("[STEP E] Model training complete.")

# =========================
# STEP F: EVALUATION & REGISTRATION
# =========================
print("\n================ STEP F: EVALUATION =================")
print("[STEP F] Generating predictions on original test distribution...")

y_pred = model.predict(X_test)
score_f1 = f1_score(y_test, y_pred)
score_acc = accuracy_score(y_test, y_pred)

print("[STEP F] Evaluation Metrics:")
print(f" - Accuracy: {score_acc:.4f}")
print(f" - F1 Score: {score_f1:.4f}")

print("[STEP F] Saving trained model artifact...")
model_file = os.path.join(artifact_path, "model.pkl")
joblib.dump(model, model_file)

print("[STEP F] Uploading model to Hugging Face Model Hub...")
# FIXED: Using explicit keyword arguments
api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo="model.pkl",
    repo_id=f"{REPO_ID}/tourism-model",
    repo_type="model"
)

print("\n================ PIPELINE COMPLETED SUCCESSFULLY ================")
print("[SUCCESS] Balanced model trained, evaluated, and registered ")
