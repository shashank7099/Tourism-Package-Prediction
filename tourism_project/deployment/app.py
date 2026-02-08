
import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# --- CONFIGURATION FROM ENVIRONMENT ---
# These can be set in Hugging Face Space Settings > Variables/Secrets
REPO_ID = os.getenv("REPO_ID", "shashankksaxena/tourism-model")
FILENAME = os.getenv("MODEL_FILENAME", "model.pkl")

@st.cache_resource
def load_model():
    # Loading the model from the specified Hugging Face Hub repository
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    return joblib.load(model_path)

model = load_model()

st.set_page_config(page_title="Visit with Us - Predictor", layout="wide")
st.title("Wellness Tourism Package Predictor")
st.write(f"Currently using model from: {REPO_ID}")

# --- GET INPUTS AND SAVE INTO A DATAFRAME ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration = st.number_input("Duration of Pitch (mins)", 1, 150, 15)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    persons = st.number_input("Number of Persons Visiting", 1, 10, 2)
    followups = st.number_input("Number of Follow-ups", 1, 10, 3)
    product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

with col2:
    star_rating = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    trips = st.number_input("Annual Trips", 1, 20, 2)
    passport = st.selectbox("Has Passport?", [0, 1])
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Owns a Car?", [0, 1])
    children = st.number_input("Children Visiting", 0, 5, 0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly Income", 1000, 200000, 25000)

# Mappings (Matches LabelEncoder from Training)
mapping = {
    "TypeofContact": {"Company Invited": 0, "Self Enquiry": 1},
    "Occupation": {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3},
    "Gender": {"Female": 0, "Male": 1},
    "ProductPitched": {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4},
    "MaritalStatus": {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3},
    "Designation": {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}
}

# --- PROCESS INPUTS INTO DATAFRAME ---
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': mapping["TypeofContact"][type_of_contact],
    'CityTier': city_tier,
    'DurationOfPitch': duration,
    'Occupation': mapping["Occupation"][occupation],
    'Gender': mapping["Gender"][gender],
    'NumberOfPersonVisiting': persons,
    'NumberOfFollowups': followups,
    'ProductPitched': mapping["ProductPitched"][product],
    'PreferredPropertyStar': star_rating,
    'MaritalStatus': mapping["MaritalStatus"][marital_status],
    'NumberOfTrips': trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': children,
    'Designation': mapping["Designation"][designation],
    'MonthlyIncome': income
}])

# --- PREDICTION ---
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction[0] == 1:
        st.success(f"[Y] Likely to Purchase! (Probability: {probability:.2%})")
    else:
        st.error(f"[X] Unlikely to Purchase. (Probability: {1-probability:.2%})")
