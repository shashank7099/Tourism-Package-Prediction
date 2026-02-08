# Wellness Tourism Package Predictor

## Project Overview

This project uses machine learning to predict whether a customer will purchase a newly introduced **Wellness Tourism Package**. By analyzing demographic data and engagement metrics, the model helps travel agencies target the right audience and optimize sales efforts.

The project is designed with a full MLOps lifecycle approach, including data versioning, pipeline orchestration, and a containerized Streamlit interface.

---

## Project Structure

The repository is organized into specific modules for data handling, model development, and cloud deployment:

```text
.
├── Learner_Template_Notebook_AML_and_MLOps_Project.ipynb  # Exploratory analysis & experiments
├── README.md                                              # Project documentation
├── tourism_project/                                       # Main project directory
│   ├── run_pipeline.py                                    # Orchestrator for the ML pipeline
│   ├── data/                                              # Data storage
│   │   ├── tourism.csv                                    # Raw data
│   │   ├── train.csv                                      # Training set
│   │   └── test.csv                                       # Testing set
│   ├── model_building/                                    # Model training artifacts
│   │   └── xgb_tourism_model.pkl                          # Serialized XGBoost model
│   └── deployment/                                        # Production-ready files
[cite_start]│       ├── app.py                                         # Streamlit application [cite: 1]
[cite_start]│       ├── Dockerfile                                     # Docker container config [cite: 1]
[cite_start]│       └── requirements.txt                               # Production dependencies [cite: 4]

```

---

## Key Features

* **Automated Pipeline**: The `run_pipeline.py` script manages the flow from raw data to model generation.
* 
**Predictive Dashboard**: A Streamlit-based UI where users can input features like Age, Monthly Income, and Pitch Satisfaction to get real-time results.


* 
**Containerized Architecture**: A custom `Dockerfile` ensures the app runs consistently across different environments.


* 
**Dynamic Model Integration**: The app is configured to pull the latest model artifacts from the Hugging Face Hub during initialization.



---

## Technical Stack

* 
**Core Logic**: Python 3.9 


* 
**ML Framework**: XGBoost 


* 
**Data Processing**: Pandas 


* 
**Interface**: Streamlit 


* 
**Deployment**: Docker (Linux-based python:3.9-slim) 


* 
**Storage**: Hugging Face Model Hub 



---

## How to Run Locally

### 1. Process Data & Train Model

To run the end-to-end pipeline:

```bash
cd tourism_project
python run_pipeline.py

```

### 2. Launch the Streamlit App

To test the production interface:

```bash
cd tourism_project/deployment
pip install -r requirements.txt
streamlit run app.py

```

---

## Docker Deployment

To build and run the containerized application:

```bash
cd tourism_project/deployment
docker build -t wellness-tourism-predictor .
docker run -p 7860:7860 wellness-tourism-predictor

```

---

## Author

**Shashank Saxena**

* Hugging Face: [@shashankksaxena](https://www.google.com/search?q=https://huggingface.co/shashankksaxena)
* GitHub: [shashankksaxena](https://www.google.com/search?q=https://github.com/shashankksaxena)