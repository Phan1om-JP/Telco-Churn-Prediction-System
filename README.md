# Telco Customer Churn Prediction System

A production-ready machine learning system that predicts customer churn for telecommunications companies, deployed as a REST API with Docker containerization.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Model Performance](#model-performance)
- [Key Business Insights](#key-business-insights)
- [Technical Architecture](#technical-architecture)
- [Installation and Usage](#installation-and-usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results and Analysis](#results-and-analysis)
- [Future Enhancements](#future-enhancements)

---

## Problem Statement

Customer churn represents a significant cost for telecommunications companies, with acquisition costs for new customers being 5-25 times higher than retention costs. This project addresses the challenge of identifying customers at risk of churning before they leave, enabling targeted retention strategies.

**Objectives:**
- Predict which customers are likely to churn with high recall
- Provide interpretable risk scores for business decision-making
- Deploy a scalable API for integration with existing systems

---

## Solution Overview

This project delivers an end-to-end machine learning pipeline that processes customer data and returns churn predictions via a REST API. The system uses logistic regression with an optimized decision threshold to maximize recall while maintaining acceptable precision.

**System Capabilities:**
- Processes 28 customer features including demographics, service usage, and billing information
- Returns churn probability, risk classification, and actionable recommendations
- Handles data validation and preprocessing automatically
- Scales numerical features using the same transformation applied during training

---

## Model Performance

The final model was selected after comparing three algorithms: Logistic Regression, Random Forest, and XGBoost. Logistic Regression was chosen for its balance of performance, interpretability, and inference speed.

**Performance Metrics (Test Set):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Recall | 86.6% | Identifies 87 out of 100 customers who will churn |
| Precision | 46.6% | 47% of churn predictions are correct |
| F1-Score | 60.6% | Harmonic mean of precision and recall |
| ROC-AUC | 84.2% | Strong discriminative ability |
| Accuracy | 70.0% | Overall correctness (less relevant due to class imbalance) |

**Decision Threshold:** 0.4 (optimized for high recall rather than default 0.5)

**Business Impact Estimate:**
Assuming 10,000 customers with 26.5% churn rate:
- Without model: Random targeting captures 265 churners
- With model: Targeting top 1,000 predicted risks captures 865 churners
- Improvement: 227% increase in retention campaign efficiency

If retention cost is $50 per customer and average customer lifetime value is $500:
- Cost of targeting 1,000 customers: $50,000
- Value of retaining 865 customers: $432,500
- Net benefit: $382,500 per campaign

---

## Key Business Insights

Analysis of 7,043 customer records revealed several critical patterns:

**High-Risk Factors:**
1. Contract Type: Month-to-month contracts show 42.7% churn compared to 2.8% for two-year contracts
2. Internet Service: Fiber optic customers churn at 41.9%, significantly higher than DSL (19.0%) or no internet (7.4%)
3. Payment Method: Electronic check users exhibit 45.3% churn rate
4. Tenure: New customers (under 12 months) demonstrate 50%+ churn rates
5. Senior Citizens: 41.7% churn rate versus 23.6% for non-seniors
6. Service Bundling: Customers without add-on services (OnlineSecurity, TechSupport) churn at double the rate

**Protective Factors:**
1. Long-term contracts reduce churn by 93%
2. Having dependents correlates with 51% lower churn
3. Add-on service subscriptions reduce churn by approximately 50%

**Actionable Recommendations:**
1. Incentivize annual contract conversions for month-to-month customers
2. Investigate fiber optic service quality issues
3. Implement early-intervention programs for customers in first 12 months
4. Bundle security and support services in standard packages
5. Transition electronic check users to automatic payment methods

---

## Technical Architecture

**Machine Learning Stack:**
- Scikit-learn 1.3.2 for model training and preprocessing
- Pandas 2.1.3 and NumPy 1.26.2 for data manipulation
- Joblib 1.3.2 for model serialization

**API Framework:**
- FastAPI 0.104.1 for REST API with automatic documentation
- Pydantic 2.5.0 for request/response validation
- Uvicorn 0.24.0 as ASGI server

**Deployment:**
- Docker containerization for consistent deployment
- Python 3.11-slim base image
- Environment-based configuration

---

## Installation and Usage

### Prerequisites

- Python 3.11 or higher
- Docker (optional, for containerized deployment)

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Telco-Churn-Prediction-System.git
cd telco-churn-prediction-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the API server:
```bash
uvicorn app.main:app --reload
```

4. Access the interactive documentation:
```
http://localhost:8000/docs
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t telco-churn-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 telco-churn-api
```

3. Verify the deployment:
```bash
curl http://localhost:8000/health
```

---

## API Documentation

### Endpoints

#### Health Check
```
GET /health
```
Returns API status and model availability.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "api_version": "1.0.0"
}
```

#### Predict Churn
```
POST /predict
```
Accepts customer data and returns churn prediction.

**Request Body Example:**
```json
{
  "SeniorCitizen": 0,
  "Partner": 1,
  "Dependents": 0,
  "tenure": 12,
  "PhoneService": 1,
  "MultipleLines": 1,
  "OnlineSecurity": 0,
  "OnlineBackup": 0,
  "DeviceProtection": 0,
  "TechSupport": 0,
  "StreamingTV": 1,
  "StreamingMovies": 1,
  "PaperlessBilling": 1,
  "MonthlyCharges": 85.50,
  "TotalCharges": 1026.00,
  "InternetService_DSL": 0,
  "InternetService_Fiber optic": 1,
  "InternetService_No": 0,
  "Contract_Month-to-month": 1,
  "Contract_One_year": 0,
  "Contract_Two_year": 0,
  "PaymentMethod_Bank transfer (automatic)": 0,
  "PaymentMethod_Credit card (automatic)": 0,
  "PaymentMethod_Electronic_check": 1,
  "PaymentMethod_Mailed_check": 0,
  "gender_Female": 0,
  "gender_Male": 1
}
```

**Response:**
```json
{
  "customer_id": "a7b3c9d2",
  "churn_probability": 0.7234,
  "churn_prediction": "Will Churn",
  "risk_level": "High Risk",
  "confidence": 0.7234,
  "recommendation": "Immediate retention action required. Offer premium support or contract incentives."
}
```

#### Model Information
```
GET /model-info
```
Returns model metadata and performance metrics.

### Testing the API

Using curl:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @data/sample_request.json
```

Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "SeniorCitizen": 0,
        "Partner": 1,
        "tenure": 12,
        # ... remaining fields
    }
)

print(response.json())
```

---

## Project Structure
```
telco-churn-prediction/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and route definitions
│   ├── schemas.py           # Pydantic models for request/response validation
│   ├── model_loader.py      # Model loading and prediction logic
│   └── config.py            # Configuration management
│
├── models/
│   ├── churn_model.pkl      # Trained logistic regression model
│   ├── scaler.pkl           # StandardScaler for feature normalization
│   └── feature_names.pkl    # Feature column names and order
│
├── notebooks/
│   ├── 01_EDA.ipynb         # Exploratory data analysis
│   ├── 02_Modeling.ipynb    # Model training and evaluation
│   └── 03_Analysis.ipynb    # Feature importance and insights
│
├── data/
│   ├── telco_churn.csv      # Original dataset
│   └── sample_request.json  # Example API request payload
│
├── tests/
│   └── test_api.py          # API endpoint tests
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Methodology

### Data Preprocessing

1. **Data Cleaning:**
   - Converted TotalCharges from object to numeric type
   - Filled missing values (11 records) with 0 for new customers
   - Removed customerID as non-predictive identifier

2. **Feature Engineering:**
   - Converted binary Yes/No features to 0/1 encoding
   - Applied one-hot encoding to categorical features (InternetService, Contract, PaymentMethod, gender)
   - Standardized numerical features (tenure, MonthlyCharges, TotalCharges) using StandardScaler

3. **Class Imbalance Handling:**
   - Applied class weighting (balanced mode) to account for 26.5% churn rate
   - Adjusted decision threshold from 0.5 to 0.4 to prioritize recall

### Model Selection Process

Three algorithms were evaluated using 5-fold cross-validation:

1. **Logistic Regression**
   - Linear model with L2 regularization
   - Interpretable coefficients for feature importance
   - Fast inference time

2. **Random Forest**
   - Ensemble of 100 decision trees
   - Captures non-linear relationships
   - Provides feature importance rankings

3. **XGBoost**
   - Gradient boosting with scale_pos_weight adjustment
   - Strong performance on imbalanced datasets
   - Built-in regularization

**Selection Criteria:**
Primary metric was recall (minimizing false negatives), followed by F1-score and ROC-AUC. Logistic Regression was selected despite similar performance across models due to superior interpretability and deployment simplicity.

### Evaluation Strategy

- 80/20 train-test split with stratification to maintain class distribution
- Evaluation focused on test set performance to ensure generalization
- Threshold optimization performed on validation set before final testing

---

## Results and Analysis

### Model Comparison

| Model | Recall | Precision | F1-Score | ROC-AUC | Training Time |
|-------|--------|-----------|----------|---------|---------------|
| Logistic Regression | 0.786 | 0.506 | 0.616 | 0.842 | 0.05s |
| XGBoost | 0.770 | 0.521 | 0.621 | 0.840 | 0.20s |
| Random Forest | 0.719 | 0.550 | 0.623 | 0.839 | 0.45s |

### Feature Importance

Top predictors of churn based on logistic regression coefficients:

1. Contract_Month-to-month (positive correlation)
2. InternetService_Fiber optic (positive correlation)
3. tenure (negative correlation - longer tenure reduces churn)
4. TotalCharges (negative correlation)
5. Contract_Two year (negative correlation)

### Threshold Analysis

Performance at different decision thresholds:

| Threshold | Recall | Precision | F1-Score | Use Case |
|-----------|--------|-----------|----------|----------|
| 0.3 | 92.8% | 43.2% | 58.9% | Maximum churner capture, high false positives |
| 0.4 | 86.6% | 46.6% | 60.6% | Balanced approach (selected) |
| 0.5 | 78.6% | 50.6% | 61.6% | Default threshold |
| 0.6 | 70.3% | 53.9% | 61.0% | Higher precision, miss more churners |

The 0.4 threshold was selected as it captures 87% of churners while maintaining reasonable precision for cost-effective targeting.

### Confusion Matrix Interpretation

At threshold 0.4 on test set (1,409 samples):

|  | Predicted Stay | Predicted Churn |
|---|----------------|-----------------|
| **Actually Stayed** | 856 | 178 |
| **Actually Churned** | 50 | 325 |

- True Negatives: 856 (correctly identified loyal customers)
- False Positives: 178 (unnecessary retention efforts)
- False Negatives: 50 (missed churners - highest business cost)
- True Positives: 325 (correctly identified churners)

---

## Future Enhancements

**Model Improvements:**
1. Incorporate time-series features (usage patterns over time)
2. Add customer lifetime value (CLV) estimation
3. Implement model retraining pipeline with new data
4. Experiment with ensemble methods combining multiple models

**Feature Additions:**
1. Customer service interaction data (call frequency, complaint history)
2. Competitor pricing information
3. Network quality metrics by region
4. Marketing campaign response data

**System Enhancements:**
1. Implement A/B testing framework for retention strategies
2. Add batch prediction endpoint for processing multiple customers
3. Integrate with CRM systems (Salesforce, HubSpot)
4. Build monitoring dashboard for model performance tracking
5. Add explainability features (SHAP values for individual predictions)

**Deployment:**
1. Set up CI/CD pipeline for automated testing and deployment
2. Implement model versioning and rollback capabilities
3. Add logging and monitoring (Prometheus, Grafana)
4. Scale horizontally with Kubernetes for high-traffic scenarios

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaboration opportunities, please open an issue in this repository.

## Acknowledgments

Dataset: Telco Customer Churn dataset from Kaggle