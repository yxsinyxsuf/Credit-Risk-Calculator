import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open('rf_model_filtered.pkl', 'rb') as f:
    model = pickle.load(f)

# Define function to calculate PD and Expected Loss
def calculate_pd_and_loss(model, borrower_details, loan_amount, recovery_rate=0.1):
    borrower_array = [borrower_details]  # Convert input to 2D array
    pd_probability = model.predict_proba(borrower_array)[0][1]  # Get probability of default
    expected_loss = pd_probability * (1 - recovery_rate) * loan_amount
    return {"Probability of Default": pd_probability, "Expected Loss": expected_loss}

# Streamlit app layout
st.title("Loan Default Risk Calculator")
st.write("Use this app to explore loan data, visualize feature importance, and predict default risk.")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
st.markdown(
    "This file is used to demonstrate the app's ability to analyze loan data, "
    "generate insights like feature importance, and make predictions on default risk. "
    "You can use the included `Loan_Data.csv` as a sample, or proceed directly to using the calculator below."
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Correlation matrix
    st.subheader("Correlation Matrix")
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Feature importance (assuming filtered features are used)
    st.subheader("Feature Importance (Filtered Model)")
    # List of features used during model training (update this manually if needed)
trained_features = ['fico_score', 'income', 'years_employed', 'loan_amt_outstanding']  # Example

feature_importances = pd.DataFrame({
    'Feature': trained_features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)


fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(feature_importances['Feature'], feature_importances['Importance'])
ax.set_title("Feature Importances")
ax.set_xlabel("Features")
ax.set_ylabel("Importance")
st.pyplot(fig)


# Loan Risk Calculator
st.subheader("Loan Default Risk Calculator")
fico_score = st.number_input("FICO Score (e.g., 650)", min_value=300, max_value=850, step=1, value=650)
income = st.number_input("Annual Income (e.g., 50000)", min_value=0, step=1000)
years_employed = st.number_input("Years Employed (e.g., 6)", min_value=0, step=1)
loan_amount = st.number_input("Loan Amount Outstanding (e.g., 4000)", min_value=0, step=1000)

if st.button("Calculate Risk"):
    if fico_score and income and years_employed and loan_amount:
        borrower_details = [fico_score, income, years_employed, loan_amount]
        results = calculate_pd_and_loss(model, borrower_details, loan_amount)
        st.write(f"Borrower details: {borrower_details}")
        st.write(f"**Probability of Default (PD): {results['Probability of Default']:.2%}**")
        st.write(f"**Expected Loss: ${results['Expected Loss']:.2f}**")
    else:
        st.error("Please fill in all the fields.")


# Show the full project report below the calculator
st.markdown("""
# 📊 Credit Risk Assessment Project Report

## 🌟 Objective
To develop a machine learning model that predicts the **Probability of Default (PD)** and **Expected Loss** for borrowers based on key financial and employment features, simulating real-world credit risk analysis.

---

## 📄 Dataset Summary
**Source**: Loan Data CSV file  
**Relevant Features**:
- `fico_score`: Credit score of the borrower
- `income`: Annual income
- `years_employed`: Number of years employed
- `loan_amt_outstanding`: Outstanding loan amount

**Removed Features (Leakage Identified)**:
- `credit_lines_outstanding`
- `total_debt_outstanding`

---

## ⚙️ Workflow Overview

### ✏️ Data Preprocessing
- Loaded dataset using Pandas.
- Dropped features causing data leakage.
- Verified and cleaned data types.
- Split dataset into **80% training** and **20% testing**.

### 📊 Model Training & Evaluation

#### Random Forest Classifier (Selected Model)
- **Pros**: Handles non-linear data, interpretable, balanced performance.
- **Metrics**:
  - Accuracy: ~77%
  - AUC-ROC: ~0.76
  - High Recall for Class 1 (defaults): ~56%

#### XGBoost Classifier (Benchmark Model)
- **Pros**: Stronger precision, gradient boosting technique.
- **Metrics**:
  - Accuracy: ~81%
  - AUC-ROC: ~0.77
  - Precision up, but recall down (~43%)

#### Final Choice: **Random Forest**
- Chosen for its **better recall**, which is critical in credit risk.
- Feature importance was interpretable and balanced.

### 📊 Feature Importance
- Top predictors:
  - `fico_score`
  - `years_employed`
  - `income`
- Less impactful:
  - `loan_amt_outstanding`

---

## 🧠 Challenges Faced

### ❌ Overfitting
- Initial model gave 100% training accuracy.
- **Fix**: Validated on separate test set, confirmed leakage, dropped leaked features.

### ⚠️ Data Leakage
- Features like `total_debt_outstanding` revealed direct info about default status.
- **Fix**: Manually dropped such features and retrained the model.

### 🤷 Model Selection Struggles
- Logistic Regression failed to generalize.
- XGBoost was overconfident but missed risk cases.
- **Fix**: Focused on Random Forest for recall and interpretability.

### ⚖️ Testing Edge Cases
- Profiles with low FICO, high loans, etc.
- **Fix**: Created edge case testing function. Model handled most edge cases well.

### 🚧 Local Development Barriers
- PATH issues, terminal errors, environment setup.
- **Fix**: Resolved via environment variable setup, Python + pip fixes, and Streamlit.

---

## 🌐 Streamlit App Functionality
- Allows users to input:
  - FICO Score
  - Income
  - Years Employed
  - Loan Amount
- Computes:
  - **Probability of Default** using Random Forest model
  - **Expected Loss** using:  
    `Expected Loss = PD × (1 - Recovery Rate) × Loan Amount`
- Displayed in real time through an interactive UI.

---

## 🚀 Key Takeaways
- Learned complete ML workflow: from raw CSV to deployed web app.
- Understood impact of feature leakage and proper testing.
- Balanced model selection with real-world cost (false negatives).
- Gained hands-on experience with:
  - Scikit-learn
  - Random Forests and XGBoost
  - Streamlit for web app deployment
  - Virtual environments and package management

---

## 🚤 Next Steps
- Save the model with `joblib` and reuse without retraining.
- Improve feature set (e.g., employment type, loan purpose).
- Deploy the app on **Streamlit Cloud** or **Heroku**.
- Add data visualizations (e.g., feature importances, risk profiles).

---

*This report reflects an end-to-end machine learning project tailored for real-world credit scoring and will serve as a valuable showcase of applied data science and deployment skills.*
""")



