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
st.markdown("---")
st.header("📄 Project Summary")

with st.expander("Click to expand detailed technical report"):
    st.markdown("""
    ### 🎯 Objective
    To develop a machine learning model that predicts the **Probability of Default (PD)** and **Expected Loss** for borrowers based on key financial and employment features, simulating real-world credit risk analysis.

    ### 📁 Dataset Summary
    **Features used**: `fico_score`, `income`, `years_employed`, `loan_amt_outstanding`  
    **Removed due to data leakage**: `credit_lines_outstanding`, `total_debt_outstanding`

    ### ⚙️ Workflow
    - Cleaned and preprocessed the data
    - Removed leaked features
    - Split into training/testing sets (80/20)
    - Trained and compared models: Random Forest, XGBoost

    ### ✅ Final Model: Random Forest Classifier
    - **Accuracy**: ~77%
    - **AUC-ROC**: ~0.76
    - **Recall for default cases**: ~56%
    - Chosen over XGBoost due to better recall and interpretability

    ### 🧠 Challenges & Fixes
    - **Overfitting**: Detected 100% training accuracy → fixed by removing leaked features
    - **Data Leakage**: Found and dropped variables with direct info about target
    - **Edge Case Testing**: Created profiles to test extreme borrower scenarios

    ### 🌐 App Features
    - Collects borrower data via form
    - Calculates PD using trained Random Forest
    - Computes Expected Loss using:
      ```
      Expected Loss = PD × (1 - Recovery Rate) × Loan Amount
      ```

    ### 🧰 Tools Used
    - Python (Scikit-learn, Pandas)
    - Streamlit for deployment
    - Joblib for model serialization

    ### 🚀 Next Steps
    - Add feature visualizations
    - Deploy on Streamlit Cloud
    - Enhance input flexibility and model tuning
    """)


