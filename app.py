# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Define class names for HAR dataset
class_names = ['STANDING' 'SITTING' 'LAYING' 'WALKING' 'WALKING_DOWNSTAIRS'
 'WALKING_UPSTAIRS']

# Load pre-trained models
@st.cache_resource
def load_model(model_name):
    return joblib.load(f"models/{model_name}.pkl")

# Model selection options
model_dict = {
    "Logistic Regression": "logistic_model",
    "SVC": "svc_rbf_model",
    "XGBoost": "xgbmodel",
    "Artificial Neural Network (ANN)": "ann_model",
    "Random Forest": "randomforestmodel"
}

# Streamlit UI
st.title("🏃‍♂️ Human Activity Recognition (HAR) Model Evaluation")
st.write("Select models and upload a test CSV to evaluate the accuracy and make predictions.")

# Model selection using checkboxes
selected_models = []
for model_name in model_dict.keys():
    if st.checkbox(f"Use {model_name}", key=model_name):
        selected_models.append(model_name)

# File uploader for test dataset
uploaded_file = st.file_uploader("📤 Upload a test CSV file", type=["csv"])

# Display instructions
if uploaded_file:
    test_data = pd.read_csv(uploaded_file)
    st.write("✅ Test data successfully loaded!")

    # Check if target column exists
    if 'Activity' not in test_data.columns:
        st.error("❗️ The CSV must contain an 'Activity' column.")
    else:
        X_test = test_data.drop(columns=['Activity'])
        y_test = test_data['Activity']

        # Iterate over selected models
        for model_name in selected_models:
            model_file = model_dict[model_name]
            model = load_model(model_file)

            # Predict and evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=class_names)

            # Display results
            st.subheader(f"📊 Results for {model_name}")
            st.write(f"🎯 **Accuracy:** {accuracy * 100:.2f}%")
            st.text("📄 Classification Report")
            st.text(report)

            # Show prediction comparison
            st.write("🔎 **Predictions vs. Actual:**")
            results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            st.write(results_df.head(10))

else:
    st.warning("⚠️ Please upload a test CSV file to continue.")

# Footer
st.markdown("---")
st.markdown("💡 Built with ❤️ using Streamlit | Human Activity Recognition (HAR) Model")