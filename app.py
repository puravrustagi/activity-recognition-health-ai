# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load transformers and selected columns
@st.cache_resource
def load_transformers():
    scaler = joblib.load('models/scaler.pkl')
    pca = joblib.load('models/pca_transformer.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return scaler, pca, encoder

# Load pre-trained models
@st.cache_resource
def load_model(model_name):
    return joblib.load(f"models/{model_name}.pkl")

# Define class names for HAR dataset
class_names = ['STANDING' 'SITTING' 'LAYING' 'WALKING' 'WALKING_DOWNSTAIRS'
 'WALKING_UPSTAIRS']

# Model selection options
model_dict = {
    "Logistic Regression": "logistic_model",
    "SVC": "svc_rbf_model",
    "XGBoost": "xgbmodel",
    "Artificial Neural Network (ANN)": "ann_model",
    "Random Forest": "randomforestmodel",
    "Ensemble Model": "ensemble_model"
}

# Streamlit UI
st.title("🏃‍♂️ Human Activity Recognition (HAR) Model Evaluation with PCA")
st.write("Select models and upload a test CSV to evaluate the accuracy and make predictions.")

# Model selection using checkboxes
selected_models = []
for model_name in model_dict.keys():
    if st.checkbox(f"Use {model_name}", key=model_name):
        selected_models.append(model_name)

# File uploader for test dataset
uploaded_file = st.file_uploader("📤 Upload a test CSV file", type=["csv"])

# Load transformers and selected columns
scaler, pca, encoder = load_transformers()

# Display instructions if no file uploaded
if uploaded_file:
    test_data = pd.read_csv(uploaded_file)
    st.write("✅ Test data successfully loaded!")

    # Check if 'Activity' column exists in test data
    if 'Activity' not in test_data.columns:
        st.error("❗️ The CSV must contain an 'Activity' column.")
    else:
        # Drop correlated columns
        X_test = test_data.drop(columns=['Activity'])
        y_test = test_data['Activity']

        #st.write("COLUMNS IN TEST DATA:",X_test.shape)

        # Apply Standard Scaling
        X_scaled = scaler.transform(X_test)
        #st.write("COLUMNS AFTER SCALING:",X_scaled.shape)

        # Apply PCA on Test Data
        X_pca = pca.transform(X_scaled)
       # st.write("COLUMNS AFTER PCA:",X_pca.shape)
        # Iterate over selected models
        for model_name in selected_models:
            model_file = model_dict[model_name]
            model = load_model(model_file)

            # Predict and evaluate
            y_pred = model.predict(X_pca)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=class_names)

            # Display results
            st.subheader(f"📊 Results for {model_name}")
            st.write(f"🎯 **Accuracy:** {accuracy * 100:.2f}%")
            st.text("📄 Classification Report")
            st.text(report)

            # Show prediction comparison
            st.write("🔎 **Predictions vs. Actual:**")
            results_df = pd.DataFrame({"Actual": y_test, "Predicted": encoder.inverse_transform(y_pred)})
            st.write(results_df.head(10))

else:
    st.warning("⚠️ Please upload a test CSV file to continue.")

# Footer
st.markdown("---")
st.markdown("💡 Built with ❤️ using Streamlit | Human Activity Recognition (HAR) Model")