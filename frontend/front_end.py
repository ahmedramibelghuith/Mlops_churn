import requests
import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from streamlit_option_menu import option_menu

# Get the backend URL from the environment variable
backend_url = os.getenv("BACKEND_URL", "http://localhost:8080")

# Create a horizontal navigation bar
selected_page = option_menu(
    menu_title=None,
    options=["Home", "Predict with Form", "Predict with CSV", "Power BI Dashboard"],
    icons=["house", "clipboard-data", "file-earmark-arrow-up", "graph-up"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important"},
        "icon": {"color": "#00796b", "font-size": "20px"}, 
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "color": "#004d40",
        },
        "nav-link-selected": {"background-color": "#00796b", "color": "white"},
    },
)

# Apply global styling
st.markdown(
    """
    <style>
    /* General background color */
    
    /* Main content box styling */
    .main {
       /* White for contrast */
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
    }
    /* Header and text colors */
    h1, h2, h3 {
        font-family: 'Arial', sans-serif;
        color: #00796b; /* Teal */
    }
    h1 {
        border-bottom: 3px solid #004d40;
        padding-bottom: 12px;
    }
    /* Button styling */
    .stButton>button {
        background-color: #00796b; /* Teal for buttons */
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
    }
    .stButton>button:hover {
        background-color: #004d40; /* Darker teal on hover */
    }
    /* File uploader styling */
    .stFileUploader {
        padding: 12px;
        border: 2px dashed #00796b;
        border-radius: 8px;
    }
    /* Navigation bar styling */
    .css-1m3hb7r {
        background-color: #004d40; /* Dark teal for nav bar */
    }
    .css-1m3hb7r a {
        color: #e0f7fa !important; /* Light text color for nav links */
        font-size: 18px;
    }
    .css-1m3hb7r a:hover {
        color: #ffffff !important; /* Pure white on hover */
    }
    .css-1m3hb7r .nav-item-selected {
        background-color: #00796b; /* Active state background */
        color: #ffffff !important; /* Active state text color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Home Page
if selected_page == "Home":
    st.title("📈 Churn Prediction App")
    st.image(
        str(Path(__file__).with_name("churn_dashboard.jpeg")),
        caption="Understand and predict customer churn"
    )
    st.markdown(
        """
        ## Welcome!
        This application leverages advanced machine learning to predict customer churn based on user inputs.

        ### Features:
        - **Interactive Form**: Enter customer details to get an instant prediction.
        - **CSV Upload**: Analyze multiple customer records at once.
        - **Power BI Dashboard**: Visualize churn trends and data insights.
        
        Use the navigation bar to explore the app.
        """
    )

# Predict with Form Page
elif selected_page == "Predict with Form":
    st.title("📋 Predict Customer Churn - Form")
    st.subheader("Fill in the customer details to get a churn prediction.")

    with st.form("form1", clear_on_submit=False):
        # Input fields
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=22, step=1)
            gender = st.selectbox("Gender", ["Female", "Male"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=25, step=1)
            usage_frequency = st.number_input("Usage Frequency", min_value=0, value=14, step=1)
            support_calls = st.number_input("Support Calls", min_value=0, value=4, step=1)

        with col2:
            payment_delay = st.number_input("Payment Delay (days)", min_value=0, value=27, step=1)
            subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
            total_spend = st.number_input("Total Spend (USD)", min_value=0, value=598, step=1)  # Converted to int
            last_interaction = st.number_input("Days Since Last Interaction", min_value=0, value=9, step=1)

        # Collect data in a dictionary
        input_data = {
            "Age": int(age),
            "Gender": gender,
            "Tenure": int(tenure),
            "Usage Frequency": int(usage_frequency),
            "Support Calls": int(support_calls),
            "Payment Delay": int(payment_delay),
            "Subscription Type": subscription_type,
            "Contract Length": contract_length,
            "Total Spend": int(total_spend),  # Converted to int
            "Last Interaction": int(last_interaction),
        }

        # Submit button
        submit = st.form_submit_button("💡 Get Prediction")

        if submit:
            try:
                response = requests.post(f"{backend_url}/predict", data=json.dumps(input_data), headers={"Content-Type": "application/json"}, timeout=60)
                response.raise_for_status()
                predictions = response.json().get("predictions")

                # Display the prediction result
                if predictions == [0]:
                    st.success("✅ The customer is not likely to churn.")
                elif predictions == [1]:
                    st.warning("⚠️ The customer is likely to churn.")
                else:
                    st.error(predictions)
                    st.error("❌ An error occurred while processing the prediction.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Predict with CSV Page
# Predict with CSV Page
elif selected_page == "Predict with CSV":
    st.title("📁 Predict Customer Churn - CSV")
    st.subheader("Upload a CSV file to analyze multiple customer records.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Send the file to the backend
            response = requests.post(
                f"{backend_url}/predict/csv",
                files={"file": (uploaded_file.name, uploaded_file, "text/csv")},
                timeout=60,
            )
            
            # Check if the response is valid
            if response.status_code == 200:
                predictions = response.json().get("predictions")

                # Display predictions for each record
                st.subheader("Predictions for uploaded data")
                for idx, prediction in enumerate(predictions, start=1):
                    st.text(f"Record {idx}: {'⚠️ Likely to churn' if prediction == 1 else '✅ Not likely to churn'}")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")


# Power BI Dashboard Page
elif selected_page == "Power BI Dashboard":
    st.title("📊 Churn Analysis Dashboard")
    st.markdown(
        """
        This dashboard provides an overview of trends and insights into customer churn data.
        """
    )

    st.markdown(
        f'<iframe title="Churn Dashbord" width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiZDVlZmNlNTgtMmUzNC00YWRkLWIwMjUtNzE1YjYyYTg3MmJhIiwidCI6ImRiZDY2NjRkLTRlYjktNDZlYi05OWQ4LTVjNDNiYTE1M2M2MSIsImMiOjl9" frameborder="0" allowFullScreen="true"></iframe>',
        unsafe_allow_html=True,
    )
