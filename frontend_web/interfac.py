import streamlit as st
import pandas as pd

from src.pipline.prediction_pipeline import ChurnDataClassifer, ChurnData
from src.pipline.training_pipeline import TrainingPipeline

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction App")

st.write("Fill the form below to predict whether the customer will churn.")

# ---------- FORM INPUTS ----------
def user_input_form():

    st.subheader("Customer Information Form")

    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.selectbox("Gender", [0, 1])
        Married = st.selectbox("Married", [0, 1])
        Offer = st.selectbox("Offer", [0, 1, 2, 3])
        Phone_Service = st.selectbox("Phone Service", [0, 1])
        Multiple_Lines = st.selectbox("Multiple Lines", [0, 1])

    with col2:
        Internet_Service = st.selectbox("Internet Service", [0, 1, 2])
        Internet_Type = st.selectbox("Internet Type", [0, 1, 2])
        Online_Security = st.selectbox("Online Security", [0, 1])
        Online_Backup = st.selectbox("Online Backup", [0, 1])
        Device_Protection_Plan = st.selectbox("Device Protection Plan", [0, 1])

    with col3:
        Premium_Tech_Support = st.selectbox("Premium Tech Support", [0, 1])
        Streaming_TV = st.selectbox("Streaming TV", [0, 1])
        Streaming_Movies = st.selectbox("Streaming Movies", [0, 1])
        Streaming_Music = st.selectbox("Streaming Music", [0, 1])
        Unlimited_Data = st.selectbox("Unlimited Data", [0, 1])

    # ----- Second row -----
    st.subheader("Billing & Demographics")
    colA, colB, colC = st.columns(3)

    with colA:
        Contract = st.selectbox("Contract", [0, 1, 2])
        Paperless_Billing = st.selectbox("Paperless Billing", [0, 1])
        Payment_Method = st.selectbox("Payment Method", [0, 1, 2, 3])
        # Customer_Status removed (target column) - not requested from user input

    with colB:
        Age = st.number_input("Age", min_value=18, max_value=100, step=1)
        Number_of_Dependents = st.number_input("Number of Dependents", 0, 10)
        Number_of_Referrals = st.number_input("Number of Referrals", 0, 10)
        Tenure_in_Months = st.number_input("Tenure in Months", 0, 100)

    with colC:
        Avg_Monthly_Long_Distance_Charges = st.number_input("Avg Monthly Long Distance Charges")
        Avg_Monthly_GB_Download = st.number_input("Avg Monthly GB Download")
        Monthly_Charge = st.number_input("Monthly Charge")
        Total_Charges = st.number_input("Total Charges")

    # Create a dictionary for DataFrame
    data = {
        "Gender": Gender,
        "Married": Married,
        "Offer": Offer,
        "Phone_Service": Phone_Service,
        "Multiple_Lines": Multiple_Lines,
        "Internet_Service": Internet_Service,
        "Internet_Type": Internet_Type,
        "Online_Security": Online_Security,
        "Online_Backup": Online_Backup,
        "Device_Protection_Plan": Device_Protection_Plan,
        "Premium_Tech_Support": Premium_Tech_Support,
        "Streaming_TV": Streaming_TV,
        "Streaming_Movies": Streaming_Movies,
        "Streaming_Music": Streaming_Music,
        "Unlimited_Data": Unlimited_Data,
        "Contract": Contract,
        "Paperless_Billing": Paperless_Billing,
        "Payment_Method": Payment_Method,
        "Age": Age,
        "Number_of_Dependents": Number_of_Dependents,
        "Number_of_Referrals": Number_of_Referrals,
        "Tenure_in_Months": Tenure_in_Months,
        "Avg_Monthly_Long_Distance_Charges": Avg_Monthly_Long_Distance_Charges,
        "Avg_Monthly_GB_Download": Avg_Monthly_GB_Download,
        "Monthly_Charge": Monthly_Charge,
        "Total_Charges": Total_Charges,
    }

    return data


# ---------- MAIN UI ----------
with st.form("prediction_form"):
    user_data = user_input_form()
    submit_button = st.form_submit_button("Predict Churn")

# ---------- PREDICTION ----------
if submit_button:
    st.subheader("🔍 Prediction Result")

    try:
        # Create ChurnData object from user inputs
        churn_obj = ChurnData(
            Gender=user_data["Gender"],
            Married=user_data["Married"],
            Offer=user_data["Offer"],
            Phone_Service=user_data["Phone_Service"],
            Multiple_Lines=user_data["Multiple_Lines"],
            Internet_Service=user_data["Internet_Service"],
            Internet_Type=user_data["Internet_Type"],
            Online_Security=user_data["Online_Security"],
            Online_Backup=user_data["Online_Backup"],
            Device_Protection_Plan=user_data["Device_Protection_Plan"],
            Premium_Tech_Support=user_data["Premium_Tech_Support"],
            Streaming_TV=user_data["Streaming_TV"],
            Streaming_Movies=user_data["Streaming_Movies"],
            Streaming_Music=user_data["Streaming_Music"],
            Unlimited_Data=user_data["Unlimited_Data"],
            Contract=user_data["Contract"],
            Paperless_Billing=user_data["Paperless_Billing"],
            Payment_Method=user_data["Payment_Method"],
            Age=user_data["Age"],
            Number_of_Dependents=user_data["Number_of_Dependents"],
            Number_of_Referrals=user_data["Number_of_Referrals"],
            Tenure_in_Months=user_data["Tenure_in_Months"],
            Avg_Monthly_Long_Distance_Charges=user_data["Avg_Monthly_Long_Distance_Charges"],
            Avg_Monthly_GB_Download=user_data["Avg_Monthly_GB_Download"],
            Monthly_Charge=user_data["Monthly_Charge"],
            Total_Charges=user_data["Total_Charges"],
        )

        # Get DataFrame from ChurnData object
        df = churn_obj.get_churn_input_data_frame()

        model = ChurnDataClassifer()
        prediction = model.predict(df)[0]

        if prediction == 1:
            st.error("❌ Customer is Likely to Churn")
        else:
            st.success("✅ Customer Will Stay")
    except Exception as e:
        st.error(f"Error: {e}")

# ---------- TRAINING BUTTON ----------
st.write("---")
if st.button("🚀 Train Model"):
    try:
        train = TrainingPipeline()
        train.run_pipeline()
        st.success("Model Training Completed Successfully!")
    except Exception as e:
        st.error(f"Training Error: {e}")