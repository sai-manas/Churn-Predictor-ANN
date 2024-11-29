import pickle
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


#Load the model
model = tf.keras.models.load_model("models/model.keras", compile=False)

#Load the Encoders and Scaler
with open("models/label_encoder_gender.pkl","rb") as file:
    label_encoder_gender = pickle.load(file)

with open("models/onehot_encoder_geo.pkl","rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("models/scalar.pkl","rb") as file:
    scaler = pickle.load(file)

## app
st.title("Customer Churn Prediction")

# User input
name = st.text_input("Enter your name:", placeholder="Your name here")
geography = st.selectbox("Geography",onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
has_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox("Is Active Member",[0,1])
age = st.slider("Age",18,92)
tenure = st.slider("Tenure",0,10)
num_of_products = st.slider("Number of Products",1,4)
estimated_salary = st.number_input("Estimated Salary")
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")

if st.button("Submit"):
    #prepare the input as df
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)


    # Predict churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.markdown(
        f"<h4 style='font-weight:bold;'>Churn Probability for {name}: {prediction_proba:.2f}</h4>",
        unsafe_allow_html=True
    )

    if prediction_proba > 0.5:
        st.markdown(
            """
            <style>
            @keyframes discoGlow {
                0% { border-color: red; box-shadow: 0 0 10px red; }
                25% { border-color: yellow; box-shadow: 0 0 10px yellow; }
                50% { border-color: green; box-shadow: 0 0 10px green; }
                75% { border-color: blue; box-shadow: 0 0 10px blue; }
                100% { border-color: red; box-shadow: 0 0 10px red; }
            }
            .disco-text {
                font-size: 2em;
                font-weight: bold;
                text-align: center;
                border: 5px solid;
                border-radius: 10px;
                padding: 10px;
                animation: discoGlow 2s infinite;
            }
            </style>
            <div class="disco-text">The customer {name} is likely to churn.</div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            @keyframes discoGlow {
                0% { border-color: lime; box-shadow: 0 0 10px lime; }
                25% { border-color: cyan; box-shadow: 0 0 10px cyan; }
                50% { border-color: magenta; box-shadow: 0 0 10px magenta; }
                75% { border-color: orange; box-shadow: 0 0 10px orange; }
                100% { border-color: lime; box-shadow: 0 0 10px lime; }
            }
            .disco-text {
                font-size: 2em;
                font-weight: bold;
                text-align: center;
                border: 5px solid;
                border-radius: 10px;
                padding: 10px;
                animation: discoGlow 2s infinite;
            }
            </style>
            <div class="disco-text">The customer {name} is not likely to churn.</div>
            """,
            unsafe_allow_html=True
        )


