import streamlit as st
import numpy as np
import pickle

# Load trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature columns order saved from training
with open('feature_columns.pkl', 'rb') as f:
    feature_columns_order = pickle.load(f)

st.title("Amazon Delivery Time Prediction")

# Numeric inputs
distance = st.number_input('Distance (km)', min_value=0.0, max_value=100.0, value=10.0)
agent_age = st.number_input('Agent Age', min_value=15, max_value=60, value=30)
agent_rating = st.slider('Agent Rating', min_value=1.0, max_value=5.0, step=0.1, value=4.5)

# Categorical inputs
traffic_levels = ['Low', 'Medium', 'Jam', 'NaN']
weather_conditions = ['Sunny', 'Fog', 'Stormy', 'Sandstorms', 'Windy']
vehicles = ['motorcycle', 'scooter', 'van']
areas = ['Other', 'Semi-Urban', 'Urban']
categories = ['Books', 'Clothing', 'Cosmetics', 'Electronics', 'Grocery',
              'Home', 'Jewelry', 'Kitchen', 'Outdoors', 'Pet Supplies', 'Shoes',
              'Skincare', 'Snacks', 'Sports', 'Toys']

traffic = st.selectbox('Traffic', traffic_levels)
weather = st.selectbox('Weather', weather_conditions)
vehicle = st.selectbox('Vehicle', vehicles)
area = st.selectbox('Area', areas)
category = st.selectbox('Category', categories)

# Initialize features dictionary with all zeros
feature_dict = dict.fromkeys(feature_columns_order, 0)

# Fill numeric features
feature_dict['Agent_Age'] = agent_age
feature_dict['Agent_Rating'] = agent_rating
feature_dict['Distance_KM'] = distance

# One-hot encode categorical features - set 1 for selected value keys
feature_dict[f'Traffic_{traffic}'] = 1
feature_dict[f'Weather_{weather}'] = 1
feature_dict[f'Vehicle_{vehicle}'] = 1
feature_dict[f'Area_{area}'] = 1
feature_dict[f'Category_{category}'] = 1

# Create features array in exact order
features = np.array([feature_dict[col] for col in feature_columns_order]).reshape(1, -1)

if st.button('Predict Delivery Time'):
    prediction = model.predict(features)
    st.success(f"Predicted Delivery Time: {prediction[0]:.2f} minutes")
