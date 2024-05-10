import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Load saved models and scaler
scaler = load("E:\DOAN\TranTienDiep_2020603359\TranTienDIep\data/scaler.joblib")
nb_model = load('E:\DOAN\TranTienDiep_2020603359\TranTienDIep\data/gaussian_naive_bayes_model.joblib')
knn_model = load('E:\DOAN\TranTienDiep_2020603359\TranTienDIep\data/weighted_knn_model.joblib')

# Load training data
X_train = pd.DataFrame(load('E:\DOAN\TranTienDiep_2020603359\TranTienDIep\data/X_train.joblib'))
y_train = pd.Series(load('E:\DOAN\TranTienDiep_2020603359\TranTienDIep\data/y_train.joblib'))

# Set full page background image
st.markdown(
    """
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #FF4B4B;
    }
    .custom-color {
        color: #3333FF;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<p class="big-font">Weather Classification Application</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

def predict_with_model(model, features):
    predictions = model.predict(features)
    return predictions

def new_fusion_model(X_train, y_train, X_test, n_neighbors_percent=20):
    n_neighbors = max(1, int(n_neighbors_percent / 100 * len(X_train)))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    neighbors = knn.kneighbors(X_test, return_distance=False)

    nb = GaussianNB()
    fusion_preds = []
    for i in range(len(X_test)):
        nb.fit(X_train.iloc[neighbors[i]], y_train.iloc[neighbors[i]])
        pred = nb.predict(X_test[i:i+1])
        fusion_preds.append(pred[0])
    return np.array(fusion_preds)

with col1:
    st.markdown('<p class="custom-color">Enter Manual Parameters</p>', unsafe_allow_html=True)
    with st.form(key='input_form'):
        tempmax = st.number_input('TempMax', value=30.0)
        tempmin = st.number_input('TempMin', value=20.0)
        temp = st.number_input('Temp', value=25.0)
        feelslikemax = st.number_input('FeelsLikeMax', value=32.0)
        feelslikemin = st.number_input('FeelsLikeMin', value=21.0)
        feelslike = st.number_input('FeelsLike', value=26.5)
        humidity = st.number_input('Humidity', value=80.0)
        precip = st.number_input('Precip', value=0.0)
        windgust = st.number_input('WindGust', value=20.0)
        windspeed = st.number_input('WindSpeed', value=10.0)
        cloudcover = st.number_input('CloudCover', value=40.0)
        model_choice = st.selectbox("Choose the prediction model", ('Naive Bayes', 'KNN', 'Fusion'))
        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        input_data = np.array([[tempmax, tempmin, temp, feelslikemax, feelslikemin, feelslike, humidity, precip,
                                windgust, windspeed, cloudcover]])
        scaled_data = scaler.transform(input_data)
        if model_choice == 'Naive Bayes':
            prediction = predict_with_model(nb_model, scaled_data)
        elif model_choice == 'KNN':
            prediction = predict_with_model(knn_model, scaled_data)
        elif model_choice == 'Fusion':
            prediction = new_fusion_model(X_train, y_train, scaled_data)
        st.success(f'Prediction using {model_choice}: {prediction[0]}')

with col2:
    st.markdown('<p class="custom-color">Upload CSV File for Weather Data</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader('', type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data)
        model_choice_csv = st.selectbox("Choose the prediction model for CSV", ('Naive Bayes', 'KNN', 'Fusion'), key='csv')
        if st.button('Classify CSV File'):
            input_features = data.drop(["name", "datetime", "description", "icon"], axis=1, errors='ignore')
            input_features = input_features.fillna(input_features.mean())
            scaled_features = scaler.transform(input_features)
            if model_choice_csv == 'Naive Bayes':
                predictions = predict_with_model(nb_model, scaled_features)
            elif model_choice_csv == 'KNN':
                predictions = predict_with_model(knn_model, scaled_features)
            elif model_choice_csv == 'Fusion':
                predictions = new_fusion_model(X_train, y_train, scaled_features)
            data['Prediction'] = predictions
            st.write('Prediction results:')
            st.dataframe(data[['name', 'datetime', 'Prediction', 'icon']])
            data.to_csv('prediction_results.csv', index=False)
            st.success('Predicted data has been saved into "prediction_results.csv"')
