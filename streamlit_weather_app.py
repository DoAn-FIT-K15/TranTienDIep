import streamlit as st
import numpy as np
import pandas as pd
from weatherapi.api import get_weather_data
from models.loader import load_scaler, load_models, load_training_data
from models.predictor import predict_with_model, new_fusion_model

# Load everything
scaler = load_scaler()
nb_model, knn_model = load_models()
X_train, y_train = load_training_data()

def main():
    st.title('Weather Classification and Forecast Application')
    # Sidebar for manual input
    submit_button = False
    with st.sidebar:
        st.header("Manual Weather Parameters")
        with st.form(key='manual_input_form'):
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
            submit_button = st.form_submit_button('Predict')

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

    # Sidebar for CSV Upload
    classify_button = False
    with st.sidebar:
        st.header("Upload CSV File for Weather Data")
        uploaded_file = st.file_uploader("Choose a file", type=['csv'])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            model_choice_csv = st.selectbox("Model for CSV Prediction", ('Naive Bayes', 'KNN', 'Fusion'),
                                            key='csv_model')
            classify_button = st.button('Classify CSV File')

    if classify_button:
        input_features = data.drop(["name", "datetime", "description", "icon"], axis=1, errors='ignore')
        scaled_features = scaler.transform(input_features)
        if model_choice_csv == 'Naive Bayes':
            predictions = predict_with_model(nb_model, scaled_features)
        elif model_choice_csv == 'KNN':
            predictions = predict_with_model(knn_model, scaled_features)
        elif model_choice_csv == 'Fusion':
            predictions = new_fusion_model(X_train, y_train, scaled_features)
        data['Prediction'] = predictions
        st.dataframe(data[['name', 'datetime', 'Prediction', 'icon']])
        data.to_csv('prediction_results.csv', index=False)
        st.success('Predicted data has been saved into "prediction_results.csv"')

    # Sidebar for fetching weather data
    fetch_weather_button = False
    with st.sidebar:
        st.header("Get Weather Data")
        city = st.text_input("Enter city name", "Nghe an")
        api_key = 'GYJXXDWV5CBJYDAJW4ZYSTPU6'  # Replace with your actual API key

    if st.button("Fetch Weather now"):
        weather_data = get_weather_data(city, api_key)
        if isinstance(weather_data, str):
            st.error(weather_data)
        else:
            st.markdown('<div class="weather-card">', unsafe_allow_html=True)
            if 'icon' in weather_data['currentConditions']:
                icon_url = f"https://youriconsource.com/{weather_data['currentConditions']['icon']}.png"
                st.markdown(f'<img src="{icon_url}" class="weather-icon">', unsafe_allow_html=True)
            st.markdown(f"### Weather for {city} now")
            # st.markdown(f"**TempMax:** {weather_data['currentConditions']['tempmax']} °C")
            # st.markdown(f"**TempMin:** {weather_data['currentConditions']['tempmin']} °C")
            st.markdown(f"**Temp:** {weather_data['currentConditions']['temp']} °C")
            # st.markdown(f"**Feels max:** {weather_data['currentConditions']['feelslikemax']} °C")
            # st.markdown(f"**Feels min:** {weather_data['currentConditions']['feelslikemin']} °C")
            st.markdown(f"**Feels Like:** {weather_data['currentConditions']['feelslike']} °C")
            st.markdown(f"**humidity:** {weather_data['currentConditions']['humidity']}")
            st.markdown(f"**precip:** {weather_data['currentConditions']['precip']}")
            st.markdown(f"**windgust:** {weather_data['currentConditions']['windgust']}")
            st.markdown(f"**windspeed:** {weather_data['currentConditions']['windspeed']}")
            st.markdown(f"**cloudcover:** {weather_data['currentConditions']['cloudcover']}")
            st.markdown(f"**Conditions:** {weather_data['currentConditions']['conditions']}")
            # Display full JSON data in an expandable section
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
