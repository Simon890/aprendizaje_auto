import streamlit as st
import joblib
import pandas as pd
import os
import keras 

input_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
       'Temp9am', 'Temp3pm', 'RainToday', 'RainTomorrow', 'Canberra',
       'Melbourne', 'MelbourneAirport', 'Sydney', 'SydneyAirport',
       'WindGustDir__E', 'WindGustDir__N', 'WindGustDir__S', 'WindGustDir__W',
       'WindDir9am__E', 'WindDir9am__N', 'WindDir9am__S', 'WindDir9am__W',
       'WindDir3pm__E', 'WindDir3pm__N', 'WindDir3pm__S', 'WindDir3pm__W',
       'Season__1', 'Season__2', 'Season__3', 'Season__4'
]

path_dir=os.path.dirname(os.path.abspath(__file__))
pkl_path=os.path.join(path_dir, 'models/rain.pkl')
pipe = joblib.load(pkl_path)

st.title('Modelo predictor de lluvia')

def get_user_input():

    input_dict = {}

    with st.form(key='default_form'):
        for feat in input_features:
            input_value = st.number_input(f"Enter value for {feat}", value=0.0, step=0.01)
            input_dict[feat] = input_value

       
        submit_button = st.form_submit_button(label='Submit')

    return pd.DataFrame([input_dict]), submit_button

user_input, submit_button = get_user_input()


# When the 'Submit' button is pressed, perform the prediction
if submit_button:
    # Predict wine quality
    prediction = pipe.predict(user_input)
    prediction_value = prediction[0]

    # Display the prediction
    st.header("Predicted Quality")
    st.write(prediction_value)
    

st.markdown(
    """
    Trabajo Practio Aprendizaje Automatico:<br>
    [ Github ](https://github.com/Simon890/aprendizaje_auto)
    """, unsafe_allow_html=True
)