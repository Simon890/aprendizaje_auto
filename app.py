import streamlit as st
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from os import path
import random
import sys
from tensorflow.keras.saving import load_model

class StreamlitModel:
    filename = None
    columns : np.ndarray
    pipeline : Pipeline = None
    props = {
        "title": "Predecir lluvia",
        "result_cb": lambda value: value
    }
    
    def __init__(self, filename : str, **kwargs):
        self.filename = path.join("models", filename)
        self.props = {**self.props, **kwargs}
        self.__load_model()
    
    def run(self):
        self.__setup_streamlit()
    
    def __load_model(self):
        """
        Carga el modelo
        """

        pipeline_data = joblib.load(self.filename)
        self.columns = pipeline_data["columns"]
        self.pipeline = pipeline_data["pipeline"]
    
    def __setup_streamlit(self):
        """
        Inicia streamlit
        """
        st.title(self.props["title"])
        submit_btn = True #Aunque esta variable no se use es necesario tenerla para que el submit button se muestre en el html.
        with st.form(key="form_"):
            for column in self.columns:
                st.number_input(f"Valor para {column}:", value=0.0, step=1.0, key=column)
            
            submit_btn = st.form_submit_button(label="Predecir", on_click=self.__submit_btn_cb)
        st.markdown(
            """
            Trabajo Practio Aprendizaje Automatico:<br>
            [ Github ](https://github.com/Simon890/aprendizaje_auto)
            """, unsafe_allow_html=True
        )
    
    def __submit_btn_cb(self):
        input_form = {}
        for column in self.columns:
            input_form[column] = st.session_state[column]
        input_df = pd.DataFrame([input_form])
        self.__predict(input_df)
            
    def __predict(self, input):
        """
        Realiza la predicción
        """
        pred = self.pipeline.predict(input)
        value = pred[0]
        st.write("Resultado:")
        st.write(self.props["result_cb"](value))

model = None
model_type = "clas"

if len(sys.argv) == 2:
    model_type = sys.argv[1]

if model_type == "reg":
    print("Starting LINEAL regression model")
    model = StreamlitModel("reg.pkl", title="Regresión Lineal")
else:
    def render_pred(value):
        return "Va a llover" if value >= 0.5 else "No va a llover"
    model = StreamlitModel("clas.pkl", title="Regresión Logística", result_cb=render_pred)

model.run()