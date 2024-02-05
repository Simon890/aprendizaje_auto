import streamlit as st
import joblib
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from os import path
import random
import sys

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
        input = {}
        submit_btn = True #Aunque esta variable no se use es necesario tenerla para que el submit button se muestre en el html.
        with st.form(key="form_" + self.__rand_id()):
            for column in self.columns:
                value = st.number_input(f"Valor para {column}:", value=0.0, step=1.0)
                input[column] = value
            input_df = pd.DataFrame([input])
            
            submit_btn = st.form_submit_button(label="Predecir", on_click=lambda: self.__predict(input_df))
        st.markdown(
            """
            Trabajo Practio Aprendizaje Automatico:<br>
            [ Github ](https://github.com/Simon890/aprendizaje_auto)
            """, unsafe_allow_html=True
        )
            
    def __predict(self, input):
        """
        Realiza la predicción
        """
        pred = self.pipeline.predict(input)
        value = pred[0]
        st.write("Resultado:")
        st.write(self.props["result_cb"](value))
    
    def __rand_id(self):
        return "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for i in range(6))

model = None
model_type = "reg"

if len(sys.argv) == 2:
    model_type = sys.argv[1]
if model_type == "reg":
    model = StreamlitModel("reg.pkl", title="Regresión Lineal")
else:
    model = StreamlitModel("clas.pkl", title="Regresión Logística")

model.run()