import sys
import joblib
from datetime import datetime
from os import path
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

class StreamlitModel:
    filename = None
    columns : np.ndarray
    pipeline : Pipeline = None
    props = {
        "title": "Predecir lluvia",
        "result_cb": lambda value: value,
        "cities": ["Canberra", "Melbourne", "Sydney"]
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
            for column in [ col for col in self.columns if col not in self.props["cities"] ]:
                if column == "DayCos":
                    st.date_input("Valor para Fecha:", key=column)
                else:
                    st.number_input(f"Valor para {column}:", value=0.0, step=1.0, key=column)
            
            st.selectbox(f"Elija una ciudad:",  self.props["cities"], key="City")

            submit_btn = st.form_submit_button(label="Predecir", on_click=self.__submit_btn_cb)
        st.markdown(
            """
            Trabajo Practio Aprendizaje Automatico:<br>
            [ Github ](https://github.com/Simon890/aprendizaje_auto)
            """, unsafe_allow_html=True
        )
    
    def __submit_btn_cb(self):
        input_form = {}
        for column in  [ col for col in self.columns if col not in self.props["cities"] ] + ["City"]:

            val = st.session_state[column]

            if column == "DayCos":
                try:
                    doy = val.timetuple().tm_yday
                    val =  np.cos(doy / 365 * 2 * np.pi)
                    input_form["DayCos"] = val

                    print(f"DayOfYear:{doy} DayCos: {val}")
                except Exception as e:
                    print(e)                    
            elif column == "City":
                input_form[val] = 1
                print(f"City: {val} {input_form[val]}")
                for city in [ c for c in self.props["cities"] if c != val]:
                    input_form[city] = 0
                    print(f"City: {city} {input_form[city]}")
                
            else:
                input_form[column] = val

        sorted_input = {k: input_form[k] for k in self.columns}
        input_df = pd.DataFrame([sorted_input])
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