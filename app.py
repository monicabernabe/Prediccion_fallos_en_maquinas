# Importar librerías
import streamlit as st
import pandas as pd
import pickle as pkl
from sklearn.linear_model import LogisticRegression

# Cargar el modelo entrenado
with open('logisticRegression_fallos.pkl', 'rb') as file:
    model = pkl.load(file)
    
# Interfaz de usuario con Streamlit
st.title("Predicción de Fallos de Maquinaria")
st.write("Introduce los valores de las variables para predecir si habrá un fallo.")

# Lista de variables a cumplimentar
variables = ['footfall', 'tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature']

# Diccionario para almacenar los valores introducidos por el usuario
user_input = {}
all_values_filled = True

# Creación de las casillas de entrada para cada variable
for var in variables:
    value = st.number_input(f"Valor de {var} (ver 'Acerca de' para rango)", value=None)
    user_input[var] = value
    if value is None:
        all_values_filled = False

# Botón de predicción
if st.button("Predicción"):
    if all_values_filled:
        # Crear un DataFrame con los datos del usuario
        input_df = pd.DataFrame([user_input])

        try:
            # Realizar la predicción
            prediction = model.predict(input_df)[0]

            # Mostrar el resultado de la predicción
            if prediction == 1:  # Predicción de fallo
                st.error("¡ALERTA! Se predice un fallo en la máquina. \nPare la máquina y realice mantenimiento preventivo")
            else:
                st.success("No se predice ningún fallo en la máquina.")

        except Exception as e:
            st.error(f"Ocurrió un error al realizar la predicción: {e}")
    else:
        st.warning("Por favor, cumplimente todos los datos.")

st.sidebar.title("Acerca de")
st.sidebar.info(
    '''
    Esta aplicación utiliza un modelo Logistic Regression entrenado con los datos del archivo 'sensores_01' para predecir la ocurrencia de posibles averías en un equipo de producción antes de que ocurran, basándose en los valores de diferentes variables.
    Para predecir la ocurrencia de fallo en el equipo de producción, indicar el valor de las siguientes variables:
    - footfall:
         - Número de personas u objetos que pasan por la máquina.

    - tempMode:
        - El modo o ajuste de temperatura en el que opera la máquina.
        - Los valores posibles son números enteros de 0 a 7.

    - AQ:
        - Índice de calidad (pureza) del aire cerca de la máquina.
        - Los valores posibles son números enteros de 1 (calidad excelente) a 7 (calidad muy mala).

    - USS:
        - Datos del sensor ultrasónico, que indica las mediciones de proximidad a la máquina.
        - Los valores posibles van de 1 (muy próximo) hasta 7 (muy alejado).
    - CS:
        - Lecturas del sensor de corriente, indicando el uso de corriente eléctrica de la máquina.
        - Los valores posibles son números enteros de 1 (muy poca corriente eléctrica) a 7 (máxima corriente eléctrica).

    - VOC:
        - Nivel de compuestos orgánicos volátiles detectado cerca de la máquina.
        - Los valores posibles van de 0 (nivel nulo) hasta 6 (nivel muy alto).

    - RP:
        - Revoluciones por minuto de las piezas de la máquina.

    - IP:
        - Presión de entrada del fluido (aire, agua, aceite, etc.) a la máquina.
        - Los valores posibles van de 1 (presión muy baja) hasta 7 (presión muy alta).

    - Temperature:
        - Temperatura de funcionamiento de la máquina en ºC.

    Pulsar el botón "Predicción" para obtener la predicción.
    '''
)