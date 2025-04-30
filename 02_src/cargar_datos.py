# Importar librerías
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

# Cargar datos y dividir el conjunto de datos
def load_and_split_data():
    try:
        df_sensores = pd.read_csv(data/'sensores_01.csv')
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'sensores_01.csv'.")
        sys.exit()
    # Dividir el conjunto de datos
    train_data, test_data = train_test_split(df_sensores, test_size=0.2, random_state=42)
    return train_data, test_data