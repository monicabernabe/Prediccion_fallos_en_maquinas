#Importar librerías
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split

#Función para preprocesamiento (filas duplicadas y valores atípicos) y separación de variable objetivo
def cleaning_separate_data(train_data, test_data):
    # Eliminar filas duplicadas
    train_data = train_data.drop_duplicates()
    test_data = test_data.drop_duplicates()

    # Eliminar valores atípicos
    train_data.loc[train_data['footfall'] > 1000, 'footfall'] = 1000
    test_data.loc[test_data['footfall'] > 1000, 'footfall'] = 1000

    # Extracción de la variable objetivo
    y_train = train_data['fail']
    y_test = test_data['fail']

    # Eliminación de la variable objetivo del DataFrame
    X_train = train_data.drop('fail', axis=1)
    X_test = test_data.drop('fail', axis=1)
    return X_train, y_train, X_test, y_test
