from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def train_model(X_train, y_train):
    # Columnas para el preprocesamiento
    cols_to_log = ['footfall', 'USS', 'RP']
    numeric_cols = ['tempMode', 'AQ', 'CS', 'VOC', 'IP', 'Temperature']

    # Preprocesador
    preprocessor_pipeline_1_refined = ColumnTransformer(
        transformers=[
            ('log', PowerTransformer(method='yeo-johnson'), cols_to_log),
            ('scaler', StandardScaler(), numeric_cols)
        ],
        remainder='passthrough'
    )

    # Modelo y sus hiperparámetros
    model = LogisticRegression(
        random_state=42,
        solver='liblinear',
        class_weight='balanced',
        C=0.1,
        penalty='l2'
    )

    # Pipeline completo
    pipeline_1 = Pipeline(steps=[
        ('preprocessor', preprocessor_pipeline_1_refined),
        ('balancer', SMOTE(random_state=42)),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('classifier', model)
    ])

    # Entrenar el modelo con los datos de entrenamiento
    pipeline_1.fit(X_train, y_train)
    return pipeline_1