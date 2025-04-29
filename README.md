**INFORME DE PREDICCIÓN AUTOMÁTICA DE FALLOS EN MÁQUINAS MEDIANTE DATOS DE SENSORES**


**Introducción:**

El objetivo de este proyecto es desarrollar un modelo de predicción automática de fallos en distintas máquinas que permita predecir la ocurrencia de posibles averías antes de que ocurran, con el objetivo de:

  - Reducir el tiempo de inactividad de las máquinas al no produrcirse paradas de los equipos no planificadas para la realización de mantenimientos correctivos.
  - Reducir la producción de material defectuoso como consecuencia de los fallos en los equipos.
  - Optimizar los trabajos de mantenimiento, ya que podrá realizarse un mantenimiento preventivo únicamente cuando sea necesario, en base a la predicción del posible fallo, antes de que ocurra.
  - Aumentar la productividad de los equipos.


**Descripción de los datos:**

Los datos utilizados en este proyecto provienen de Kaggle y están disponibles públicamente para su uso.


https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fumerrtx%2Fmachine-failure-prediction-using-sensor-data%2Fdata

Este conjunto de datos contiene datos de sensores recogidos de varias máquinas.

Incluye diversas lecturas de sensores, así como los fallos registrados de las máquinas.

Descripción de las columnas:

- **footfall:**
  - El número de personas u objetos que pasan por la máquina.
  - Influye en el desgaste o acumulación de partículas, afectando el rendimiento.

- **tempMode:**
  - El modo o ajuste de temperatura en el que opera la máquina.
  - Está relacionado con diferentes condiciones de trabajo.

- **AQ:**
  - Índice de calidad (pureza) del aire cerca de la máquina.
  - Una mala calidad del aire (presencia de polvo, humedad, partículas) puede contribuir al deterioro o mal funcionamiento de la máquina.

- **USS:**
  - Datos del sensor ultrasónico, que indica las mediciones de proximidad a la máquina.
  - Indica la proximidad de materiales, obstáculos o acumulaciones de residuos que podrían afectar el funcionamiento.

- **CS:**
  - Lecturas del sensor de corriente, indicando el uso de corriente eléctrica de la máquina, es decir, el consumo eléctrico de la máquina.
  - Un aumento inesperado de corriente puede indicar sobrecarga o fallo inminente en componentes eléctricos.

- **VOC:**
  - Nivel de compuestos orgánicos volátiles detectado cerca de la máquina, es decir, la concentración de sustancias químicas en el aire.
  - La exposición a altos niveles de VOC puede afectar la vida útil de los componentes internos.

- **RP:**
  - Posición rotacional o RPM (revoluciones por minuto) de las piezas de la máquina.
  - Cambios anómalos en la velocidad pueden ser señales de desgaste mecánico o problemas en la lubricación.

- **IP:**
  - Presión de entrada del fluido (aire, agua, aceite, etc.) a la máquina.
  - Bajadas de presión podrían indicar fugas, bloqueos o problemas en la alimentación del sistema.

- **Temperature:**
  - Temperatura de funcionamiento de la máquina.
  - Un aumento repentino podría señalar fricción excesiva, sobrecarga o fallos en la refrigeración.

- **fail:**
  - Indicador binario de fallo de la máquina (1 para fallo, 0 para ningún fallo).
  - Variable objetivo a predecir.


**Tecnologías utilizadas:**

- Python
- Pandas
- Numpy
- Matplotlib 
- Seaborn
- Scikit-learn
- imblearn
- Streamlit

**Análisis Exploratorio (EDA) y Preprocesamiento de los datos:**

Se realizó un análisis para identificar correlaciones entre las variables y detectar patrones que pudieran influir en la predicción de los fallos en las máquinas.

**Modelado y evaluación de modelos:**
- **Limpieza de datos**: se eliminaron las filas duplicadas y se trataron los valores atípipcos.
- **Preprocesamiento de datos**: se crearon 3 pipelines con distintos procesamientos de datos con el fin de adecuar el preprocesamiento de los datos al tipo de modelo a entrenar.
  -  **Pipeline 1**: Transformación Logarítmica + Normalización + Balanceo de la variable objetivo, para el entrenamiento de modelos lineales y modelos basados en distancia y modelos probabilísticos.
  - **Pipeline 2**: Transformación Logarítmica + Balanceo de la variable objetivo, para modelos basados en árboles y modelos probabilísticos.
  - **Pipeline 3**: Solo Balanceo de la variable objetivo, para modelos basados en árboles.
- **Modelos entrenados**:
  - **Modelos lineales**:
    - LogisticRegression: Modelo lineal que estima la probabilidad de una clase en función de una combinación lineal de las características de entrada. Es comúnmente utilizado para problemas de clasificación binaria y multiclase.
  - **Modelos Basados en Árboles**:
    - DecisionTreeClassifier: Modelo basado en árboles de decisión que divide los datos en ramas según las decisiones tomadas en cada nodo, basadas en los valores de las características. Es intuitivo y puede capturar relaciones no lineales.
    - RandomForestClassifier: Conjunto de múltiples árboles de decisión entrenados en subconjuntos aleatorios de los datos y utilizando subconjuntos aleatorios de las características. Las predicciones se realizan por votación de los árboles, lo que mejora la generalización y reduce el sobreajuste.
    -  GradientBoostingClassifier: Algoritmo de boosting que construye un modelo aditivo secuencialmente, donde cada nuevo modelo débil (típicamente árboles de decisión) corrige los errores del modelo anterior. Optimiza la clasificación combinando múltiples modelos débiles en un modelo fuerte.
    - LightGBM: Un framework de boosting de gradiente diseñado para ser distribuido y altamente eficiente. Utiliza técnicas como el muestreo de gradiente basado en histogramas y el crecimiento de hojas por niveles para acelerar el entrenamiento y reducir el uso de memoria.
    - XGBoost: Una implementación popular y eficiente del algoritmo de boosting de gradiente. Proporciona regularización L1 y L2, manejo de valores faltantes y paralelización, lo que lo hace potente y robusto.
  - **Modelos Basados en Distancia**:
    - KNeighborsClassifier: Clasificador basado en la proximidad de los puntos a sus vecinos más cercanos en el espacio de características. La clase de un nuevo punto se decide por la mayoría de las clases de sus k vecinos más cercanos.
    - SVC (Support Vector Classifier): Modelo basado en máquinas de soporte vectorial, que encuentra un hiperplano óptimo en un espacio de alta dimensión para separar las clases, maximizando el margen entre ellas. Puede utilizar diferentes funciones kernel para modelar relaciones no lineales.
  - **Modelos Probabilísticos**:
    - GaussianNB: Modelo basado en la teoría de Bayes, que asume que las características de cada clase siguen una distribución gaussiana (normal). Es un clasificador probabilístico que calcula la probabilidad de pertenencia a cada clase basándose en las probabilidades condicionales de las características.
- **Métricas utilizadas**:
    - Exactitud (Accuracy): Representa el porcentaje de predicciones correctas realizadas por el modelo sobre el total de predicciones. 
    - Precisión (Precision): Mide la proporción de instancias clasificadas como positivas (fallo de la máquina) que realmente lo fueron.
    - Recall (Sensibilidad o Exhaustividad): Mide la proporción de instancias positivas reales (fallos de la máquina) que fueron correctamente identificadas por el modelo. Un alto recall indica que el modelo es bueno para detectar la mayoría de los fallos reales.
    - F1-Score: Es la media armónica de la precisión y el recall, proporcionando una medida equilibrada del rendimiento del modelo, especialmente útil cuando las clases están desbalanceadas. 
    - Curva ROC y AUC (Área Bajo la Curva ROC): La curva ROC grafica la tasa de verdaderos positivos (TPR) contra la tasa de falsos positivos (FPR) para diferentes umbrales de clasificación.
  
**Resultados y conclusiones:**

Finalmente se ha seleccionado el modelo **Logistic Regression** por ser el modelo que ha obtenido el mejor mejor Recall (0,911) y el mejor ROC AUC (0,971) en comparación con los otros modelos. 

**Archivos del proyecto:**
- sensores_01.csv: dataset utilizado para el entrenamiento del modelo.
- cargar_datos.py: archivo para realizar la carga de datos y la separación del dataset en datos de entrenamiento y datos de test.
- data_preprocessing.py: archivo para el preprocesamiento de los datos.
- model_training.py: archivo para el entrenamiento del modelo.
- logisticRegression_fallos.pkl: pipeline del modelo entrenado.