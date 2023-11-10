import streamlit as st
import numpy as np 
import pandas as pd 
from pandas import read_csv

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
from PIL import Image
import plotly.express as px
from graphviz import Source
import graphviz

def dividirDatasetEntrenamiento():
    #Label encoder: Su propósito principal es convertir etiquetas categóricas en una representación numérica
    label_encoder = LabelEncoder()
    dtAC['gender'] = label_encoder.fit_transform(dtAC['gender'])
    dtAC['ever_married'] = label_encoder.fit_transform(dtAC['ever_married'])
    dtAC['work_type'] = label_encoder.fit_transform(dtAC['work_type'])
    dtAC['Residence_type'] = label_encoder.fit_transform(dtAC['Residence_type'])
    dtAC['smoking_status'] = label_encoder.fit_transform(dtAC['smoking_status'])

    X = dtAC.drop('stroke', axis=1)
    y = dtAC['stroke']

    #Se separa el dataset en 4 datasets diferentes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X, y

def perceptronMulticapa():
    st.subheader('Perceptrón multicapa (Red Neuronal Artificial)')
    X_train, X_test, y_train, y_test, X, y = dividirDatasetEntrenamiento()

    #scalling data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    st.markdown('dfsfsdfdfsdfsdf')
    model.fit(X_train, y_train)
    MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
                solver='lbfgs')
    
    
    accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    precision = cross_val_score(model, X, y, cv=5, scoring='precision_weighted')
    recall = cross_val_score(model, X, y, cv=5, scoring='recall_weighted')
    f1 = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')

    st.markdown('Exactitud de la red neuronal: '+ str(accuracy))
    st.markdown('Presición: '+ str(precision))

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                    index=['Predict Positive:1', 'Predict Negative:0'])


def arbolDeClasificacion():
    st.subheader('Arbol de clasificación')
    
    X_train, X_test, y_train, y_test, X, y = dividirDatasetEntrenamiento()
    st.markdown('Data set de entrenamiento')
    st.dataframe(X_train)

    st.markdown('Data set de prueba')
    st.dataframe(y_train)

    parameter_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    }

    
    clf = DecisionTreeClassifier()
    grid_search = GridSearchCV(clf, parameter_grid, cv=10, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    

    y_pred = best_estimator.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.markdown('Exactitud del modelo: '+ str(accuracy))
    st.markdown('Presición: '+ str(precision))
    dot_data = export_graphviz(best_estimator, out_file=None, feature_names=X.columns, class_names=np.unique(y).astype(str), filled=True, rounded=True, special_characters=True)
    st.text('sdssds')
    #graph = graphviz.Source(dot_data)

    # Display the decision tree in Streamlit
    st.graphviz_chart(dot_data)



#Titulo del dashboard
st.title('CTIN 2023 Taller de ciencia de datos')

#Definición de la barra izquierda
option = st.sidebar.selectbox('select option',['Datos generales','Análisis exploratorio de datos','Algoritmos de clasificación'])

#Lee los datos desde un archivo CSV
df = read_csv('datasets/healthcare-dataset-stroke-data.csv')

#Cargar información dependiendo de la opción seleccionada por el usuario
if option == 'Datos generales':
    st.subheader('Datos generales')

    image = Image.open('Images/Mild Stroke.jpg')

    st.image(image)

    st.caption('Introducción')

    st.markdown('¿Sabías que los accidentes cerebrovasculares, una de las principales causas de mortalidad en todo el mundo, suelen ser más prevenibles de ' +
            ' lo que pensamos? Los estudios demuestran que hasta el 80% de los accidentes cerebrovasculares se pueden evitar mediante medidas proactivas. En este completo cuaderno, nos embarcamos en un viaje basado en datos para desentrañar la intrincada red de factores que contribuyen a los accidentes cerebrovasculares.')

    st.markdown('Según la Organización Mundial de la Salud (OMS), el accidente cerebrovascular es la segunda causa de muerte a nivel mundial y es responsable de aproximadamente el 11% del total de muertes.')

    st.markdown('En este curso, nos centraremos en un conjunto de datos de más de 5000 pacientes, que incluye una variedad de información clínica y biométrica relevante. A través de algoritmos de Machine Learning, aprenderás a analizar estos datos de manera sistemática, identificar patrones ocultos y, finalmente, clasificar a los pacientes en grupos de riesgo o no riesgo de sufrir accidentes cardiovasculares')
    st.write("Los datos fueron obtenidos de [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data), una plataforma en línea que se especializa en competencias de ciencia de datos y aprendizaje automático."+
             "Ofrece a los científicos de datos, analistas y entusiastas de la inteligencia artificial un entorno donde pueden acceder a conjuntos de datos, participar en desafíos, colaborar en proyectos y mejorar sus habilidades en estas disciplinas.")
    st.markdown('Caracteristicas de los datos:')
    lst = ['age: Esta es la edad del paciente. La edad es un factor crucial en la predicción de un accidente cerebrovascular, ya que el riesgo de sufrir un accidente cerebrovascular aumenta con la edad.', 
           'hypertension: Esta es una característica binaria que indica si el paciente tiene hipertensión (presión arterial alta) o no.', 
           'heart_disease: Esta característica binaria indica si el paciente tiene una enfermedad cardíaca o no.',
           'ever_married: Representa si el paciente está casado o no.',
           'work_type: Esta característica categórica describe el tipo de ocupación del paciente. Ciertas ocupaciones pueden estar asociadas con niveles más altos de estrés o comportamiento sedentario, lo que puede influir en el riesgo de accidente cerebrovascular.',
           'Residence_type: Esta característica indica si el paciente vive en una zona rural o urbana.',
           'avg_glucose_level: Representa el nivel promedio de glucosa en la sangre del paciente.',
           'bmi: Este es el Índice de Masa Corporal del paciente, calculado como el peso en kilogramos dividido por el cuadrado de la altura en metros.',
           'smoking_status: Esta característica categórica indica si el paciente es fumador, exfumador o nunca ha fumado. Fumar puede aumentar el riesgo de accidente cerebrovascular, ya que puede dañar los vasos sanguíneos, aumentar la presión arterial y reducir la cantidad de oxígeno que llega al cerebro.',
           'gender: Esta característica representa el sexo del paciente.']

    s = ''

    for i in lst:
        s += "- " + i + "\n"

    st.markdown(s)
    
    #Despliega los datos en la interfaz
    st.caption('Conjunto de datos')
    st.dataframe(df)

    

if option == 'Análisis exploratorio de datos':
    st.caption('Análisis exploratorio de datos')

    st.subheader('Descripción del dataset obtenido')
    st.dataframe(df.describe())

    st.subheader('Correlación entre las variables del dataset')
    st.dataframe(df.corr())

    st.subheader('Información del conjunto de datos')
    st.markdown(df.info())

    st.subheader('Visualización de los datos')
    st.caption('Agrupamiento de los datos')
    st.markdown('El dataset se puede agrupar de muchas maneras según sus características, en esta sección se pueden visualiar los datos agrupados de diferentes maneras, por sexo, edad, etc.')
    #Se hace un copia del dataset para poder manipularlo libremente durante el EDA
    df2 = df
    #Se reemplazan los valores de la columna de stroke de 0 y 1 a SI y NO
    df2['stroke'] = df2['stroke'].replace({1: 'Yes', 0: 'No'})
    #Contar cuantas personas han tenido un accidente y cuantas no
    stroke_counts = df2['stroke'].value_counts().reset_index()
    stroke_counts.columns = ['Stroke', 'Count']
    st.markdown('Conteo de pacientes que tienen posibilidad de tener un accidente cerebrovascular')
    st.dataframe(stroke_counts)
    #Graficar los resultados
    fig = px.pie(stroke_counts, values='Count', names='Stroke', title='Pacientes que son propensos a un accidente cerebrovasculares')
    st.plotly_chart(fig, theme="streamlit")

    #Contar los valores agrupandolos por la columna de genero

    grouped_data = df.groupby(['gender', 'stroke']).size().reset_index(name='count')
    st.markdown('Pacientes agrupados por genero que pueden o no tener un accidete cerebrovascular')
    st.dataframe(grouped_data)

    #formando la figura de la grafica de los datos agrupados por genero
    fig = px.bar(grouped_data, x='gender', y='count', color='stroke', barmode='group')

    fig.update_layout(
    title='Gráfica de Barras Agrupada por Género y posibilidad de tener un accidente cerebrovascular',
    xaxis_title='Género',
    yaxis_title='Conteo',
    legend_title='Accidente'
    )
    st.plotly_chart(fig, theme="streamlit")

    #Distribucion por edad: se forma una copia del dataset donde solo contenga las columnas de id, edad y si el paciente sufrio o no un accidente
    distribucionEdad = df2[['id', 'age', 'stroke']]
    st.markdown('Distribucón de pacientes por edad')
    st.dataframe(distribucionEdad)
    fig = px.histogram(distribucionEdad, x='age', color='stroke', barmode='overlay')
    fig.update_layout(
    title='Distribución por Edad Agrupada por accident',
    xaxis_title='Edad',
    yaxis_title='Conteo',
    legend_title='Accidente'
    )
    st.plotly_chart(fig, theme="streamlit")


if(option == 'Algoritmos de clasificación'):
    st.caption('Algoritmos de clasificación')

    dtAC = df
    dtAC.fillna(0, inplace=True)

    option1 = st.selectbox(
    'Selecciona un algortimo de clasificación',
    ('Arbol de clasifiación', 'Red neuronal'))

    if option1 == 'Arbol de clasifiación':
        arbolDeClasificacion()
    if option1 == 'Red neuronal':
        perceptronMulticapa()
    