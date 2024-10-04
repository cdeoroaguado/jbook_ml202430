import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris

# Cargar el dataset Iris
iris_data = load_iris()
iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

# Título del Dashboard
st.title("Dashboard Interactivo: Dataset Iris con Plotly")

# Mostrar los primeros datos
st.write("### Vista previa de los datos")
st.dataframe(iris.head())

# Descripción de los datos
st.write("### Estadísticas descriptivas")
st.write(iris.describe())

# Gráfico de dispersión interactivo con Plotly
st.write("### Gráfico de dispersión por especies")
x_axis = st.selectbox("Selecciona la característica del eje X", iris_data.feature_names)
y_axis = st.selectbox("Selecciona la característica del eje Y", iris_data.feature_names)

# Crear gráfico de dispersión interactivo
scatter_plot = px.scatter(iris, x=x_axis, y=y_axis, color='species',
                          title=f'Dispersión entre {x_axis} y {y_axis}',
                          labels={x_axis: x_axis, y_axis: y_axis, 'species': 'Especies'},
                          template='plotly_dark')

st.plotly_chart(scatter_plot)

# Histograma interactivo con Plotly
st.write("### Histograma de una característica seleccionada")
selected_feature = st.selectbox("Selecciona la característica para el histograma", iris_data.feature_names)

histogram = px.histogram(iris, x=selected_feature, color='species',
                         title=f'Histograma de {selected_feature} por especies',
                         labels={selected_feature: selected_feature, 'species': 'Especies'},
                         nbins=20, template='plotly_dark')

st.plotly_chart(histogram)

# Mapa de calor de la correlación usando Plotly
st.write("### Mapa de calor de la correlación entre las características")

# Excluir la columna 'species' ya que es categórica
iris_numeric = iris.drop(columns=['species'])
correlation = iris_numeric.corr()

heatmap = px.imshow(correlation, text_auto=True, title="Mapa de calor de correlación",
                    labels={'color': 'Correlación'}, template='plotly_dark')

st.plotly_chart(heatmap)
