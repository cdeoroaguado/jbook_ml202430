import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Control deslizante en la barra lateral
num_points = st.sidebar.slider("Número de puntos", 100, 1000, 500)

# Generar datos aleatorios
data = np.random.randn(num_points)

# Crear la figura con dos gráficos: diagrama de cajas y bigotes y un histograma
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Gráfico 1: Diagrama de cajas y bigotes
ax1.boxplot(data)
ax1.set_title("Diagrama de Cajas y Bigotes")
ax1.set_ylabel("Valores")

# Gráfico 2: Histograma
ax2.hist(data, bins=30, color="skyblue", edgecolor="black")
ax2.set_title("Histograma")
ax2.set_xlabel("Valor")
ax2.set_ylabel("Frecuencia")

# Ajustar el espaciado entre gráficos
plt.tight_layout()

# Mostrar la figura en Streamlit
st.pyplot(fig)

