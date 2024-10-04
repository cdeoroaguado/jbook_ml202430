from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Crear la aplicación FastAPI
app = FastAPI()

# Definir la estructura de la fila de entrada usando Pydantic
class InputData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity_1: int  # Variable dummy 1
    ocean_proximity_2: int  # Variable dummy 2
    ocean_proximity_3: int  # Variable dummy 3

# Cargar el modelo guardado (Pickle)
with open('modelo_lasso.pkl', 'rb') as archivo_pickle:
    modelo_lasso = pickle.load(archivo_pickle)

# Definir la ruta POST para recibir la fila y devolver la predicción
@app.post("/predict/")
async def predict(data: InputData):
    # Convertir la entrada en un array numpy
    fila = np.array([[data.longitude, 
                      data.latitude,
                      data.housing_median_age,
                      data.total_rooms,
                      data.total_bedrooms,
                      data.population,
                      data.households,
                      data.median_income,
                      data.ocean_proximity_1,
                      data.ocean_proximity_2,
                      data.ocean_proximity_3]])
    
    # Realizar la predicción
    prediccion = modelo_lasso.predict(fila)
    
    # Devolver la predicción
    return {"prediccion": prediccion[0]}
