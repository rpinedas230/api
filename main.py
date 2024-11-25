from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import dill
import numpy as np
import os
import csv
from datetime import datetime

# Ruta base de los archivos en la carpeta "app"
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Ruta del archivo de log
LOG_FILE = os.path.join(BASE_PATH, "api_logs.csv")

# Inicializa el archivo de log si no existe
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "claim_id", "marca_vehiculo", "antiguedad_vehiculo",
                         "tipo_poliza", "taller", "partes_a_reparar", "partes_a_reemplazar", "prediction"])

def log_request(data, prediction):
    """
    Registra cada consulta a la API en un archivo CSV.
    """
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().isoformat(),
            data["claim_id"],
            data["marca_vehiculo"],
            data["antiguedad_vehiculo"],
            data["tipo_poliza"],
            data["taller"],
            data["partes_a_reparar"],
            data["partes_a_reemplazar"],
            prediction
        ])

# Iniciar la API
app = FastAPI()

# Diccionario de imputación
imputation_dict = {
    'log_total_piezas': 1.4545,
    'marca_vehiculo_encoded': 0,
    'valor_vehiculo': 3560,
    'valor_por_pieza': 150,
    'antiguedad_vehiculo': 1,
    'tipo_poliza': 1,
    'taller': 1,
    'partes_a_reparar': 3,
    'partes_a_reemplazar': 1,
}

# Clase para la entrada de la API
class ClaimRequest(BaseModel):
    claim_id: int
    marca_vehiculo: str
    antiguedad_vehiculo: int
    tipo_poliza: int
    taller: int
    partes_a_reparar: int
    partes_a_reemplazar: int

# Ruta base de los archivos carpeta "app"
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Función para cargar pipelines
def load_pipeline(filepath):
    with open(filepath, "rb") as f:
        return dill.load(f)

# Carga de los pipelines en el orden especificado, debido a las dependencias
pipeline_1 = load_pipeline(os.path.join(BASE_PATH, "pipeline_1.pkl"))
pipeline_2 = load_pipeline(os.path.join(BASE_PATH, "pipeline_2.pkl"))
pipeline_3 = load_pipeline(os.path.join(BASE_PATH, "pipeline_3.pkl"))
pipeline_4 = load_pipeline(os.path.join(BASE_PATH, "pipeline_4.pkl"))
pipeline_5 = load_pipeline(os.path.join(BASE_PATH, "pipeline_5.pkl"))

# Carga del modelo
def load_model(filepath):
    with open(filepath, "rb") as f:
        return dill.load(f)

model = load_model(os.path.join(BASE_PATH, "linnear_regression.pkl"))

# Preprocesamiento
def preprocess_data(df):
    """
    Procesa el DataFrame en el orden correcto según las dependencias de los pipelines.
    """
    try:
        # Imputación de valores nulos inicial
        df.fillna(imputation_dict, inplace=True)

        # Ejecutar pipelines como funciones
        df = pipeline_1(df)  # pipeline 1 es independiente
        df = pipeline_2(df)  # pipeline 2 depende de pipeline 1
        df = pipeline_3(df)  # pipeline 3 es independiente
        df = pipeline_4(df)  # pipeline 4 depende de pipeline 2
        df = pipeline_5(df)  # pipeline 5 depende de pipeline 3

        # Imputación de valores nulos después de los pipelines
        df.fillna(imputation_dict, inplace=True)

        return df
    except Exception as e:
        print(f"Error en preprocess_data: {e}")
        raise HTTPException(status_code=500, detail=f"Error en el preprocesamiento: {e}")

# Predicción
def predict(model, data):
    """
    Realiza la predicción utilizando el modelo cargado.
    """
    try:
        # Seleccionar las columnas necesarias
        features = [
            'log_total_piezas',
            'marca_vehiculo_encoded',
            'valor_vehiculo',
            'valor_por_pieza',
            'antiguedad_vehiculo'
        ]

        # Verificar si hay valores NaN en las columnas 
        if data[features].isnull().any().any():
            raise ValueError("Los datos contienen valores NaN después del preprocesamiento.")

        prediction = model.predict(data[features])
        return np.round(prediction[0], 2)
    except Exception as e:
        print(f"Error en predict: {e}")
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {e}")

@app.post("/predict/")
async def predict_claim(request: ClaimRequest):
    try:
        # Convertir entrada a DataFrame
        data = pd.DataFrame([request.dict()])

        # Preprocesar 
        processed_data = preprocess_data(data)

        # Condición especial para tipo de póliza
        if processed_data["tipo_poliza"].iloc[0] == 4:
            log_request(request.dict(), -1)  # Log para tipo de póliza 4
            return {"claim_id": request.claim_id, "prediction": -1}

        # Realizar predicción
        prediction = predict(model, processed_data)

        # Log de la consulta
        log_request(request.dict(), prediction)

        return {"claim_id": request.claim_id, "prediction": prediction}
    except Exception as e:
        print(f"Error en el endpoint /predict/: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")

# Ejecución del servidor (solo funciona en local)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
