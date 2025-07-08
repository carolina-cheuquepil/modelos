#II Modelo de predicción de supervivencia en CS:GO
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import pickle
import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# Cargar modelo entrenado
modelo = pickle.load(open("checkpoints/model.pkl", "rb"))

# Crear la app
app = FastAPI()

# Configurar carpeta de templates
templates = Jinja2Templates(directory="templates")

# Ruta para formulario
@app.get("/", response_class=HTMLResponse)
async def mostrar_formulario(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Ruta para procesar la predicción
@app.post("/resultado", response_class=HTMLResponse)
async def procesar_formulario(
    request: Request,
    TimeAlive: float = Form(...),
    TravelledDistance: float = Form(...),
    FirstKillTime: float = Form(...),
    RoundStartingEquipmentValue: float = Form(...),
    TeamStartingEquipmentValue: float = Form(...),
    Kills_por_minuto: float = Form(...),
    Headshot_rate: float = Form(...),
    Tuvo_asistencia: int = Form(...),
    Equipamiento_total: float = Form(...)
):
    # Crear arreglo con las variables
    entrada = np.array([[TimeAlive, TravelledDistance, FirstKillTime,
                         RoundStartingEquipmentValue, TeamStartingEquipmentValue,
                         Kills_por_minuto, Headshot_rate, Tuvo_asistencia,
                         Equipamiento_total]])

    # Hacer predicción
    pred = modelo.predict(entrada)[0]
    prob = modelo.predict_proba(entrada)[0][1]  # prob de sobrevivir

    # Convertir a texto
    resultado = "🟢 Sobrevive" if pred == 1 else "🔴 No sobrevive"
    probabilidad = round(prob * 100, 2)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "resultado": resultado,
        "probabilidad": probabilidad
    })

