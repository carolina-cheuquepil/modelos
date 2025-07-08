#I Hacer el modelo en Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pickle
import os

# 1. Cargar archivo
df = pd.read_csv("C:/Users/Carolina/Downloads/datos_limpios_con_derivadas.csv")

# 2. Crear nuevas variables necesarias para el modelo
df['Kills_por_minuto'] = df['RoundKills'] / (df['TimeAlive'] + 1)
df['Headshot_rate'] = df['RoundHeadshots'] / (df['RoundKills'] + 1)
df['Tuvo_asistencia'] = (df['RoundAssists'] > 0).astype(int)
df['Equipamiento_total'] = df['RoundStartingEquipmentValue'] + df['TeamStartingEquipmentValue']

# 3. Seleccionar las 9 variables
features = [
    'TimeAlive',
    'TravelledDistance',
    'FirstKillTime',
    'RoundStartingEquipmentValue',
    'TeamStartingEquipmentValue',
    'Kills_por_minuto',
    'Headshot_rate',
    'Tuvo_asistencia',
    'Equipamiento_total'
]
X = df[features]
y = df['Survived']

# 4. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=666)

# 5. Entrenar el modelo
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# 6. Evaluar y mostrar AUC
y_proba = modelo.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print("✅ AUC del nuevo modelo:", round(auc, 4))

# 7. Guardar modelo entrenado
os.makedirs("checkpoints", exist_ok=True)
with open("checkpoints/model.pkl", "wb") as f:
    pickle.dump(modelo, f)

