import pandas as pd
import os
import requests

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from datetime import datetime

# 1. Carga del dataset - Exploración de datos y validación
### Se usa ruta dinámica para lectura del archivo raíz, para que funcione en cualquier PC
base_path = os.path.dirname(os.path.dirname(__file__))  # Subir de /src a raíz del proyecto
file_path = os.path.join(
    base_path,
    "data",
    "processed",
    "dataset_model_ready.csv"
)
df = pd.read_csv(file_path, low_memory=False)

# 2. Definir X y y
X = df.drop(columns=["HAS_TROUBLE_CODE"])
y = df["HAS_TROUBLE_CODE"]

# 3. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Entrenar el modelo RANDOM FOREST
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced"  # manejo de desbalance
)

model.fit(X_train, y_train)

# 5. Predicciones
y_pred = model.predict(X_test)


# 5.1. Probabilidades y Elasticsearch
y_prob = model.predict_proba(X_test)

url = "http://localhost:9200/obd_predictions/_doc"

limite_envio = min(10000, len(y_pred))

#for i in range(len(y_pred)):
for i in range(limite_envio):
    real = int(y_test.iloc[i])
    pred = int(y_pred[i])
    prob = float(max(y_prob[i]))

    # Clasificación tipo matriz de confusión
    if real == 1 and pred == 1:
        tipo = "TP"
    elif real == 0 and pred == 0:
        tipo = "TN"
    elif real == 0 and pred == 1:
        tipo = "FP"
    else:
        tipo = "FN"

    documento = {
        "real": real,
        "prediccion": pred,
        "resultado": "correcto" if real == pred else "error",
        "tipo_error": tipo,
        "probabilidad": prob,
        "modelo": "RandomForest",
        "timestamp": datetime.now().isoformat()
    }

    #requests.post(url, json=documento)
    requests.post(url, json=documento, timeout=10)

#print("\nResultados enriquecidos enviados a Elasticsearch.")
print(f"\nResultados enriquecidos enviados a Elasticsearch: {limite_envio}")

"""
# 5.1. Enviar resultados a ELASTICSEARCH
url = "http://localhost:9200/obd_predictions/_doc"

for i in range(len(y_pred)):
    documento = {
        "real": int(y_test.iloc[i]),
        "prediccion": int(y_pred[i]),
        "resultado": "correcto" if int(y_test.iloc[i]) == int(y_pred[i]) else "error"
    }

    response = requests.post(url, json=documento)

print("\nResultados enviados a Elasticsearch.")
"""

# 6. Evaluación - Métricas
print("\n=== RESULTADOS RANDOM FOREST ===")

print("Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))


# 7. Importancia de valores para el modelo
importancias = pd.Series(model.feature_importances_, index=X.columns)
importancias = importancias.sort_values(ascending=False)

print("\n=== IMPORTANCIA DE VARIABLES RELIEF ALGORITHM ===")
print(importancias)









