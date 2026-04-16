import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

print("\n=== IMPORTANCIA DE VARIABLES RELIEF ALGORITHMS ===")
print(importancias)









