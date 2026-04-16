import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree

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
X = df.drop("HAS_TROUBLE_CODE", axis=1)
y = df["HAS_TROUBLE_CODE"]
print("\n=== DATASET PARA MODELO ===")
print("Shape X:", X.shape)
print("Shape y:", y.shape)


# 3. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTrain:", X_train.shape)
print("Test:", X_test.shape)

# 4. Entrenar el modelo
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)


# 5. Predicciones
y_pred = model.predict(X_test)


# 6. Evaluación - Métricas
print("\n=== RESULTADOS DECISION TREE ===")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))


# 8. Visualización del árbol
plt.figure(figsize=(20,10))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Falla", "Falla"],
    filled=True
)
plt.title("Árbol de decisión - Predicción de fallas")
plt.show()










