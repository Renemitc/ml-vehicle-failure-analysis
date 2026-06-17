import pandas as pd
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# 1. Cargar dataset procesado
### Se usa ruta dinámica para lectura del archivo raíz, para que funcione en cualquier PC
base_path = os.path.dirname(os.path.dirname(__file__))  # Subir de /src a raíz del proyecto
file_path = os.path.join(
    base_path,
    "data",
    "processed",
    "dataset_model_ready.csv"
)
df = pd.read_csv(file_path, low_memory=False)

print("Shape dataset model-ready:", df.shape)
print("\nDistribución original de HAS_TROUBLE_CODE:")
print(df["HAS_TROUBLE_CODE"].value_counts())
print("\nPorcentaje original:")
print(df["HAS_TROUBLE_CODE"].value_counts(normalize=True) * 100)

# 2. Separar variables X e y
X = df.drop(columns=["HAS_TROUBLE_CODE"])
y = df["HAS_TROUBLE_CODE"]


# 3. Separar entrenamiento, validación y prueba
# 70% entrenamiento, 15% validación, 15% prueba
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDistribución antes de SMOTE:")
print("Entrenamiento:")
print(y_train.value_counts().sort_index())

print("\nValidación:")
print(y_val.value_counts().sort_index())

print("\nPrueba:")
print(y_test.value_counts().sort_index())


# 4. Aplicar SMOTE SOLO al entrenamiento
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nDistribución después de SMOTE en entrenamiento:")
print(y_train_smote.value_counts().sort_index())

print("\nShapes finales:")
print("X_train original:", X_train.shape)
print("X_train SMOTE:", X_train_smote.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)