#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re

# 1. Carga - Exploración de datos y validación
# Se usa ruta dinámica para lectura del archivo raíz, para que funcione en cualquier PC
base_path = os.path.dirname(os.path.dirname(__file__))  # Subir de /src a raíz del proyecto
file_path = os.path.join(
    base_path,
    "data",
    "raw",
    "exp1_14drivers_14cars_dailyRoutes.csv"
)

df = pd.read_csv(file_path, low_memory=False)

print("Shape original:", df.shape)
print("Nulos iniciales:")
print(df.isnull().sum())    # Revisar valores nulos

# print(df.head())
# print(df.shape)    # Ver tamaño
# print(df.columns)  # Ver columnas
# print(df.info())   # Ver estructura general
# print(df.describe(include="all"))   # Estadísticas básicas
# print(df.isnull().sum())    # Revisar valores nulos

# 2. Limpieza básica de datos
df = df.drop_duplicates().copy()   # Eliminar duplicados


# 3. Funciones auxiliares para normalización

# Limpiar y cambiar porcentajes a decimales de columnas (estos datos son valores tipo string, usan "%" y ",")
def limpiar_porcentaje(col):
    return (
        col.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )


def extraer_mil_on(valor):
    """
    Convierte textos como:
    'MIL is OFF0 codes' -> 0
    'MIL is ON0 codes'  -> 1
    """
    if pd.isna(valor):
        return pd.NA
    valor = str(valor).upper()
    if "MIL IS ON" in valor:
        return 1
    if "MIL IS OFF" in valor:
        return 0
    return pd.NA

def extraer_dtc_count(valor):
    """
    Extrae el número de códigos desde strings como:
    'MIL is OFF0 codes' -> 0
    'MIL is OFF1 codes' -> 1
    'MIL is OFF107 codes' -> 107
    """
    if pd.isna(valor):
        return pd.NA
    match = re.search(r'(\d+)\s*codes', str(valor), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return pd.NA

def extraer_codigos_dtc(valor):
    """
    Extrae códigos DTC en formato tipo:
    P0133, C0300, U1004, B0004
    incluso si vienen concatenados:
    P0079P2004P3000
    """
    if pd.isna(valor):
        return []
    return re.findall(r'[PCBU][0-3][0-9A-F]{3}', str(valor).upper())


# 4. Se realiza copia de trabajo
df_limpio = df.copy()

# 5. Se limpia columnas tipo porcentaje
columnas_porcentaje = [
    "ENGINE_LOAD",
    "THROTTLE_POS",
    "FUEL_LEVEL",
    "TIMING_ADVANCE",
    "EQUIV_RATIO"
]

for col in columnas_porcentaje:
    df_limpio[col] = limpiar_porcentaje(df_limpio[col])
    df_limpio[col] = pd.to_numeric(df_limpio[col], errors="coerce")
    df_limpio[col] = df_limpio[col] / 100


# 6. Convertir columnas numéricas OBD-II
columnas_numericas = [
    "ENGINE_COOLANT_TEMP",
    "AMBIENT_AIR_TEMP",
    "ENGINE_RPM",
    "INTAKE_MANIFOLD_PRESSURE",
    "MAF",
    "AIR_INTAKE_TEMP",
    "FUEL_PRESSURE",
    "SPEED",
    "LONG TERM FUEL TRIM BANK 2",
    "SHORT TERM FUEL TRIM BANK 2",
    "SHORT TERM FUEL TRIM BANK 1",
    "ENGINE_RUNTIME",
    "BAROMETRIC_PRESSURE(KPA)",
    "MIN",
    "HOURS",
    "DAYS_OF_WEEK",
    "MONTHS",
    "YEAR"
]

for col in columnas_numericas:     # Convertir a numéricas las columnas importantes
    df_limpio[col] = pd.to_numeric(df_limpio[col], errors="coerce")


# 7. Ingeniería de features DTC
# DTC_NUMBER -> features interpretables
df_limpio["MIL_ON"] = df_limpio["DTC_NUMBER"].apply(extraer_mil_on)
df_limpio["DTC_COUNT"] = df_limpio["DTC_NUMBER"].apply(extraer_dtc_count)

# TROUBLE_CODES -> lista de códigos
df_limpio["DTC_LIST"] = df_limpio["TROUBLE_CODES"].apply(extraer_codigos_dtc)

# Features generales
df_limpio["HAS_TROUBLE_CODE"] = df_limpio["DTC_LIST"].apply(lambda x: 1 if len(x) > 0 else 0)
df_limpio["TROUBLE_CODE_COUNT"] = df_limpio["DTC_LIST"].apply(len)

# Crear variables binarias para los códigos más frecuentes
todos_los_codigos = (
    df_limpio["DTC_LIST"]
    .explode()
    .dropna()
)

top_codigos = todos_los_codigos.value_counts().head(10).index.tolist()

for codigo in top_codigos:
    nueva_col = f"TC_{codigo}"
    df_limpio[nueva_col] = df_limpio["DTC_LIST"].apply(lambda lista: 1 if codigo in lista else 0)


# 8. Relleno solo en sensores numéricos
columnas_sensores_modelo = columnas_porcentaje + columnas_numericas
df_limpio[columnas_sensores_modelo] = df_limpio[columnas_sensores_modelo].ffill().bfill()

# 9. Realizo validación general
print("\n=== VALIDACIÓN DE PREPROCESAMIENTO ===")
print("Registros totales:", df_limpio.shape[0])
print("Columnas totales:", df_limpio.shape[1])
print("Valores nulos restantes:", df_limpio.isnull().sum().sum())

print("\nNulos por columna:")
print(df_limpio.isnull().sum())

print("\nTop códigos DTC detectados:")
print(todos_los_codigos.value_counts().head(10))

# 10. EDA del TARGET (Análisis exploratorio de datos)
print("\n=== DISTRIBUCIÓN TARGET ===")
print(df_limpio["HAS_TROUBLE_CODE"].value_counts())

# Definiendo explícitamente "X" y "y"
print("\nProporción:")
print(df_limpio["HAS_TROUBLE_CODE"].value_counts(normalize=True))

# Variables de entrada (solo sensores OBD-II)
columnas_input_modelo = columnas_porcentaje + columnas_numericas
X = df_limpio[columnas_input_modelo].copy()


# Validar nulos solo dentro de X
print("\n=== NULOS EN X ===")
print(X.isnull().sum())

print("\n=== COLUMNAS 100% NULAS EN X ===")
print(X.columns[X.isnull().all()].tolist())

# Eliminar columnas completamente vacías en X
X = X.dropna(axis=1, how="all")

print("\n=== SHAPE X DESPUÉS DE ELIMINAR COLUMNAS 100% NULAS ===")
print(X.shape)

print("\n=== COLUMNAS FINALES DE X ===")
print(X.columns.tolist())



# Variable objetivo TARGET
y = df_limpio["HAS_TROUBLE_CODE"].copy()

print("\n=== VALIDACIÓN DE MODELADO ===")
print("Shape X:", X.shape)
print("Shape y:", y.shape)
print("\nColumnas de X:")
print(X.columns.tolist())

print("\nDistribución del target:")
print(y.value_counts())

print("\nProporción del target:")
print(y.value_counts(normalize=True))

# Bloque de columnas final para X
### 1. Definir columnas finales (solo sensores reales)
columnas_finales = [
    #"ENGINE_LOAD",
    "THROTTLE_POS", #ok
    "FUEL_LEVEL", #ok
    "TIMING_ADVANCE", #ok
    "ENGINE_COOLANT_TEMP", #ok
    #"AMBIENT_AIR_TEMP",
    #"ENGINE_RPM",
    "INTAKE_MANIFOLD_PRESSURE", #ok
    "MAF", #ok
    "AIR_INTAKE_TEMP", #ok
    #"SPEED,
    "BAROMETRIC_PRESSURE(KPA)" #ok
]

### 2. Crear X_final
X_final = df_limpio[columnas_finales].copy()

### 3. Validar
print("\n=== X FINAL ===")
print("Shape:", X_final.shape)
print("Columnas:", X_final.columns.tolist())

### 4. Realizar correlación con TARGET
print("\n=== CORRELACIÓN CON TARGET ===")
correlaciones = X_final.copy()
correlaciones["TARGET"] = y
corr_matrix = correlaciones.corr()

print(corr_matrix["TARGET"].sort_values(ascending=False))

print("\n=== VARIANZA ===")
print(X_final.var().sort_values())

# ================================
# LAB III — PREPARACIÓN FINAL
# ================================
# Variables finales
X = X_final.copy()
y = df_limpio["HAS_TROUBLE_CODE"].copy()

print("\n=== DATASET FINAL PARA LAB III ===")
print("Shape X:", X.shape)
print("Shape y:", y.shape)

# ================================
# LAB III — FEATURE SELECTION
# ================================

# Crear dataframe para correlación con target
df_fs = X.copy()
df_fs["TARGET"] = y

corr_matrix = df_fs.corr()

# Ordenar por importancia (valor absoluto)
corr_target = corr_matrix["TARGET"].drop("TARGET").abs().sort_values(ascending=False)

print("\n=== IMPORTANCIA POR CORRELACIÓN ===")
print(corr_target)


# Selección automática por umbral
umbral = 0.05

columnas_seleccionadas = corr_target[corr_target > umbral].index.tolist()

print("\nColumnas seleccionadas:")
print(columnas_seleccionadas)

# Nuevo dataset reducido
X_selected = X[columnas_seleccionadas].copy()

print("\nShape después de selección:", X_selected.shape)


# ================================
# DATASET FINAL PARA MODELADO
# ================================
X = X_selected.copy()
#df_model = X_selected.copy()

print("\n=== DATASET FINAL PARA LAB III ===")
print("Shape X final:", X.shape)
print("Columnas finales:", X.columns.tolist())

df_model = X.copy()
df_model["HAS_TROUBLE_CODE"] = y

# 11. Se construye la ruta de la carpeta donde se guarda el archivo procesado

output_path = os.path.join(base_path, "data", "processed")
os.makedirs(output_path, exist_ok=True)    # Crear carpeta processed si no existe

file_output = os.path.join(    # Ruta completa y nombre del archivo a guardar
    output_path,
    "dataset_model_ready.csv"
)

df_model.to_csv(file_output, index=False)
print("\nArchivo guardado en:", file_output)   # Con este codigo confirmo








"""
# Data splitting
from sklearn.model_selection import train_test_split   # Dejar el split como preparación general

train_data, test_data = train_test_split(
    df_limpio,
    test_size=0.2,
    random_state=42
)

print(train_data.shape)
print(test_data.shape)
"""