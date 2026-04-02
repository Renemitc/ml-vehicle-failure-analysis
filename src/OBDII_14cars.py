#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Data exploration and validation
df=pd.read_csv("/data/raw/exp1_14drivers_14cars_dailyRoutes.csv", low_memory=False)
print(df.head())
print(df.shape)    # Ver tamaño
print(df.columns)  # Ver columnas
print(df.info())   # Ver estructura general
print(df.describe(include="all"))   # Estadísticas básicas
print(df.isnull().sum())    # Revisar valores nulos

# Data cleaning
df = df.drop_duplicates()   # Eliminar duplicados
print(df.isnull().sum())    # Revisar valores nulos
columnas_utiles = [         # Análisis inicial
    "ENGINE_COOLANT_TEMP",
    "ENGINE_LOAD",
    "AMBIENT_AIR_TEMP",
    "ENGINE_RPM",
    "INTAKE_MANIFOLD_PRESSURE",
    "MAF",
    "AIR_INTAKE_TEMP",
    "FUEL_PRESSURE",
    "SPEED",
    "THROTTLE_POS",
    "DTC_NUMBER",
    "TIMING_ADVANCE",
    "EQUIV_RATIO"
]

#df_limpio = df[columnas_utiles].copy()
df_limpio = df.copy()

# Sección de corrección de valores columnas en particular

#df_limpio["ENGINE_LOAD"] = df_limpio["ENGINE_LOAD"].astype(str)   # Limpiar columna
#df_limpio["ENGINE_LOAD"] = df_limpio["ENGINE_LOAD"].str.replace("%", "", regex=False)    # Quitar %
#df_limpio["ENGINE_LOAD"] = df_limpio["ENGINE_LOAD"].str.replace(",", ".", regex=False)   # Cambiar coma por punto
#df_limpio["ENGINE_LOAD"] = pd.to_numeric(df_limpio["ENGINE_LOAD"], errors="coerce")    # Convertir a número
#df_limpio["ENGINE_LOAD"] = df_limpio["ENGINE_LOAD"] / 100     # Sintaxis para dejarlo en proporción
#print(df_limpio["ENGINE_LOAD"].head())    # Validación de salida
#print(df_limpio["ENGINE_LOAD"].isnull().sum())

def limpiar_porcentaje(col):
    return (
        col.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

df_limpio["ENGINE_LOAD"] = limpiar_porcentaje(df_limpio["ENGINE_LOAD"])
df_limpio["ENGINE_LOAD"] = pd.to_numeric(df_limpio["ENGINE_LOAD"], errors="coerce")

for col in columnas_utiles:     # Convertir a numéricas las columnas importantes
    df_limpio[col] = pd.to_numeric(df_limpio[col], errors="coerce")

df_limpio = df_limpio.ffill()  # Relleno hacia adelante
df_limpio = df_limpio.bfill()  # Relleno hacia atrás

#for col in columnas_utiles:    # Opcional: Relleno lo que quede con media
#    df_limpio[col].fillna(df_limpio[col].mean(), inplace=True)

print("Shape original:", df.shape)

print("Nulos finales:")    # Validación
print(df_limpio.isnull().sum())

print("Shape final:", df_limpio.shape)

# Convertir el DF limpio a un archivo CSV
df_limpio.to_csv("data/processed/exp1_14drivers_14cars_dailyRoutes_clean.csv", index=False)

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