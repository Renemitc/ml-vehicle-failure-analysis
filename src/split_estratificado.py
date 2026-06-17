import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

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


print("Shape dataset completo:", df.shape)
print("\nDistribución original:")
print(df["HAS_TROUBLE_CODE"].value_counts().sort_index())
print("\nPorcentaje original:")
print((df["HAS_TROUBLE_CODE"].value_counts(normalize=True).sort_index() * 100).round(2))


# 2. Separar variables predictoras X y variable objetivo y
target = "HAS_TROUBLE_CODE"

X = df.drop(columns=[target])
y = df[target].astype(int)


# 3. Primera partición: 70% entrenamiento y 30% temporal
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 4. Segunda partición: dividir el 30% temporal en 15% y 15%
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# 5. Función para mostrar distribución por conjunto
def resumen_clases(nombre, y_data):
    conteo = y_data.value_counts().sort_index()
    total = len(y_data)

    clase_0 = conteo.get(0, 0)
    clase_1 = conteo.get(1, 0)

    return {
        "Conjunto": nombre,
        "Clase 0": clase_0,
        "Clase 1": clase_1,
        "Total": total,
        "% Clase 0": round((clase_0 / total) * 100, 2),
        "% Clase 1": round((clase_1 / total) * 100, 2)
    }


resumen = pd.DataFrame([
    resumen_clases("Entrenamiento", y_train),
    resumen_clases("Validación", y_val),
    resumen_clases("Prueba", y_test)
])

print("\nDistribución por conjunto:")
print(resumen)


# 6. Guardar los conjuntos si deseas usarlos después
output_path = os.path.join(base_path, "data", "processed", "splits")
os.makedirs(output_path, exist_ok=True)    # Crear carpeta processed si no existe

train_df = X_train.copy()
train_df[target] = y_train

val_df = X_val.copy()
val_df[target] = y_val

test_df = X_test.copy()
test_df[target] = y_test

# 7. Guardar cada conjunto por separado
train_df.to_csv(os.path.join(output_path, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_path, "validation.csv"), index=False)
test_df.to_csv(os.path.join(output_path, "test.csv"), index=False)

# 8. Guardar también el resumen de la partición
resumen.to_csv(os.path.join(output_path, "resumen_particion_estratificada.csv"), index=False)

print("\nArchivos guardados correctamente en:")
print(output_path)

print("\nArchivos generados:")
print("- train.csv")
print("- validation.csv")
print("- test.csv")
print("- resumen_particion_estratificada.csv")

#output_path = base_path / "data" / "processed" / "splits"
#output_path.mkdir(parents=True, exist_ok=True)

#file_output = os.path.join(    # Ruta completa y nombre del archivo a guardar
#    output_path,
#    "train.csv",
#    "validation.csv",
#    "test.csv"
#)

#resumen.to_csv(file_output, index=False)
#print("\nArchivo guardado en:", file_output)   # Con este codigo confirmo

#print("\nArchivos guardados en:")
#print(output_path)
