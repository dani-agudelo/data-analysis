import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Procesamiento ETL y variable(s) categórica(s) (a predecir).

RUTA_DATASET = "data/Emergencias.csv"
RUTA_DATASET_LIMPIO = "data/emergencias_limpio.csv"
dataset = pd.read_csv(RUTA_DATASET, sep=",", encoding="utf-8")
# ---------------------------------------------
# A0. Convertir columnas que parecen numéricas pero están como texto
# ---------------------------------------------

for col in dataset.columns:
    # Verificar si el nombre de la columna contiene palabras clave que indican valores numéricos
    if any(
        palabra in col.upper()
        for palabra in [
            "COSTE",
            "PRECIO",
            "GASTO",
            "VALOR",
            "RECURSOS",
            "FIC",
            "TRANSFERENCIAS",
            "RETROEXCAVADORA",
            "CARROTANQUES",
            "OBRAS",
            "SACOS",
            "MATERIALES",
            "ASISTENCIA",
            "SUBSIDIO",
            "APOYO",
            "OTROS",
            "INFRA",
        ]
    ):
        # Primero limpiar los valores
        dataset[col] = dataset[col].astype(str)
        dataset[col] = dataset[col].str.replace(" ", "")
        dataset[col] = dataset[col].str.replace(",", "")
        dataset[col] = dataset[col].str.replace("$", "")

        # Convertir a valores numéricos
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

        # Reemplazar NaN con el promedio de la columna (forma segura)
        if dataset[col].isnull().any():
            dataset[col] = dataset[col].fillna(dataset[col].mean())


# Información básica del dataset con minimo, maximo y media sólo en columnas numéricas
print(dataset.describe())

# print(dataset["OTROS-AFECTACION"].unique())
# print(dataset["DESCRIPCION MATERIALES DE CONSTRUCCION"].unique())

## A1. Aplicación de mapeo con una función de limpieza para las variables categóricas
def limpieza_categoricas(dataset: pd.DataFrame):
    # Primero tomamos las variables categóricas
    dataset_categoricas = dataset.select_dtypes(include=["object"])
    print(dataset_categoricas.columns)

    # Pasamos el contenido de las columnas a mayúsculas
    dataset_categoricas = dataset_categoricas.apply(lambda x: x.str.upper() if x.dtype == "object" else x)


## A2. Aplicación de mapeo con otra función de limpieza para las variables categóricas

limpieza_categoricas(dataset)
print(dataset.columns, 'miremos')


### B1. Determinar valores nulos
# Revisar columna a columna los valores nulos y mostrarlos ordenados ASC
print("Valores nulos en cada columna:")
# for col in dataset.columns:
    # print(f"{col}: {dataset[col].isnull().sum()}")
    # print(f"{col}: {dataset[col].isnull().sum()}")

print(dataset.isnull().sum().sort_values(ascending=False))

print(f"Cantidad total de filas: {dataset.shape[0]}")

COL_OTROS_AFECTACION = 'OTROS-AFECTACION'
COL_DESCRIPCION_MATERIALES = 'DESCRIPCION MATERIALES DE CONSTRUCCION'


# Porcentaje de NAN en las columnas mencionadas: Si pasa del 1% se eliminará la columna

prob_drop_otros = dataset[COL_OTROS_AFECTACION].isnull().sum() / dataset.shape[0]
prob_drop_desc = dataset[COL_DESCRIPCION_MATERIALES].isnull().sum() / dataset.shape[0]


print(f"Probabilidad de eliminar {COL_OTROS_AFECTACION}: {prob_drop_otros}")
print(f"Probabilidad de eliminar {COL_DESCRIPCION_MATERIALES}: {prob_drop_desc}")


# Eliminar columnas con menos del 5% de valores no nulos #

columnas_a_eliminar = []

if prob_drop_otros > 0.95:
    columnas_a_eliminar.append(COL_OTROS_AFECTACION)
if prob_drop_desc > 0.95:
    columnas_a_eliminar.append(COL_DESCRIPCION_MATERIALES)

# Eliminar solo si hay columnas para eliminar
dataset_sin_nans = dataset.drop(columns=columnas_a_eliminar)

### B2. Eliminación de valores atípicos (outliers) o su normalización

# Normalización de tejas fibrocemento (variable categórica) con una media de 0 y desviación estándar de 1

# 1. Identificar columnas categóricas
columnas_categoricas = dataset.select_dtypes(include=["object"]).columns

# 2. Método 1: Usar Label Encoding (asigna un número a cada categoría)
# Útil para calcular media/moda después
# le = LabelEncoder()
# for col in columnas_categoricas:
#     # Primero rellenar NaN con la moda
#     moda = columnas_categoricas[col].mode()[0]
#     columnas_categoricas[col] = columnas_categoricas[col].fillna(moda)

#     # Convertir a numérico con LabelEncoder
#     columnas_categoricas[f"{col}_encoded"] = le.fit_transform(columnas_categoricas[col])

#     # Ahora puedes calcular estadísticas sobre la versión codificada
#     print(f"Media de {col}: {columnas_categoricas[f'{col}_encoded'].mean()}")
#     print(f"Moda de {col}: {columnas_categoricas[f'{col}_encoded'].mode()[0]}")


# for col in dataset.columns:
#     print(f"Valores únicos en {col}: {dataset[col].unique()}")


### B1. Determinar valores nulos y rellenarlos


# C1. Exportar el dataset limpio a un archivo CSV
dataset_sin_nans.to_csv(
    RUTA_DATASET_LIMPIO,
    index=False,
)


## DIVIPOLA

# Recorremos el dataset y mergeamos las columnas DIVIPOLA con Código, Longitud y Latitud del archivo codigo_municipios de excel

# Cargar el archivo de código de municipios
df_municipios = pd.read_excel("data/codigo_municipios.xlsx")

# Asegurar que las columnas tengan el mismo tipo
dataset_sin_nans["DIVIPOLA"] = dataset_sin_nans["DIVIPOLA"].astype(str)
df_municipios["Código"] = df_municipios["Código"].astype(str)

# Hacer merge entre DIVIPOLA (dataset) y Código (df_municipios), añadiendo solo Latitud y Longitud
dataset_sin_nans = pd.merge(
    dataset_sin_nans,
    df_municipios[["Código", "Latitud", "Longitud"]],
    left_on="DIVIPOLA",
    right_on="Código",
    how="left"
)

# (Opcional) Eliminar la columna "Código" duplicada después del merge si ya no la necesitas
dataset_sin_nans.drop(columns=["Código"], inplace=True)

# Guardar el dataset_sin_nans actualizado
dataset_sin_nans.to_csv(RUTA_DATASET_LIMPIO, index=False)
print(f"Dataset actualizado guardado en {RUTA_DATASET_LIMPIO}")
# print(dataset_sin_nans.head())
# print(dataset_sin_nans.describe())
# print(dataset_sin_nans.info())


