import pandas as pd

# Procesamiento ETL y variable(s) categórica(s) (a predecir).

RUTA_DATASET = "data/emergencias.csv"
RUTA_DATASET_LIMPIO = "data/emergencias_limpio.csv"
dataset = pd.read_csv(RUTA_DATASET, sep=",", encoding="utf-8")

# print(dataset)


# aplicar un unique sobre cada columna para ver los valores únicos

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

        # Reemplazar NaN con el promedio de la columna
        if dataset[col].isnull().any():
            dataset[col].fillna(dataset[col].mean(), inplace=True)

dataset_con_numeros_limpios = dataset.copy()

# dataset_con_numeros_limpios.to_csv(
#     RUTA_DATASET_LIMPIO,
#     index=False,
# )


dataset = pd.read_csv(RUTA_DATASET_LIMPIO, sep=",", encoding="utf-8")

# Información básica del dataset con minimo, maximo y media sólo en columnas numéricas
print(dataset.describe())

# print(dataset["OTROS-AFECTACION"].unique())
# print(dataset["DESCRIPCION MATERIALES DE CONSTRUCCION"].unique())

# for col in dataset.columns:
#     print(f"Valores únicos en {col}: {dataset[col].unique()}")


# dataset.info()
# dataset.columns
# tablita con resumenes estadisticos

# SystemError(1)  # Ejecutar hasta aquí.

## A1. Aplicación de mapeo con una función de limpieza para las variables numéricas


# def limpieza_numeros(dataset: pd.DataFrame):
#     # Aquí se implementa la lógica de limpieza de datos

#     # Por ejemplo, eliminar filas con valores nulos o corregir tipos de datos
#     dataset_numeros_corregidos = None
#     return dataset_numeros_corregidos


# COL_NUMERICAS = dataset.select_dtypes(include=["int64", "float64"]).columns.tolist()

# dataset_con_numeros_limpios = dataset.apply(lambda x: x, axis=0)

### B1. Determinar valores nulos
### B2. Eliminación de valores atípicos (outliers) o su normalización


## A2. Aplicación de mapeo con otra función de limpieza para las variables categóricas


### B1. Determinar valores nulos y rellenarlos


# C1. Exportar el dataset limpio a un archivo CSV
