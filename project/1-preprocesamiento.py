import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Procesamiento ETL y variable(s) categórica(s) (a predecir).

RUTA_DATASET = "data/Emergencias.csv"
RUTA_DATASET_LIMPIO = "data/emergencias_limpio.csv"
dataset = pd.read_csv(RUTA_DATASET, sep=",", encoding="utf-8")

# 1. Convertir columnas que parecen numéricas pero están como texto

def convertir_a_numerico(dataset: pd.DataFrame):
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

convertir_a_numerico(dataset)

# 2. Limpieza a variables categóricas
def limpieza_categoricas(dataset: pd.DataFrame):
    # Primero tomamos las variables categóricas
    dataset_categoricas = dataset.select_dtypes(include=["object"])
    print(dataset_categoricas.columns, 'antes')

    # Pasamos el contenido de las columnas a mayúsculas
    dataset_categoricas = dataset_categoricas.apply(lambda x: x.str.upper() if x.dtype == "object" else x)

limpieza_categoricas(dataset)

# 3   Manejo de valores nulos
def manejo_nulos(dataset: pd.DataFrame):
    print("Valores nulos en cada columna:")
    for col in dataset.columns:
        print(f"{col}: {dataset[col].isnull().sum()}")

    print(dataset.isnull().sum().sort_values(ascending=False))
    print(f"Cantidad total de filas: {dataset.shape[0]}")

    COL_OTROS_AFECTACION = 'OTROS-AFECTACION'
    COL_DESCRIPCION_MATERIALES = 'DESCRIPCION MATERIALES DE CONSTRUCCION'

    # Porcentaje de NAN en las columnas mencionadas: Si pasa del 1% se eliminará la columna
    prob_drop_otros = dataset[COL_OTROS_AFECTACION].isnull().sum() / dataset.shape[0]
    prob_drop_desc = dataset[COL_DESCRIPCION_MATERIALES].isnull().sum() / dataset.shape[0]

    print(f"Probabilidad de eliminar {COL_OTROS_AFECTACION}: {prob_drop_otros}")
    print(f"Probabilidad de eliminar {COL_DESCRIPCION_MATERIALES}: {prob_drop_desc}")

    # Eliminar columnas con menos del 5% de valores no nulos
    columnas_a_eliminar = []
    if prob_drop_otros > 0.95:
        columnas_a_eliminar.append(COL_OTROS_AFECTACION)
    if prob_drop_desc > 0.95:
        columnas_a_eliminar.append(COL_DESCRIPCION_MATERIALES)

    # Eliminar solo si hay columnas para eliminar
    dataset = dataset.drop(columns=columnas_a_eliminar)
    return dataset

dataset = manejo_nulos(dataset)

# 4. Procesar la columna DIVIPOLA para obtener latitud y longitud
def procesar_divipola(dataset: pd.DataFrame, ruta_municipios: str) -> pd.DataFrame:
    # Cargar el archivo de código de municipios
    df_municipios = pd.read_excel(ruta_municipios)

    # Asegurar que las columnas tengan el mismo tipo
    dataset["DIVIPOLA"] = dataset["DIVIPOLA"].astype(str)
    df_municipios["Código"] = df_municipios["Código"].astype(str)

    # Hacer merge entre DIVIPOLA (dataset) y Código (df_municipios), añadiendo solo Latitud y Longitud
    dataset = pd.merge(
        dataset,
        df_municipios[["Código", "Latitud", "Longitud"]],
        left_on="DIVIPOLA",
        right_on="Código",
        how="left"
    )

    # (Opcional) Eliminar la columna "Código" duplicada después del merge si ya no la necesitas
    dataset.drop(columns=["Código"], inplace=True)

    # Reemplazar coma por punto y convertir a float
    dataset["Latitud"] = dataset["Latitud"].astype(str).str.replace(",", ".").astype(float)
    dataset["Longitud"] = dataset["Longitud"].astype(str).str.replace(",", ".").astype(float)

    return dataset

# Llamar a la función procesar_divipola
RUTA_MUNICIPIOS = "data/codigo_municipios.xlsx"
dataset = procesar_divipola(dataset, RUTA_MUNICIPIOS)

# 5. Limpia y unifica los nombres de los eventos en la columna especificada.
def limpiar_eventos(dataset: pd.DataFrame, columna_evento: str) -> pd.DataFrame:
    
    # Diccionario de correcciones: clave es el valor incorrecto, valor es el correcto
    correcciones = {
        "INMERSIoN": "INMERSION",
        "IMERSION": "INMERSION",
        "INUNDACIoN": "INUNDACION",
        "INUNDACIÓN": "INUNDACION",
        "Movimiento en Masa": "MOVIMIENTO EN MASA",
        "Creciente Subita": "CRECIENTE SUBITA",
        "ACCIDENTE TRANSPORTE MARÍTIMO O FLUVIAL": "ACCIDENTE TRANSPORTE MARITIMO O FLUVIAL",
        "ACCIDENTE TRANSPORTE AÉREO": "ACCIDENTE TRANSPORTE AEREO",
        "ACCIDENTE AÉREO": "ACCIDENTE TRANSPORTE AEREO"
    }

    # Convertir todos los valores a mayúsculas para evitar problemas de comparación
    dataset[columna_evento] = dataset[columna_evento].str.upper()

    # Reemplazar los valores incorrectos usando el diccionario de correcciones
    dataset[columna_evento] = dataset[columna_evento].replace(correcciones)

    return dataset

# Llamar a la función para limpiar los eventos
dataset = limpiar_eventos(dataset, "EVENTO")

# 6. Mover eventos raros a la categoría "OTROS"
def mover_eventos_unicos_a_otros(dataset: pd.DataFrame, columna_evento: str) -> pd.DataFrame:
    # Contar la cantidad de registros por evento
    conteo_eventos = dataset[columna_evento].value_counts()

    # Identificcar eventos con menos de 5 registros
    eventos_unicos = conteo_eventos[conteo_eventos < 10].index.tolist()
    # Imprimir eventos únicos y su cantidad

    # Reemplazar los eventos únicos por "OTROS"
    dataset[columna_evento] = dataset[columna_evento].replace(eventos_unicos, "OTROS")
    
    

    return dataset

# Llamar a la función para mover los eventos únicos a "OTROS"
dataset = mover_eventos_unicos_a_otros(dataset, "EVENTO")

# 7 Unificar tipos de incendios
def unificar_incendios(dataset: pd.DataFrame, columna_evento: str) -> pd.DataFrame:
    # Definir los tipos de incendios a unificar
    tipos_incendio = [
        "INCENDIO ESTRUCTURAL",
        "INCENDIO DE COBERTURA VEGETAL",
        "INCENDIO VEHICULAR",
    ]

    # Reemplazar todos los tipos de incendios por "INCENDIO"
    dataset[columna_evento] = dataset[columna_evento].replace(tipos_incendio, "INCENDIO")

    return dataset

unificar_incendios(dataset, "EVENTO")

# Guardar el dataset actualizado
dataset.to_csv(RUTA_DATASET_LIMPIO, index=False)
print(f"Dataset actualizado guardado en {RUTA_DATASET_LIMPIO}")

# imprimir las filas unicas de evento y la cantidad
def imprimir_eventos_unicos(dataset: pd.DataFrame):
    eventos_unicos = dataset["EVENTO"].unique()
    # imprimir cantidad de eventos unicos
    print(f"Cantidad de eventos únicos: {len(eventos_unicos)}")
    print("Eventos únicos y su cantidad:")
    for evento in eventos_unicos:
        cantidad = dataset[dataset["EVENTO"] == evento].shape[0]
        print(f"{evento}: {cantidad}")
        
# Imprimir eventos únicos después de la limpieza
imprimir_eventos_unicos(dataset)


