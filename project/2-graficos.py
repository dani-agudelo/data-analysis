import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta al archivo CSV
RUTA_DATASET_LIMPIO = "data/emergencias_limpio.csv"

# Cargar y limpiar el dataset
df = pd.read_csv(RUTA_DATASET_LIMPIO)

# Reemplazar comas por puntos y convertir columnas relevantes a float
for col in ['Latitud', 'Longitud', 'FALLECIDOS']:
    df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')


def grafica_fallecidos(): 
    # Crear mapa
    fig = px.scatter_mapbox(
        df,
        lat="Latitud",
        lon="Longitud",
        size="FALLECIDOS",  # Tamaño según fallecidos
        color="FALLECIDOS",  # Color según fallecidos
        color_continuous_scale=["blue", "purple"],
        size_max=30,  # Tamaño máximo del punto
        zoom=5,
        mapbox_style="carto-darkmatter",
        height=600,
        width=1000,
        opacity=0.6,
        hover_name="FALLECIDOS",  # Muestra los fallecidos al pasar el mouse
    )

    # Layout
    fig.update_layout(
        title="Mapa interactivo de emergencias según número de fallecidos",
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        uirevision='constant'
    )

    fig.show()
    
def grafica_eventos():
    # Crear mapa
    fig = px.scatter_mapbox(
        df,
        lat="Latitud",
        lon="Longitud",
        color="EVENTO",  # Color según el tipo de evento
        color_discrete_sequence=px.colors.qualitative.Plotly,
        zoom=5,
        mapbox_style="carto-darkmatter",
        height=600,
        width=1000,
        opacity=0.6,
    )
    fig.update_layout(
        title="Mapa interactivo de emergencias según tipo de evento",
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        uirevision='constant'
    )

    fig.show()

# grafica_eventos()
full_dataframe_resampled = pd.read_csv(RUTA_DATASET_LIMPIO)
def matriz_correlacion():


    # ---------- PARTE 1: LIMPIAR LAS COLUMNAS CON COMAS ----------
    # Reemplazar comas por puntos y convertir a float donde sea posible
    for col in full_dataframe_resampled.columns:
        if full_dataframe_resampled[col].dtype == 'object':
            full_dataframe_resampled[col] = full_dataframe_resampled[col].str.replace(",", ".", regex=False)
            try:
                full_dataframe_resampled[col] = full_dataframe_resampled[col].astype(float)
            except ValueError:
                pass  # Dejar las que no se pueden convertir

   
    
    # Seleccionar columnas numéricas
    df_numeric = full_dataframe_resampled.select_dtypes(include=['float64', 'int64'])

  
    # Matriz de correlación
    correlation_matrix = df_numeric.corr()

    # ---------- PARTE 3: GRÁFICA ----------
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlación entre Características')
    plt.show()


# matriz_correlacion()

grafica_fallecidos()


# # Gráfico de dispersión entre dos variables
# # Crear un gráfico de dispersión
# fig = px.scatter(
#     df,
#     x="FALLECIDOS",
#     y="HERIDOS",
#     color="EVENTO",
#     title="Gráfico de dispersión entre FALLECIDOS y HERIDOS",
# )
# # Actualizar el layout
# fig.update_layout(
#     xaxis_title="FALLECIDOS",
#     yaxis_title="HERIDOS",
#     width=800,
#     height=600,
# )