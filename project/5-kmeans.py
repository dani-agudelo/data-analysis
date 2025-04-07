# 1. Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import MarkerCluster

# 2. Cargar datos
df = pd.read_csv("data/emergencias_limpio.csv")

# 3. Seleccionar características para el clustering
features_cluster = [
    'Latitud', 'Longitud',  # Ubicación geográfica
    'PERSONAS', 'FAMILIAS',  # Afectación humana
    'VIVIENDAS_DESTRUIDAS', 'VIVIENDAS_AVERIADAS'  # Afectación infraestructura
]

# 4. Filtrar datos y limpiar valores nulos
cluster_df = df.dropna(subset=['Latitud', 'Longitud'])
X_cluster = cluster_df[features_cluster].fillna(0)  # Reemplazar NaN con 0

# 5. Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# 6. Determinar número óptimo de clusters (método del codo)
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Visualizar el método del codo
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.title('Método del codo para determinar el número óptimo de clusters')
plt.grid(True)
plt.savefig('metodo_codo.png')
plt.show()

# 7. Aplicar K-means con el número óptimo de clusters (ejemplo: 5)
n_clusters = 5  # Ajustar según el método del codo
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_df['cluster'] = kmeans.fit_predict(X_scaled)

# 8. Analizar características de cada cluster
cluster_stats = cluster_df.groupby('cluster').agg({
    'PERSONAS': 'mean',
    'FAMILIAS': 'mean',
    'VIVIENDAS_DESTRUIDAS': 'mean',
    'VIVIENDAS_AVERIADAS': 'mean',
    'EVENTO': lambda x: x.value_counts().index[0],  # Evento más común
    'DEPARTAMENTO': lambda x: x.value_counts().index[0],  # Departamento más común
    'Latitud': 'mean',
    'Longitud': 'mean'
}).reset_index()

print("Características de cada cluster:")
print(cluster_stats)

# 9. Crear un mapa de Colombia con los clusters
map_center = [4.5709, -74.2973]  # Centro aproximado de Colombia
mymap = folium.Map(location=map_center, zoom_start=6)

# Definir colores para los clusters
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']

# 10. Añadir puntos al mapa por cluster
for cluster in range(n_clusters):
    cluster_points = cluster_df[cluster_df['cluster'] == cluster]
    
    # Mostrar resumen del cluster
    print(f"\nCluster {cluster}:")
    print(f"Número de eventos: {len(cluster_points)}")
    print(f"Evento más común: {cluster_stats.loc[cluster_stats['cluster']==cluster, 'EVENTO'].values[0]}")
    print(f"Departamento más común: {cluster_stats.loc[cluster_stats['cluster']==cluster, 'DEPARTAMENTO'].values[0]}")
    
    # Crear grupo de marcadores
    marker_cluster = MarkerCluster(name=f'Cluster {cluster}').add_to(mymap)
    
    # Añadir cada punto
    for _, row in cluster_points.iterrows():
        popup_text = f"""
        <b>Evento:</b> {row['EVENTO']}<br>
        <b>Fecha:</b> {row['FECHA']}<br>
        <b>Departamento:</b> {row['DEPARTAMENTO']}<br>
        <b>Municipio:</b> {row['MUNICIPIO']}<br>
        <b>Personas afectadas:</b> {row['PERSONAS']}<br>
        <b>Cluster:</b> {row['cluster']}
        """
        
        folium.CircleMarker(
            location=[row['Latitud'], row['Longitud']],
            radius=5,
            color=colors[cluster % len(colors)],
            fill=True,
            fill_color=colors[cluster % len(colors)],
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(marker_cluster)

# 11. Guardar el mapa
mymap.save('mapa_clusters_vulnerabilidad.html')
print("Mapa guardado como 'mapa_clusters_vulnerabilidad.html'")