from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import MarkerCluster
import matplotlib.cm as cm

# Cargar datos
df = pd.read_csv("data/emergencias_limpio.csv")

# Filtrar ubicaciones con coordenadas válidas
geo_df = df.dropna(subset=['Latitud', 'Longitud'])

# Crear características para clustering
# Incluimos coordenadas e indicadores de impacto
features_cluster = ['Latitud', 'Longitud', 'PERSONAS', 'FAMILIAS', 
                    'VIVIENDAS_DESTRUIDAS', 'VIVIENDAS_AVERIADAS']

# Reemplazar NaN con 0 en las características
X_cluster = geo_df[features_cluster].fillna(0)

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Aplicar K-means para identificar zonas de vulnerabilidad similar
kmeans = KMeans(n_clusters=5, random_state=42)
geo_df['cluster'] = kmeans.fit_predict(X_scaled)

# Calcular el impacto promedio por cluster
cluster_stats = geo_df.groupby('cluster').agg({
    'PERSONAS': 'mean',
    'FAMILIAS': 'mean',
    'VIVIENDAS_DESTRUIDAS': 'mean',
    'VIVIENDAS_AVERIADAS': 'mean',
    'EVENTO': lambda x: x.value_counts().index[0], # Evento más común
    'Latitud': 'mean',
    'Longitud': 'mean'
}).reset_index()

print("Estadísticas por cluster:")
print(cluster_stats)

# Crear mapa de Colombia con clusters
map_center = [4.5709, -74.2973]  # Centrado en Colombia
mymap = folium.Map(location=map_center, zoom_start=6)

# Colores para los clusters
colors = cm.rainbow(np.linspace(0, 1, kmeans.n_clusters))
cluster_colors = {i: '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) 
                  for i, (r, g, b, _) in enumerate(colors)}

# Añadir marcadores por cluster
for cluster in range(kmeans.n_clusters):
    cluster_data = geo_df[geo_df['cluster'] == cluster]
    
    # Crear un grupo de marcadores por cluster
    marker_cluster = MarkerCluster(name=f'Cluster {cluster}').add_to(mymap)
    
    # Añadir cada punto al grupo
    for _, row in cluster_data.iterrows():
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
            color=cluster_colors[row['cluster']],
            fill=True,
            fill_color=cluster_colors[row['cluster']],
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(marker_cluster)

# Añadir leyenda
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
            padding: 10px; border: 2px solid grey; border-radius: 5px;">
<h4>Clusters de vulnerabilidad</h4>
'''

for i in range(kmeans.n_clusters):
    legend_html += f'''
    <div>
        <span style="background-color:{cluster_colors[i]}; width:20px; height:20px; display:inline-block;"></span>
        <span> Cluster {i}: {cluster_stats.loc[cluster_stats['cluster']==i, 'EVENTO'].values[0]}</span>
    </div>
    '''

legend_html += '</div>'

mymap.get_root().html.add_child(folium.Element(legend_html))

# Guardar mapa
mymap.save('clusters_vulnerabilidad.html')