# 1. Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# 2. Función para balancear las clases
def balancear_clases(df: pd.DataFrame, columna_evento: str, umbral: int) -> pd.DataFrame:
    # Submuestreo para clases mayoritarias
    df_balanceado = pd.DataFrame()
    for clase, grupo in df.groupby(columna_evento):
        if len(grupo) > umbral:
            # Submuestrear si la clase tiene más registros que el umbral
            grupo_sub = resample(
                grupo,
                replace=False,  # Sin reemplazo
                n_samples=umbral,
                random_state=42
            )
        else:
            # Mantener la clase sin cambios si tiene menos registros que el umbral
            grupo_sub = grupo
        df_balanceado = pd.concat([df_balanceado, grupo_sub])

    # Imputar valores faltantes en las columnas seleccionadas
    df_balanceado['Latitud'] = df_balanceado['Latitud'].fillna(df_balanceado['Latitud'].mean())
    df_balanceado['Longitud'] = df_balanceado['Longitud'].fillna(df_balanceado['Longitud'].mean())
    df_balanceado['MES'] = df_balanceado['MES'].fillna(df_balanceado['MES'].mean())
    df_balanceado['DIA_AÑO'] = df_balanceado['DIA_AÑO'].fillna(df_balanceado['DIA_AÑO'].mean())

    # Aplicar SMOTE a las clases minoritarias
    X = df_balanceado[['Latitud', 'Longitud', 'MES', 'DIA_AÑO']]
    y = df_balanceado[columna_evento]

    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    # Combinar las características y la variable objetivo en un DataFrame
    df_balanceado_smote = pd.DataFrame(X_smote, columns=['Latitud', 'Longitud', 'MES', 'DIA_AÑO'])
    df_balanceado_smote[columna_evento] = y_smote

    return df_balanceado_smote

# 3. Cargar el dataset
df = pd.read_csv("data/emergencias_limpio.csv")

# 4. Preparar las columnas temporales antes del balanceo
df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
df['MES'] = df['FECHA'].dt.month  # Extraer el mes (1-12)
df['DIA_AÑO'] = df['FECHA'].dt.dayofyear  # Día del año (1-365)

# Verificar el balance inicial de clases
evento_counts = df['EVENTO'].value_counts()
print(evento_counts, 'cantidad eventos')
print(f"Ratio mayor/menor: {evento_counts.max()/evento_counts.min():.2f}")

# 5. Aplicar balanceo de clases
umbral = 500
df_balanceado = balancear_clases(df, 'EVENTO', umbral)

# Verificar el balance después del balanceo
print("--------------------------------------------------------------------------")
print("\nDistribución de clases después del balanceo:")
print(df_balanceado['EVENTO'].value_counts())

# 6. Codificar la variable objetivo (tipo de evento) para que el algoritmo pueda entenderla
le = LabelEncoder()
df_balanceado['EVENTO_ENCODED'] = le.fit_transform(df_balanceado['EVENTO'])
evento_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print("--------------------------------------------------------------------------")
print("\nCodificación de eventos:")
for evento, codigo in evento_mapping.items():
    print(f"{evento}: {codigo}")

# 7. Definir características y variable objetivo
X = df_balanceado[['Latitud', 'Longitud', 'MES', 'DIA_AÑO']]  # Características: ubicación y tiempo
y = df_balanceado['EVENTO_ENCODED']  # Variable a predecir: tipo de evento

# 8. Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 9. Escalar características numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. Crear y entrenar el modelo
print("--------------------------------------------------------------------------")
print("\nEntrenando modelo Random Forest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# 11. Evaluar el modelo
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("--------------------------------------------------------------------------")
print(f"\nPrecisión del modelo: {accuracy:.4f}")

# Mostrar reporte detallado de clasificación
print("--------------------------------------------------------------------------")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 12. Visualizar importancia de características
feature_importance = clf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Importancia de la característica')
plt.title('¿Qué factores influyen más en el tipo de evento?')
plt.tight_layout()
plt.savefig('importancia_caracteristicas.png')
plt.show()