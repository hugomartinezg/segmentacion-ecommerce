from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

# Crear aplicación Flask
app = Flask(__name__)

# Generar datos sintéticos
X, y = make_blobs(n_samples=200, centers=3, cluster_std=1.5, random_state=42)
df = pd.DataFrame(X, columns=["frecuencia_compra", "monto_promedio"])

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Clustering y clasificación
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
df['cluster'] = kmeans.labels_

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, df['cluster'])

# Ruta principal
@app.route('/')
def home():
    return render_template('formulario.html')

# Ruta de clasificación
@app.route('/clasificar', methods=['POST'])
def clasificar():
    frecuencia = float(request.form['frecuencia'])
    monto = float(request.form['monto'])
    entrada = scaler.transform([[frecuencia, monto]])
    cluster = knn.predict(entrada)[0]
    return f"El usuario pertenece al segmento: {cluster}"

# Ejecutar servidor
if __name__ == '__main__':
    app.run(debug=True)
