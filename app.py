#Крок 3
import os
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from PIL import Image

st.set_page_config(page_title="AI Image Clustering", layout="wide")
st.title("AI-система кластеризації та пошуку зображень")

FEATURE_PATH = "features/features.npy"

@st.cache_data
def load_data():
    if not os.path.exists(FEATURE_PATH):
        return None, None
    data = np.load(FEATURE_PATH, allow_pickle=True).item()
    return data["paths"], data["features"]


image_paths, features = load_data()

if image_paths is None:
    st.error("Файл з ознаками не знайдено, Будь ласка спочатку запустіть скрипт витягування ознак!")
else:
    st.sidebar.header("Налаштування")
    n_clusters = st.sidebar.slider(
        "Кількість кластерів (K)",
        min_value=2,
        max_value=10,
        value=4,
        step=1
    )

    with st.spinner("Виконується кластеризація KMeans..."):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

    score = silhouette_score(features, cluster_labels)

    st.subheader("Оцінка якості кластеризації")
    st.metric(label="Silhouette Score (вище - краще)", value=f"{score:.4f}")
    st.subheader("Вміст кластерів")
    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        with st.expander(f"Кластер {cluster_id} (Кількість зображень: {len(cluster_indices)})"):
            cols = st.columns(5)
            for i, idx in enumerate(cluster_indices[:10]):
                col_idx = i % 5
                with cols[col_idx]:
                    img = Image.open(image_paths[idx])
                    st.image(img, width=120, caption=f"ID: {idx}")




#Крок 4
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.subheader("PCA Візуалізація")

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)
var = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap="tab10", alpha=0.7)
ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
plt.colorbar(sc, ax=ax, label="Кластер")

st.pyplot(fig)





#Крок 5
from sklearn.metrics.pairwise import cosine_similarity

st.subheader("Пошук схожих зображень...")

selected_idx = st.selectbox(
    "Оберіть зображення для пошуку:",
    range(len(image_paths)),
    format_func=lambda x: image_paths[x]
)

if selected_idx is not None:
    st.image(image_paths[selected_idx], caption="Обране зображення", width=200)

    sims = cosine_similarity([features[selected_idx]], features)[0]
    top_indices = sims.argsort()[::-1][1:6]

    st.write("Топ-5 схожих:")
    cols = st.columns(5)
    for i, idx in enumerate(top_indices):
        with cols[i]:
            st.image(image_paths[idx], width=120, caption=f"Сх: {sims[idx]:.3f}")




#Крок 6
import pandas as pd
import sqlite3
from datetime import datetime

st.subheader("Збереження результатів")

if st.button("Зберегти у CSV та БД"):
    df = pd.DataFrame({
        "file_path": image_paths,
        "cluster": cluster_labels
    })
    df.to_csv("features/clustering_results.csv", index=False)

    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect("db/history.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            clusters INTEGER,
            score REAL
        )
    """)
    cur.execute(
        "INSERT INTO history (time, clusters, score) VALUES (?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), n_clusters, float(score))
    )
    conn.commit()
    conn.close()

    st.success("Успішно збережено!")
    st.download_button(
        "Завантажити CSV",
        data=df.to_csv(index=False),
        file_name="clustering_results.csv",
        mime="text/csv"
    )

