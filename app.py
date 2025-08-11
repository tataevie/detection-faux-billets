import streamlit as st
import pandas as pd
import requests
import altair as alt

# --- Style CSS simple pour améliorer l'apparence ---
st.markdown(
    """
    <style>
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f0f2f6;
        color: #0f1c4d;
        padding: 1rem 2rem;
    }
    .stButton>button {
        background-color: #043a6b;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #005a9e;
        color: white;
    }
    header blockquote {
        font-size: 1.2rem;
        font-weight: 600;
        color: #18191a;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Titre et description ---
st.title("💵 Détection de Faux Billets")
st.markdown("""
API simple et rapide pour détecter les faux billets à partir de caractéristiques géométriques.
- Uploadez un fichier CSV avec les mesures.
- Choisissez un modèle ML performant.
- Visualisez les prédictions, statistiques et graphiques interactifs.
""")

# --- Sidebar pour upload et sélection modèle ---
st.sidebar.header("Configuration de la prédiction")

uploaded_file = st.sidebar.file_uploader("📂 Importez votre fichier CSV", type=["csv"])
model_name = st.sidebar.selectbox("🧠 Sélectionnez un modèle", options=[
    "random_forest",
    "logistic_regression",
    "knn",
    "kmeans"
])

# URL de l'API - à adapter si déployée en ligne
API_URL = "http://localhost:8000/predict"

# Bouton prédiction
if uploaded_file is not None:
    if st.sidebar.button("🚀 Lancer la prédiction"):
        with st.spinner("🔄 Appel de l'API et calcul des prédictions..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                params = {"model_name": model_name}
                response = requests.post(API_URL, files=files, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    results_df = pd.DataFrame(data["results"])

                    st.success(f"✅ Prédictions réussies avec le modèle : **{model_name}**")
                    
                    # Affichage tableau clair
                    st.subheader("📋 Résultats des Prédictions")
                    st.dataframe(results_df.style.highlight_max(axis=0, color="#ffd966"))

                    # Statistiques : vrais vs faux
                    st.subheader("📊 Statistiques")
                    count_pred = results_df['prediction'].value_counts().rename({0: "Vrai billet", 1: "Faux billet"})
                    count_pred_df = count_pred.reset_index()
                    count_pred_df.columns = ["Classe", "Nombre"]

                    st.write(count_pred_df)

                    # Graphique interactif Altair - répartition classes
                    st.subheader("📈 Visualisation des Prédictions")
                    bar_chart = alt.Chart(count_pred_df).mark_bar(color="#0078d4").encode(
                        x=alt.X('Classe', sort=None, title="Classe"),
                        y=alt.Y('Nombre', title="Nombre de billets"),
                        tooltip=["Classe", "Nombre"]
                    ).properties(width=600)
                    st.altair_chart(bar_chart, use_container_width=True)


                    # Histogramme probabilités de faux si dispo
                    if "probability_faux" in results_df.columns and results_df["probability_faux"].notnull().any():
                        st.subheader("🔍 Distribution des probabilités de faux billets")
                        hist = alt.Chart(results_df).mark_bar(color="#e63946").encode(
                            alt.X("probability_faux", bin=alt.Bin(maxbins=30), title="Probabilité de faux billet"),
                            y='count()',
                            tooltip=['count()']
                        ).properties(width=600)
                        st.altair_chart(hist, use_container_width=True)

                else:
                    st.error(f"❌ Erreur API : {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"❌ Erreur lors de l'appel API : {str(e)}")

else:
    st.info("📥 Veuillez importer un fichier CSV dans la barre latérale pour commencer.")
