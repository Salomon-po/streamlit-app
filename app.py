import streamlit as st
import joblib
import numpy as np

# Charger le modèle et le scaler
model = joblib.load("Dtree_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title(" Prédiction de la gamme de prix d’un téléphone")

st.markdown("**Entrez les caractéristiques du téléphone :**")

# Création des 20 champs d'entrée
battery_power = st.slider("Battery Power (mAh)", 500, 2000)
clock_speed = st.slider("Vitesse CPU (GHz)", 0.5, 3.0, step=0.1)
fc = st.slider("Front Camera (MP)", 0, 20)
int_memory = st.slider("Mémoire Interne (Go)", 2, 128)
mobile_wt = st.slider("Poids du mobile (g)", 80, 250)
n_cores = st.slider("Nombre de cœurs", 1, 8)
pc = st.slider("Primary Camera (MP)", 0, 20)
px_height = st.slider("Hauteur en pixels", 0, 1960)
px_width = st.slider("Largeur en pixels", 500, 2000)
ram = st.slider("RAM (Mo)", 256, 4000)
sc_h = st.slider("Hauteur écran (cm)", 5, 20)
sc_w = st.slider("Largeur écran (cm)", 0, 20)
talk_time = st.slider("Autonomie en communication (heures)", 2, 20)
touch_screen = st.selectbox("Écran tactile", [0, 1])
# Liste des features dans l’ordre
features = [
    battery_power, clock_speed, fc,
    int_memory, mobile_wt, n_cores, pc, px_height,
    px_width, ram, sc_h, sc_w, talk_time, touch_screen
]
while len(features) < 20:
    features.append(0)  # Ajouter des zéros jusqu'à obtenir 20 colonnes

if st.button("Prédire"):
    X = np.array([features])  # transformer en matrice 2D
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    labels = ["Low Cost", "Medium Cost", "High Cost", "Premium"]
    st.success(f"Gamme prédite : **{labels[prediction]}**")
