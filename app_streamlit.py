import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8080")

# Configuration globale
st.set_page_config(
    page_title="Churn Analytics Pipeline",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Banque d'icônes SVG intégrée (Ajout de l'icône "download")
SVG_ICONS = {
    "chart": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>',
    "folder": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6B7280" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>',
    "success": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
    "error": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>',
    "prediction": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>',
    "cpu": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="15" x2="23" y2="15"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="15" x2="4" y2="15"></line></svg>',
    "download": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>',
}

# Injection CSS (Ajout du style du bouton de téléchargement customisé)
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
    .svg-align { display: inline-flex; align-items: center; gap: 8px; vertical-align: middle; }
    
    .download-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        width: 100%;
        background-color: #4F46E5;
        color: white !important;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 14px;
        text-decoration: none !important;
        transition: background-color 0.2s ease, transform 0.1s ease;
        border: 1px solid transparent;
        text-align: center;
        cursor: pointer;
    }
    .download-btn:hover {
        background-color: #4338CA;
        transform: translateY(-1px);
    }
    .download-btn:active {
        transform: translateY(0);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialisation des variables d'état de session
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None

# Titre Principal
st.markdown(
    f'<h1 class="svg-align">{SVG_ICONS["chart"]} Plateforme MLOps : Prédiction & Analyse</h1>',
    unsafe_allow_html=True,
)
st.caption("Suivi des métriques clés et déploiement de modèles en temps réel")

# --- SIDEBAR ---
st.sidebar.markdown(
    f'<div class="svg-align">{SVG_ICONS["folder"]} <b>Source des données</b></div>',
    unsafe_allow_html=True,
)
csv_url = st.sidebar.text_input(
    "URL du fichier CSV (Optionnel)",
    placeholder="https://url-du-fichier.csv",
    label_visibility="collapsed",
)
default_file = "Churn_Modelling(in).csv"

# Chargement sécurisé des données
df = pd.DataFrame()
try:
    source = csv_url if csv_url else default_file
    df = pd.read_csv(source)
    st.sidebar.markdown(
        f'<div class="svg-align" style="color:#10B981;">{SVG_ICONS["success"]} Données connectées</div>',
        unsafe_allow_html=True,
    )
except Exception as e:
    st.sidebar.markdown(
        f'<div class="svg-align" style="color:#EF4444;">{SVG_ICONS["error"]} Erreur : {e}</div>',
        unsafe_allow_html=True,
    )
    try:
        df = pd.read_csv(default_file)
        st.sidebar.info("Utilisation du fichier local par défaut.")
    except Exception as default_err:
        st.sidebar.error(f"Fichier par défaut introuvable : {default_err}")

# --- SECTION 1: FORMULAIRE DE PRÉDICTION ---
st.markdown(
    f'<h3 class="svg-align">{SVG_ICONS["prediction"]} Simulation d\'Inférence Client</h3>',
    unsafe_allow_html=True,
)
with st.container(border=True):
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        cs = st.number_input("Score de Crédit", min_value=300, max_value=850, value=600)
        age = st.number_input("Âge du Client", min_value=18, max_value=100, value=40)
        tenure = st.slider("Ancienneté (Années)", 0, 10, 3)
    with col_p2:
        balance = st.number_input("Solde du Compte (€)", min_value=0.0, value=60000.0)
        num_prod = st.selectbox("Nombre de Produits", [1, 2, 3, 4], index=1)
        salary = st.number_input("Salaire Estimé (€)", min_value=0.0, value=50000.0)
    with col_p3:
        gender = st.radio("Genre", ["Femme", "Homme"], horizontal=True)
        has_card = st.toggle("Possède une Carte de Crédit", value=True)
        active = st.toggle("Membre Actif", value=True)

    gender_encoded = 0 if gender == "Femme" else 1
    has_card_encoded = 1 if has_card else 0
    active_encoded = 1 if active else 0

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Calculer le Score de Risque", type="primary"):
        client_payload = {
            "CreditScore": cs,
            "Gender": gender_encoded,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_prod,
            "HasCrCard": has_card_encoded,
            "IsActiveMember": active_encoded,
            "EstimatedSalary": salary,
        }
        try:
            res = requests.post(f"{API_URL}/predict", json=client_payload, timeout=5)
            res.raise_for_status()
            result = res.json()
            prob = result.get("churn_probability", 0.0)

            st.markdown("---")
            col_r1, col_r2 = st.columns([1, 2])
            with col_r1:
                st.metric(
                    label="Prediction de Churn", value=f"{result.get('prediction')}"
                )
            with col_r2:
                if result.get("prediction") == 1:
                    st.markdown(
                        f'<div class="svg-align" style="color:#EF4444; padding:10px; border-left:4px solid #EF4444; background:#FEF2F2;">{SVG_ICONS["error"]} <b>Alerte :</b> Churn</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="svg-align" style="color:#10B981; padding:10px; border-left:4px solid #10B981; background:#F0FDF4;">{SVG_ICONS["success"]} <b>Statut Stable :</b> Not Churn</div>',
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.markdown(
                f'<div class="svg-align" style="color:#EF4444;">{SVG_ICONS["error"]} Impossible de joindre l\'API : {e}</div>',
                unsafe_allow_html=True,
            )

st.markdown("<br>", unsafe_allow_html=True)

# --- SECTION 2: VISUALISATION ET BACKEND ---
tab1, tab2 = st.tabs(["Graphiques & Exploration", "Administration du Pipeline"])

with tab1:
    if not df.empty:
        st.dataframe(df.head(5), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            feature = st.selectbox(
                "Distribution de la variable :",
                ["Age", "CreditScore", "Balance", "EstimatedSalary"],
            )
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.histplot(df[feature], kde=True, ax=ax, color="#4F46E5")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.heatmap(
                df.select_dtypes(include=["int64", "float64"]).corr(),
                cmap="coolwarm",
                annot=False,
            )
            st.pyplot(fig)
    else:
        st.warning(
            "Veuillez connecter une source de données valide pour afficher les visualisations."
        )

with tab2:
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        with st.container(border=True):
            st.markdown(
                f'<div class="svg-align">{SVG_ICONS["cpu"]} <b>Étape 1 : Structuration</b></div>',
                unsafe_allow_html=True,
            )
            if st.button("Exécuter /prepare"):
                try:
                    res = requests.post(
                        f"{API_URL}/prepare",
                        json={"data_path": default_file},
                        timeout=10,
                    )
                    if res.status_code == 200:
                        st.success("Données préparées !")
                    else:
                        st.error(
                            f"Échec de la préparation (Statut : {res.status_code})"
                        )
                except Exception as e:
                    st.error(f"Impossible de joindre l'API : {e}")

    with col_btn2:
        with st.container(border=True):
            st.markdown(
                f'<div class="svg-align">{SVG_ICONS["cpu"]} <b>Étape 2 : Entraînement</b></div>',
                unsafe_allow_html=True,
            )
            chosen_model = st.selectbox(
                "Modèle cible :",
                ["random_forest", "logistic_regression", "gradient_boosting"],
            )

            if st.button("Lancer l'entraînement (/train)"):
                try:
                    res = requests.post(
                        f"{API_URL}/train",
                        json={"model_name": chosen_model},
                        timeout=30,
                    )
                    if res.status_code == 200:
                        data_res = res.json()
                        st.session_state.trained_model = {
                            "filename": data_res.get("saved_filename", "model.joblib"),
                            "accuracy": data_res.get("accuracy", 0.0),
                        }
                    else:
                        st.session_state.trained_model = "error"
                except Exception as e:
                    st.session_state.trained_model = f"connection_error: {e}"

            # Affichage persistant des résultats d'entraînement et téléchargement via SVG custom
            if st.session_state.trained_model:
                if isinstance(st.session_state.trained_model, dict):
                    model_info = st.session_state.trained_model
                    filename = model_info["filename"]
                    accuracy = model_info["accuracy"]

                    st.metric(label="Accuracy", value=f"{int(accuracy)}")
                    st.markdown(
                        f'<div class="svg-align" style="color:#10B981; margin-top:10px; margin-bottom:10px;">{SVG_ICONS["success"]} Modèle {filename} créé.</div>',
                        unsafe_allow_html=True,
                    )

                    try:
                        # Lecture et encodage en base64 pour téléchargement en HTML natif
                        with open(filename, "rb") as f:
                            bytes_data = f.read()
                        b64 = base64.b64encode(bytes_data).decode()

                        href = f"""
                            <a href="data:application/octet-stream;base64,{b64}" download="{filename}" class="download-btn">
                                {SVG_ICONS["download"]} Télécharger le modèle (.joblib)
                            </a>
                        """
                        st.markdown(href, unsafe_allow_html=True)
                    except FileNotFoundError:
                        st.error(
                            f"Fichier de modèle '{filename}' introuvable localement."
                        )
                elif st.session_state.trained_model == "error":
                    st.markdown(
                        f'<div class="svg-align" style="color:#EF4444; margin-top:10px;">{SVG_ICONS["error"]} Erreur d\'entraînement.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    err_msg = st.session_state.trained_model.replace(
                        "connection_error: ", ""
                    )
                    st.markdown(
                        f'<div class="svg-align" style="color:#EF4444; margin-top:10px;">{SVG_ICONS["error"]} Impossible de joindre l\'API : {err_msg}</div>',
                        unsafe_allow_html=True,
                    )
