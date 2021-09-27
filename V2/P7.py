import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

    ######################################
    #Title display
html_temp = """
<div style="background-color: #0649E3; padding:10px; border-radius:10px">
<h1 style="color:white; text-align:center"> Dashboard Scoring Credit</div>
<p style="font-size: 20px; font-weight: bold; text-align:center"> Credit decision support...</p>
"""
st.markdown(html_temp, unsafe_allow_html=True)

st.write("L'application qui prédit l'accord du crédit")


# Collecter le profil d'entrée
st.sidebar.header("Les caracteristiques du client")

df=pd.read_csv("df_new2.csv")

def client_caract_entree():
    AMT_ANNUITY= st.sidebar.slider(label="Montant de l'annuité",min_value=1500, max_value=150000,step= 1000)
    CODE_GENDER=st.sidebar.selectbox("Sexe", ('Male', 'Female'))
    DAYS_BIRTH= st.sidebar.slider(label="Âge en jours", min_value=21, max_value=70, step=1)
    DAYS_EMPLOYED= st.sidebar.slider(label="Depuis combien de temps la personne travaille", min_value=0,
                                     max_value=44, step=1)
    CNT_CHILDREN = st.sidebar.slider(label="Nombre d'enfants", min_value=0, max_value=10, step=1)
    AMT_CREDIT= st.sidebar.slider(label="Montant du prêt",min_value=45000, max_value=1300000, step=10000)
    AMT_INCOME_TOTAL= st.sidebar.slider(label="Montant Total du Revenu",min_value=25000,
                                        max_value=360000, step=10000)

    if CODE_GENDER == "Male":
        CODE_GENDER = 0
    else:
        CODE_GENDER = 1

    data= {'AMT_ANNUITY': AMT_ANNUITY,
            'CODE_GENDER': CODE_GENDER,
            'DAYS_BIRTH': DAYS_BIRTH,
            'DAYS_EMPLOYED': DAYS_EMPLOYED,
            'CNT_CHILDREN' : CNT_CHILDREN,
            'AMT_CREDIT': AMT_CREDIT,
            'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL}

    profil_client=pd.DataFrame(data, index=[0])
    return profil_client

input_df=client_caract_entree()

# Transformer les données d'entrée en données adaptées à notre modèle
df=pd.read_csv("df_new2.csv")
credit_input=df.drop(["SK_ID_CURR"], axis=1)
donnee_entree=pd.concat([input_df, credit_input], axis=0)

#prendre uniquement la première ligne
donnee_entree=donnee_entree[:1]
#afficher les données transformées
st.subheader("Les caracteristiques transformés")
st.write(donnee_entree)

# importer le modèle
load_model=pickle.load(open("LGBMClassifier_API2.pkl", "rb"))

# appliquer le modèle sur le profil d'entrée
prevision=load_model.predict_proba(donnee_entree)

st.subheader("Résultat de la prévision")
st.write(prevision)


# Section Graphiques
st.subheader("Section Graphiques")

@st.cache
def load_income_population(df):
    df_income = pd.DataFrame(df["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income["AMT_INCOME_TOTAL"] < 200000, :]
    return df_income

# Income Distribution Plot
data_income= load_income_population(df)
fig, ax = plt.subplots(figsize=(7,3))
sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor="k", color="royalblue", bins=10)
ax.set(title="Customer Income", xlabel="Income (USD)", ylabel="")
st.pyplot(fig)

@st.cache
def load_age_population(df):
    data_age=round(df["DAYS_BIRTH"], 2)
    return data_age

# Age Distribution Plot
data_age = load_age_population(df)
fig, ax = plt.subplots(figsize=(7,3))
sns.histplot(data_age, edgecolor="k", color="royalblue", bins=20)
ax.set(title="Customer Age", xlabel="Age (Year)", ylabel="")
st.pyplot(fig)
