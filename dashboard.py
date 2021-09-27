import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

plt.style.use("fivethirtyeight")
sns.set_style("darkgrid")


@st.cache
def load_data():
    data= pd.read_csv("data.csv", index_col="SK_ID_CURR", encoding= "utf-8")
    sample = pd.read_csv("sample.csv", index_col="SK_ID_CURR", encoding= "utf-8")

    target = sample.iloc[:, -1:]

    return data, sample, target

def load_model():
    pickle_in = open("LGBMClassifier_App_New.pkl", "rb")
    clf=pickle.load(pickle_in)
    return clf


@st.cache
def load_infos_gen(data):
    lst_infos = [data.shape[0],
                 round(data["AMT_INCOME_TOTAL"].mean(),2),
                 round(data["AMT_CREDIT"].mean(), 2)]

    nb_credits=lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]

    targets = data.TARGET.value_counts

    return nb_credits, rev_moy, credits_moy, targets

def identite_client(data, id):
    data_client = data[data.index == int(id)]
    return data_client

@st.cache
def load_age_population(data):
    data_age=round((data["DAYS_BIRTH"]/365), 2)
    return data_age

@st.cache
def load_income_population(sample):
    df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income["AMT_INCOME_TOTAL"] < 200000, :]
    return df_income

@st.cache
def load_prediction(sample, id, clf):
    X=sample.iloc[:, :-1]
    score = clf.predict_proba(X[X.index == int(id)])[:,1]
    return score


#Loading data....
data, sample, target =load_data()
id_client = sample.index.values
clf=load_model()

    #######################################
    # SIDEBAR
    ######################################
    #Title display
html_temp = """
<div style="background-color: #0649E3; padding:10px; border-radius:10px">
<h1 style="color:white; text-align:center"> Dashboard of Loan Request</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

#Inserting Logo Prêt à dépenser:
from PIL import Image
image = Image.open("Prêt-à-dépenser-logo.png")
st.sidebar.image(image, use_column_width="auto")

#Customer ID Selection
st.sidebar.header("**General Info**")

#Loading Selectobox
chk_id=st.sidebar.selectbox("Client ID", id_client)

#Loading General Info
nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data)

### Display of information in the sidebar ###
# Number of Loans in the sample
st.sidebar.markdown("<u>Number of loans in sample:<u>", unsafe_allow_html=True)
st.sidebar.text(nb_credits)

# Average Income
st.sidebar.markdown("<u> Average Income (USD)", unsafe_allow_html=True)
st.sidebar.text(rev_moy)

# AMT CREDIT
st.sidebar.markdown("<u> Average Loan Amount (USD)", unsafe_allow_html=True)
st.sidebar.text(credits_moy)

# HOME PAGE - MAIN CONTENT
# Display Customer ID from Sidebar
st.write("Selected Client ID:", chk_id)

# Display Customer Information: Gender, Age, Family status, Children
st.title("**Client Informations**")

infos_client = identite_client(data, chk_id)

col1,col2 = st.columns(2)

with col1:
    st.write("**Gender:**", infos_client["CODE_GENDER"].values[0])
    st.write("**Age:** {:.0f} ans ".format(int(infos_client["DAYS_BIRTH"]/365)))
    st.write("**Family Status:**", infos_client["NAME_FAMILY_STATUS"].values[0])
    st.write("**Number of Children:** {:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))

with col2:
    st.write("**Total Income :** {:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
    st.write("**Credit Amount:** {:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
    st.write("**Credit Annuity:** {:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))

# Section Graphiques
st.subheader("Graphs")

col1,col2 = st.columns(2)

@st.cache
def load_income_population(sample):
    df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income["AMT_INCOME_TOTAL"] < 200000, :]
    return df_income

# Income Distribution Plot
with col2:
    data_income= load_income_population(data)
    fig, ax = plt.subplots(figsize=(7,3))
    sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor="k", color="royalblue", bins=10)
    ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="red", linestyle="--")
    ax.set(title="Customer Income", xlabel="Income (USD)", ylabel="")
    st.pyplot(fig)

@st.cache
def load_age_population(data):
    data_age=round((data["DAYS_BIRTH"]/365), 2)
    return data_age

# Age Distribution Plot
with col1:
    data_age = load_age_population(data)
    fig, ax = plt.subplots(figsize=(7,3))
    sns.histplot(data_age, edgecolor="k", color="royalblue", bins=20)
    ax.axvline(int(infos_client["DAYS_BIRTH"].values/360), color="red", linestyle= "--")
    ax.set(title="Customer Age", xlabel="Age (Year)", ylabel="")
    st.pyplot(fig)

# Customer Solvability Display
st.title("**Client File Analysis**")
prediction = load_prediction(sample, chk_id, clf)
st.write("**Default probability: ** {:.0f} %".format(round(float(prediction)*100, 2)))

# Compute decision according to the best threshold
if prediction <= 0.35:
    decision = "<font color = 'green'> ** LOAN GRANTED ** </font>"
else:
    decision = "<font color = 'red'> ** LOAN REJECTED ** </font>"

st.write("**Decision:**", decision, unsafe_allow_html=True)

st.markdown("<u>Customer Data:</u>", unsafe_allow_html=True)
st.write(identite_client(data, chk_id))