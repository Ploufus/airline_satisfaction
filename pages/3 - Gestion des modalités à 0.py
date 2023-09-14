import pandas as pd
import streamlit as st
from PIL import Image

df_modalite_0 = pd.read_csv("download/df_modalite_0.csv", sep=",")
df_voisins = pd.read_csv("download/services_voisins_123.csv", sep=",")
df_indices = pd.read_csv("download/services_indices_123.csv", sep=",")
df = pd.read_csv("data/df_corrige_0.csv", sep=",")

# Modification du nom de la colonne
df_modalite_0 = df_modalite_0.rename(columns = {'Unnamed: 0': 'service'})
df_voisins = df_voisins.rename(columns = {'Unnamed: 0': 'service'})
df_indices = df_indices.rename(columns = {'Unnamed: 0': 'service'})

st.title('Gestion des modalités à 0')

tab1, tab2, tab3 = st.tabs(["Contexte", "Calcul des voisins", "Génération des notes"])


with tab1:
    st.header("Contexte")
    st.write("")
    st.markdown("Lors de la distribution, nous avons vu que les modalités à « 0 » des variables qualitatives ordinales"
            " ne semblaient ne pas suivre la distribution. Ces modalités peuvent être considérées comme des anomalies,"
            " sans doute parce que des clients n’ont pas renseigné ce question. Mais à ce stade, il ne s’agit que d’une hypothèse.",
            unsafe_allow_html=True)

    st.markdown("Nous avons compté les lignes comportant au moins un 0 dans les réponses aux variables qualitatives nominales."
            "<strong> Leur nombre est de 10 268 ce qui correspond à environ 7,9% des données</strong>.",unsafe_allow_html=True)

    st.write("")

    st.markdown("<h6>Nombre de modalité à 0 par service</h6>", unsafe_allow_html=True)
    st.table(df_modalite_0)
    st.write("")
    st.markdown("<strong>L'approche est d'apporter une note à ses modalités à '0' par un système de filter collaboratif.</strong>",
                unsafe_allow_html=True)



with tab2:
    st.header("Calcul des voisins")
    st.write("")

    st.markdown("<h6>Tableau des plus proches voisins selon la distance et l'indice par service</h6>", unsafe_allow_html=True)
    df_filter_coll = pd.concat([df_voisins,df_indices.iloc[:,1:]], axis = 1)
    st.table(df_filter_coll)

    st.markdown("Par exemple, pour le service « seat_comfort » la note affectée sera calculée comme suit :"
            ""
            "<strong>[(1/0,055) x note_ food_and_drink  + (1/0,082) x note_ inflight_entertainment + (1/0,83) x note_ d_a_time_convenient]"
            " /[ (1/0,055) + (1/0,082) + (1/0,083)]</strong>.",unsafe_allow_html=True)

with tab3:
    st.header("Génération des notes")
    st.write("")
    st.markdown("<h6>Dataframe corrigé des modalités à 0</h6>",
                unsafe_allow_html=True)
    st.dataframe(df)
    st.write("")
    st.markdown("<h6>Distribution des variables ordinales par modalité corrigé des modalités à 0.</h6>",
                unsafe_allow_html=True)
    image = Image.open(r"img\variables_services_0.png")
    st.image(image, width=800)