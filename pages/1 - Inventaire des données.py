import streamlit as st
import pandas as pd

st.title('Inventaire des données')


st.subheader("Source des données")

st.markdown("<strong>Le jeu de données est issu de la plateforme Kaggle</strong>. "
            "Aucune mention n’est fournie concernant la source de ces données et notamment pour quelle compagnie aérienne cette étude a été faite.", unsafe_allow_html=True)

st.subheader("Structure globale du fichier de données")

st.markdown("<strong>Le jeu de données comporte 129880 lignes</strong>", unsafe_allow_html=True)

st.markdown("<table>"
            "<caption>Source : https://www.kaggle.com/datasets/johndddddd/customer-satisfaction"
            "<thead>"
            "<tr>"
            "<td><strong>Typologie de données</strong></td>"
            "<td><strong>Description</strong></td>"
            "</tr>"
            "</thead>"
            "<tr>"
            "<td>Clients</td>"
            "<td>Age et genre</td>"
            "</tr>"
            "<tr>"
            "<td>Business</td>"
            "<td>Type de clients (loyal ou disloyal)</td>"
            "</tr>"
            "<tr>"
            "<td>Données questionnaires</td>"
            "<td>14 champs correspondant aux questions posées à propos des services de la compagnie aérienne et une donnée binaire correspondant à la satisfaction ou non du client (satisfait ou état neutre/non satisfait)</td>"
            "</tr>"
            "<tr>"
            "<td>Voyages</td>"
            "<td>Type of travel, class, distance vol et retard du vol</td>"
            "</tr>"
            "</table>", unsafe_allow_html=True)

st.write("")
st.subheader("Aperçu des données")

df = pd.read_csv("data/satisfaction.csv", sep = ";", index_col = 'id')
st.dataframe(df)