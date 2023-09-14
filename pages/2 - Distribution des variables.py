import pandas as pd
import streamlit as st
from PIL import Image

# Chargement des fichiers
df = pd.read_csv("data/satisfaction.csv", sep = ";", index_col = 'id')
df = df.rename(columns = {'satisfaction_v2' : 'satisfaction', 'Gender' : 'gender', 'Customer Type' : 'customer_type', 'Age' : 'age',
                          'Type of Travel' : 'type_travel', 'Class' : 'class', 'Departure/Arrival time convenient' : 'd_a_time_convenient',
                          'Departure Delay in Minutes' : 'd_delay_minutes','Arrival Delay in Minutes' : 'a_delay_minutes',
                          'Flight Distance' : 'flight_distance','Seat comfort' : 'seat_comfort','Food and drink' : 'food_and_drink',
                          'Gate location' : 'gate_location','Inflight wifi service' : 'inflight_wifi_service', 'Inflight entertainment' : 'inflight_entertainment',
                          'Online support' : 'online_support', 'Ease of Online booking' : 'ease_online_booking', 'On-board service' : 'on_board_service',
                          'Leg room service' : 'leg_room_service', 'Baggage handling' : 'baggage_handling', 'Checkin service' : 'checkin_service', 'Cleanliness' : 'cleanliness',
                          'Online boarding' : 'online_boarding'
                          })
df_tab2 = pd.read_csv("download/df_corr_table_contingence.csv", sep=",")
df_tab2 = df_tab2[['var1', 'var2', 'p_valeur', 'v_cramer']]
df_tab1 = pd.read_csv("download/df_corr_table_contingence_int.csv", sep=",")
df_tab1 = df_tab1[['var1', 'var2', 'p_valeur', 'v_cramer']]

st.title('Distribution des données')


tab1, tab2, tab3 = st.tabs(["Distribution univariée", "Distribution bivariée", "Conclusion"])


with tab1:
    st.header("Distribution univariée")
    st.subheader("Variables quantitatives")

    # Définition des groupes de colonnes
    column_cat_nosurvey = ['satisfaction','gender','customer_type','type_travel','class']
    columns_survey = df.columns[7:21]
    # Distribution univariée, variables quantitatives
    st.write(round(df[['age','flight_distance','d_delay_minutes','a_delay_minutes']].describe().T,0))

    st.write("")

    # Distribution univarée de la variable flight_distance
    st.markdown("<h6>Distribution de la variable flight_distance</h6>", unsafe_allow_html=True)
    image = Image.open(r"img\boxplot_flight.png")
    st.image(image, width=600)

    st.write("Ces valeurs extrêmes seront gérés dans la partie feature engineering en prenant le dernier quantile")
    code = '''quantile_flight_distance = df['flight_distance'].quantile(0.99)'''
    st.code(code,language = 'python')
    st.write("")

    st.subheader("Variables qualitatives")

    st.markdown("<h6>Distribution des variables qualitatives</h6>", unsafe_allow_html=True)
    image = Image.open(r"img\variables_qualitatives.png")
    st.image(image, width=600)

    st.subheader("Variables qualitatives ordinales")
    image = Image.open(r"img\variables_services.png")
    st.image(image, width=600)
    image = Image.open(r"img\cumul_reponses.png")
    st.image(image, width=600)



    st.write("")
    st.write("")

with tab2:
    st.header("Distribution bivariée")
    st.subheader("Variables quantitatives")

    image = Image.open(r"img\matrice_correlation.png")
    st.image(image, width=400)
    st.markdown("<h6>Distribution des variables quantitatives suivant la satisfaction client</h6>", unsafe_allow_html=True)
    st.write("En calculant le test Anova avec la variable satisfaction, les variables quantitatives présente une faible corrélation")
    st.write("")

    st.subheader("Variables qualitatives")
    st.markdown("<h6>Distribution des variables qualitatives entre elles</h6>", unsafe_allow_html=True)
    st.dataframe(df_tab1.iloc[5:9,:])
    st.markdown("<h6>Distribution des variables qualitatives suivant la satisfaction client</h6>", unsafe_allow_html=True)
    image = Image.open(r"img\variables_qualitatives_bivariees.png")
    st.image(image, width=600)
    st.table(df_tab1.iloc[0:4,:])
    st.write("")

    st.subheader("Variables qualitatives ordinales")
    st.markdown("<h6>Distribution des variables qualitatives ordinales suivant la satisfaction client</h6>", unsafe_allow_html=True)
    st.table(df_tab2.iloc[4:18,:])

with tab3:
    st.subheader("Conclusion de l'analyse de la distribution")
    st.markdown("<ul>"
                "<li>Les variables « d_delay_minutes », « a_delay_minutes » et « flight_distance» ont des valeurs extrêmes</li>"
                "<li>Corrélation forte (0,97), entre « d_delay_minutes » et « a_delay_minutes »</li>"
                "<li>La variable à prédire « satisfaction » est peu corrélée avec les variables quantitatives</li>"  
                "<li>La variable à prédire « satisfaction » est significativement corrélée avec les services correspondant"
                " au la distraction en vol, le confort du siège et ceux permettant de facilement réserver</li>"
                "<li>Voyager pour le business offre plus de satisfaction dû sans doute aux services associés</li>"
                "<li>La modalité « 0 » des variables quantitatives ordinales semblent être une anomalie</li>"
                "</ul>", unsafe_allow_html=True)
    st.write("")
    st.subheader("Actions qui en découlent")
    st.markdown("<ul>"
                "<li>Supprimer la variable corrélée « a_delay_minutes »</li>"
                "<li>Limiter l'impact des valeurs extrêmes en affectant aux valeurs extrêmes le dernier quantile</li>"
                "<li>Préparer une étude pour les clients business</li>"
                "<li>Corriger les clients présentant des modalités à '0' pour leur affecter une note</li>"
                "</ul>", unsafe_allow_html=True)




