import streamlit as st


st.image("https://www.eficiens.com/assets/uploads/2023/03/Onboarding-client-assurance1-1140x642.jpg", width=800)

st.title('Présentation du satisfaction airline')



st.header("Objectif du projet")
st.markdown("Le projet choisi a comme finalité de déterminer <strong>les facteurs de la satisfaction clients d'une compagnie aérienne<strong>.",
            unsafe_allow_html=True)
st.write(" ")
st.header("Enjeux du projet")
st.markdown("<strong>Ce projet a comme intérêts de mettre en application</strong> différentes méthodes d’exploration,"
         " de dataviz’, de modélisations supervisées et non supervisées, et autres techniques afin d’embrasser"
         " <strong>l’ensemble des techniques de machine learning et deep learning les plus communément utilisées.</strong>",
            unsafe_allow_html=True)
st.write(" ")
st.header("Plan de la présentation")
st.write("La présentation est découpée en cing parties :")
st.markdown("<li>Inventaire des données</li>", unsafe_allow_html=True)
#st.write("Cette partie liste les sources de données et liste la typologie des données présentes.")
st.markdown("<li>Distribution des données</li>", unsafe_allow_html=True)
#st.write("Ici, on explorera les données de façon univariée ou bivariée suivant qu’elles soient quantitatives"
#         " ou qualitatives et on établira les corrélations.")
st.markdown("<li>Gestion des modalités à '0'</li>", unsafe_allow_html=True)
#st.write("Cette partie donne la modalité pour créer un dataframe corrigé de certaines erreurs de collecte.")
st.markdown("<li>Pré-processing, feature engineering et modélisation</li>", unsafe_allow_html=True)
#st.write("Cette partie énumère le pipeline des traitements avant modélisation"
#         " et liste les résultats des modélisations mises en oeuvre.")
st.markdown("<li>Conclusion et analyse</li>", unsafe_allow_html=True)
#st.write("Cette dernière partie apportera des conclusions sur les traitements"
#         " et la modélisation ainsi que des analyses sur diverses approches.")