import streamlit as st
from PIL import Image

image = Image.open(r"img\onboarding.jpg")
st.image(image, width=800)

st.title('Conclusion et analyse')

st.write("")

st.header("Conclusion de l'étude")
#with st.expander("Pipeline"):
st.subheader("Df brut versus df corrigé des modalités à « 0 »")

st.markdown("Le constat de base concernant les modalités à « 0 », estimant que cela pourrait être une anomalie,"
            " s’avère erroné puisque le jeu sans correction des modalités à « 0 » a obtenu de meilleurs résultats."
            " La modalité « 0 » porte une information ; <strong>sans savoir comment les questions ont été présentées,"
            " il est difficile à ce stade d’établir d’autres conclusions.</strong>"
            , unsafe_allow_html=True)
st.markdown("Cette situation nous enseigne qu’il faut éviter d’établir des conclusions hâtives. "
            "D’une part, il est donc important de bien connaître le sourcing des données afin d’éviter des interprétations."
            " D’autre part, il est préférable de laisser « parler » les données puisqu’on est dans des jeux"
            " de données multidimensionnels et qu’un humain n’a pas la capacité à pouvoir apprécier l’ensemble"
            " des informations portées par ces jeux de données. Il est donc préférable de valider ou non son intuition par une étude."
            , unsafe_allow_html=True)
st.write("")
st.subheader("Modélisation")

st.markdown("Nous avons obtenu de très bons résultats. Les meilleurs résultats ont été obtenus avec le random forest"
            " bien que le perceptron multi couches  et les modèles en entrée à 32 neurones soient très proches."
            , unsafe_allow_html=True)
st.write("")
st.markdown("<u>Tableau récapitulatif des résulats en considérant la transformation dummies</u>"
            , unsafe_allow_html=True)
st.markdown("<table>"
               "<thead>"
               "<tr>"
               "<td><strong>Modéle</strong></td>"
               "<td><strong>Dataframe</strong></td>"
               "<td><strong>Classe 0, Score F1</strong></td>"
               "<td><strong>Classe 1, Score F1</strong></td>"
               "</tr>"
               "</thead>"
               "<tr>"
               "<td>Régression logistique</td>"
               "<td>Df_brut</td>"
               "<td>0,8948</td>"
               "<td>0,9133</td>"
               "</tr>"
               "<tr>"
               "<td>Perceptron 32 couches</td>"
               "<td>Df_brut</td>"
               "<td>0,9486</td>"
               "<td>0,9568</td>"
               "</tr>"
               "<tr>"
               "<td><strong>Random Forest</td>"
               "<td><strong>Df_brut</strong></td>"
               "<td><strong>0,9560</strong></td>"
               "<td><strong>0,9635</strong></td>"
               "</tr>"
               "<tr>"
               "<tr>"
               "<td>Random Forest</td>"
               "<td>Df_sans modalité à 0</td>"
               "<td>0,9554</td>"
               "<td>0,9609</td>"
               "</tr>"
               "<td><strong>Random Forest</strong></td>"
               "<td><strong>Df_business</strong></td>"
               "<td><strong>0,9592</strong></td>"
               "<td><strong>0,9703</strong></td>"
               "</tr>"
               "</table>", unsafe_allow_html=True)
st.write("")


st.write("")
st.markdown("<h6>Autres approches ?</h6>", unsafe_allow_html=True)
st.markdown("Cela aurait été intéressant d’associer les services entre eux en n_uplets"
            " afin d’établir des règles de satisfaction."
            , unsafe_allow_html=True)

#st.markdown("Nous avons soulevé aussi l’opposition concernant la satisfaction entre les 2 types de voyageurs voyageant pour le business ou non. "
#            "<strong>En faisant un modèle dissociant les voyageurs business des autres,"
#            " ce modèle s’est avéré encore meilleur.</strong>"
#            , unsafe_allow_html=True)
st.markdown("Concernant ces bons résultats, le questionnaire est déclaratif par définition, à l’issue des questions"
            " sur la satisfaction de chaque service, une question sur le niveau de satisfaction a été demandée."
            " Par conséquent, <strong>cela semble assez évident d’obtenir des scores très élevés étant donné la relation directe"
            " entre les features et le label</strong>."
            , unsafe_allow_html=True)


st.write("")

st.header("Signaux faibles")

st.markdown("Ces relations directes entre features et label ne sont pas suffisantes. "
            "En effet, il aurait été intéressant d’obtenir des signaux faibles en ajoutant des données telles que :"
            "<li>L’utilisation de tel ou tel template de réservation en ligne</li>"
            "<li>Une offre de service différente en vol (boissons, repas, etc.)</li>"
            "<li>Des discours différents pour les équipes support online</li>"
            "<li>La météo</li>"
            , unsafe_allow_html=True)
st.markdown("Avec ces données, nous aurions pu obtenir l’impact d’un template ou d’un type de repas en particulier."
            , unsafe_allow_html=True)

st.write("")

st.header("Pilotage du projet")
st.markdown("Ces données auraient permis de déceler l’importance de données métiers et d’identifier des leviers d’amélioration. "
            , unsafe_allow_html=True)
st.markdown("<strong>Chaque segment de données doit être mis en relation avec des équipes afin"
            " qu’on puisse interagir dans notre cas avec la satisfaction clients.</strong>"
            , unsafe_allow_html=True)
st.markdown("Dans un pilotage de projet de data science comme celui-ci, lorsqu’on met en place des modèles,"
            " des segments de features doivent être mis en relation direct avec des équipes métiers. "
            "C’est très important car cela permet de jouer directement avec le modèle et d’être dans une amélioration constante du modèle; dans notre cas, d’améliorer les services de la compagnie aérienne. "
            , unsafe_allow_html=True)

st.header("Réponse aux enjeux")
st.markdown("<strong>Ce projet a répondu à mes attentes puisque j'ai pu mettre en application la pratique du Python dans un cadre"
            " d'un projet de data science.</strong>"
            , unsafe_allow_html=True)