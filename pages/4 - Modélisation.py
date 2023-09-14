import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
from PIL import Image
import lime
import lime.lime_tabular
import tensorflow
import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from IPython.display import HTML
from joblib import dump,load


st.title('Modélisation')

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
st.write("")
st.subheader("Méthodologie")
with st.expander("Méthodologie"):
    st.markdown("<ul>"
                "<li>Appliquer le pré-processing et feature engineering classique</li>"
                "<li>Entrainer un premier modèle de régression logistique sur les données brutes</li>"
                "<li>Entrainer ce même modèle sur des données transformées en dummies</li>"
                "<li>Conclure sur le meilleur pré-processing de transformation</li>"
                "<li><strong>Déterminer les meilleurs hyper paramètres sur les différents modèles"
                " et sur les dataframes brut et corrigé des modalités à '0'</strong></li>"
                "<li>Entrainer les modèles sur ces hyper paramètres</li>"
                "<li>Entrainer les modèles sur des dataframes alternatifs</li>"
                "</ul>", unsafe_allow_html=True)
    st.write("")
    image_methodologie = Image.open(r"img\methodologie_modelisation.png")

    st.image(image_methodologie, width=550, caption="Méthodologie")


st.write("")
st.write("")
st.subheader("Pré-processing et feature engineering")
with st.expander("Pipeline"):
    st.markdown("<h6>Pré-processing</h6>",
                unsafe_allow_html=True)
    st.markdown("<ul>"
                "<li>Création d'un dataframe corrigé des valeurs à 0</li>"
                "<li>Suppression de la variable a_delay_minutes corrélée à d_delay_minutes et ayant des valeurs NaN</li>"
                "<li>Affectation du dernier quantile aux valeurs extrêmes (d_delay_minutes, flight_distance)</li>"  
                "<li>Création du df des features data et de la série target</li>"
                "<li>Partage des données d'entrainement et de test</li>"
                "<li>Feature engineering</li>"
                "<li>Normalisation des données pour la régression logistique et le deep learning</li>"
                "</ul>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<h6>Feature engineering</h6>",
                unsafe_allow_html=True)
    st.markdown("<table>"
                "<thead>"
                "<tr>"
                "<td><strong>Variable</strong></td>"
                "<td><strong>Processing</strong></td>"
                "<td><strong>Modalité</strong></td>"
                "</tr>"
                "</thead>"
                "<tr>"
                "<td>satisfaction</td>"
                "<td>en 0/1</td>"
                "<td>1:satisfied, 0:neutral or dissatisfied</td>"
                "</tr>"
                "<tr>"
                "<td>genre</td>"
                "<td>en 0/1</td>"
                "<td>1:female, 0:male</td>"
                "</tr>"
                "<tr>"
                "<td>customer_type</td>"
                "<td>en 0/1</td>"
                "<td>1:Loyal Customer, 0:disloyal Customer</td>"
                "</tr>"
                "<tr>"
                "<td>type_travel</td>"
                "<td>en 0/1</td>"
                "<td>1:Business travel, 0:Personal Travel</td>"
                "</tr>"
                "<tr>"
                "<td>class</td>"
                "<td>en dummies</td>"
                "<td>0:Business class, 1:Eco, 2:Eco Plus</td>"
                "</tr>"
                "<tr>"
                "<td>Services</td>"
                "<td>en dummies</td>"
                "<td>0 à 5 ou 1 à 5 suivant le df</td>"
                "</tr>"
                "</table>", unsafe_allow_html=True)
    st.write("")
st.write("")
st.write("")

st.subheader("Critères en entrée")
col1, col2 = st.columns(2)

with col1:
    dummies = st.radio(
        "Avec ou sans dummies",
        ('sans_dummies', 'avec_dummies'))
with col2:
    df_brut_ou_0 = st.radio(
        "Choix df : df brut ou df corrigé des modalités à 0",
        ('df_brut', 'df_0'))



if df_brut_ou_0 == 'df_brut':
    # Suppression de la variable a_delay_minutes
    df = df.drop(['a_delay_minutes'], axis=1)
else:
    df = pd.read_csv("data/df_corrige_0.csv", sep=",", index_col='id')

# Valeur extrême, fixe la valeur au dernier centile pour les clients dépassant cette valeur
decide_flight_distance = df['flight_distance'].quantile(0.99)
df['flight_distance'] = df['flight_distance'].apply(lambda x:decide_flight_distance if x>decide_flight_distance else x)
quantile_d_delay_minutes = df['d_delay_minutes'].quantile(0.99)
df['d_delay_minutes'] = df['d_delay_minutes'].apply(lambda x:quantile_d_delay_minutes if x>quantile_d_delay_minutes else x)


# Transformation des variables qualitatives en variables category
df['satisfaction'] = df['satisfaction'].apply(lambda x:1 if x == 'satisfied' else 0)
df['gender'] = df['gender'].apply(lambda x:0 if x == 'Male' else 1)
df['customer_type'] = df['customer_type'].apply(lambda x:1 if x == 'Loyal Customer' else 0)
df['type_travel'] = df['type_travel'].apply(lambda x:0 if x == 'Business travel' else 1)

fontsizeplot = 20

if dummies == 'sans_dummies':
    df = pd.get_dummies(data = df, columns = ['class'], prefix = ['class'])
    target = df['satisfaction']
    data = df.drop(['satisfaction'], axis=1)
    # cas par défaut
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)  # 66
    # Jeux de train/test, cas perceptron
    X_train_kernel, X_test_kernel, y_train_kernel, y_test_kernel = train_test_split(data, target, test_size=0.33, random_state=42)
    # Normalisation pour la régression logistique et le perceptron
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_linear = scaler.transform(X_train)
    X_test_linear = scaler.transform(X_test)
    X_train_kernel_linear = scaler.transform(X_train_kernel)
    X_test_kernel_linear = scaler.transform(X_test_kernel)
    # Prédiction
    if df_brut_ou_0 == 'df_brut':
        clf_rl_045 = load('model/clf_rl_045.joblib')
        y_pred_rl = clf_rl_045.predict(X_test_linear)
        clf_rf = load('model/clf_rf.joblib')
        estimator = clf_rf.estimators_[5]
        y_pred_rf = clf_rf.predict(X_test)
        model = keras.models.load_model("model/model2_df.h5")

        rl_fig = plt.figure(figsize=(40, 16))
        plt.bar(df.iloc[:, 1:].columns, height=clf_rl_045.coef_[0])
        plt.rcParams['font.size'] = fontsizeplot
        plt.xticks(rotation=90)
        rf_fig = plt.figure(figsize=(40, 16))
        plt.bar(df.iloc[:, 1:].columns, height=clf_rf.feature_importances_)
        plt.rcParams['font.size'] = fontsizeplot
        plt.xticks(rotation=90)
    else:
        clf_rl_058 = load('model/clf_rl_058.joblib')
        y_pred_rl = clf_rl_058.predict(X_test_linear)
        clf_rf_0 = load('model/clf_rf_0.joblib')
        estimator = clf_rf_0.estimators_[5]
        y_pred_rf = clf_rf_0.predict(X_test)
        model = keras.models.load_model("model/model4_df_0.h5")
        rl_fig_0 = plt.figure(figsize=(40, 16))
        plt.bar(df.iloc[:, 1:].columns, height=clf_rl_058.coef_[0])
        plt.rcParams['font.size'] = fontsizeplot
        plt.xticks(rotation=90)
        rf_fig_0 = plt.figure(figsize=(40, 16))
        plt.bar(df.iloc[:, 1:].columns, height=clf_rf_0.feature_importances_)
        plt.rcParams['font.size'] = fontsizeplot
        plt.xticks(rotation=90)
    # Définition du graph
    is_global = "clf_rl_045" in globals()

else:
    df_dummies = pd.get_dummies(round(df, 0), columns=['seat_comfort', 'd_a_time_convenient',
                                                       'food_and_drink', 'gate_location', 'inflight_wifi_service',
                                                       'inflight_entertainment', 'online_support',
                                                       'ease_online_booking',
                                                       'on_board_service', 'leg_room_service', 'baggage_handling',
                                                       'checkin_service', 'cleanliness', 'online_boarding', 'class'])
    target = df_dummies['satisfaction']
    data = df_dummies.drop(['satisfaction'], axis=1)
    # cas par défaut
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)  # 66
    # Jeux de train/test, cas perceptron
    X_train_kernel, X_test_kernel, y_train_kernel, y_test_kernel = train_test_split(data, target, test_size=0.33, random_state=42)
    # Normalisation pour la régression logistique et le perceptron
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_linear = scaler.transform(X_train)
    X_test_linear = scaler.transform(X_test)
    X_train_kernel_linear = scaler.transform(X_train_kernel)
    X_test_kernel_linear = scaler.transform(X_test_kernel)
    # Prédiction
    if df_brut_ou_0 == 'df_brut':
        clf_rl_045_dummies = load('model/clf_rl_045_dummies.joblib')
        y_pred_rl = clf_rl_045_dummies.predict(X_test_linear)
        clf_rf_dummies = load('model/clf_rf_dummies.joblib')
        estimator = clf_rf_dummies.estimators_[5]
        y_pred_rf = clf_rf_dummies.predict(X_test)
        model = keras.models.load_model("model/model3_df_dummies.h5")
        # Définition du graph reg logistique
        rl_fig_dummies = plt.figure(figsize=(40, 16))
        plt.bar(df_dummies.iloc[:, 1:].columns, height=clf_rl_045_dummies.coef_[0])
        plt.rcParams['font.size'] = fontsizeplot
        plt.xticks(rotation=90)
        rf_fig_dummies = plt.figure(figsize=(40, 16))
        plt.bar(df_dummies.iloc[:, 1:].columns, height=clf_rf_dummies.feature_importances_)
        plt.rcParams['font.size'] = fontsizeplot
        plt.xticks(rotation=90)
    else:
        clf_rl_058_dummies = load('model/clf_rl_058_dummies.joblib')
        y_pred_rl = clf_rl_058_dummies.predict(X_test_linear)
        clf_rf_0_dummies = load('model/clf_rf_0_dummies.joblib')
        y_pred_rf = clf_rf_0_dummies.predict(X_test)
        estimator = clf_rf_0_dummies.estimators_[5]
        model = keras.models.load_model("model/model1_df_0_dummies.h5")
        # Définition du graph reg logistique
        rl_fig_0_dummies = plt.figure(figsize=(40, 16))
        plt.bar(df_dummies.iloc[:, 1:].columns, height=clf_rl_058_dummies.coef_[0])
        plt.rcParams['font.size'] = fontsizeplot
        plt.xticks(rotation=90)
        rf_fig_0_dummies = plt.figure(figsize=(40, 16))
        plt.bar(df_dummies.iloc[:, 1:].columns, height=clf_rf_0_dummies.feature_importances_)
        plt.rcParams['font.size'] = fontsizeplot
        plt.xticks(rotation=90)

y_pred_kernel = model.predict(X_test_kernel_linear)
test_pred = y_pred_kernel
y_test_class = y_test_kernel
y_pred_class = np.argmax(test_pred, axis=1)

st.write("")
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["Régression logistique", "Random forest", "Deep learning", "df alternatif"])

with tab1:
    st.header("Régression logistique")
    st.markdown("<h6>Feature engineering spécifique</h6>",unsafe_allow_html=True)
    st.markdown("Dans le cas des modèles linéaires, afin d’éviter l’influence des variables ayant une amplitude forte"
                " comme « flight_distance » ou « d_a_time_convenient », il est préférable de normaliser les données."
                " On utilisera la fonction <strong>StandardScaler</strong>.",unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.subheader("Résultats")

    col_rl1, col_rl2 = st.columns(2)

    with col_rl1:
        st.markdown("<h6>Matrice de confusion</h6>", unsafe_allow_html=True)
        st.write(pd.crosstab(y_test, y_pred_rl, rownames=['Observé'], colnames=['Prédit']))
    with col_rl2:
        st.markdown("<h6>Principales métriques</h6>", unsafe_allow_html=True)
        st.write(round(pd.DataFrame(classification_report(y_test, y_pred_rl, output_dict=True)).iloc[0:3, 0:2], 4))
    st.write("")

    st.subheader("Importance des coefficients")

    if dummies == 'sans_dummies':
        if df_brut_ou_0 == 'df_brut':
            st.pyplot(rl_fig)
        else:
            st.pyplot(rl_fig_0)
    else:
        if df_brut_ou_0 == 'df_brut':
            st.pyplot(rl_fig_dummies)
        else:
            st.pyplot(rl_fig_0_dummies)

    if (dummies == 'avec_dummies') & (df_brut_ou_0 == 'df_brut'):
        st.subheader("Courbe de roc")
        image = Image.open(r"img\rl_courbe_roc.png")
        st.image(image, width=400, caption='Courbe de roc')


with tab2:
   st.header("Random forest")
   st.subheader("Résultats")

   col_rf1, col_rf2 = st.columns(2)

   with col_rf1:
        st.markdown("<h6>Matrice de confusion</h6>",unsafe_allow_html=True)
        st.write(pd.crosstab(y_test, y_pred_rf, rownames=['Observé'], colnames=['Prédit']))
   with col_rf2:
        st.markdown("<h6>Principales métriques</h6>", unsafe_allow_html=True)
        st.write(round(pd.DataFrame(classification_report(y_test, y_pred_rf, output_dict=True)).iloc[0:3, 0:2], 4))
   st.write("")

   st.subheader("Importance des coefficients")
   if dummies == 'sans_dummies':
       if df_brut_ou_0 == 'df_brut':
           st.pyplot(rf_fig)
       else:
           st.pyplot(rf_fig_0)
   else:
       if df_brut_ou_0 == 'df_brut':
           st.pyplot(rf_fig_dummies)
       else:
           st.pyplot(rf_fig_0_dummies)

   if (dummies == 'avec_dummies') & (df_brut_ou_0 == 'df_brut'):
       st.subheader("Courbe de roc")
       image = Image.open(r"img\rf_courbe_roc.png")
       st.image(image, width=400, caption='Courbe de roc')

   st.subheader("Arbre de décision")
   st.write("L'arbre est calculé sur la base d'une itération et avec l'ensemble des features.")


   clf_rf = RandomForestClassifier(n_estimators=1,
                                   criterion='entropy',
                                   max_depth=19,
                                   max_features=None,
                                   min_samples_leaf=4,
                                   bootstrap=False,
                                   n_jobs=-1,
                                   random_state=321)
   clf_rf.fit(X_train, y_train)
   fig_tree = plt.figure(figsize=(30, 20))

   if dummies == 'sans_dummies':
        plot_tree(clf_rf.estimators_[0],
             max_depth=3,
             feature_names=df.iloc[:, 1:].columns,
             class_names=['0', '1'],
             filled=True,
             proportion=True,
             fontsize=13,
             rounded=True)
   else:
       plot_tree(clf_rf.estimators_[0],
                 max_depth=3,
                 feature_names=df_dummies.iloc[:, 1:].columns,
                 class_names=['0', '1'],
                 filled=True,
                 proportion=True,
                 fontsize=13,
                 rounded=True)
   st.pyplot(fig_tree)


with tab3:
   st.header("Deep learning")
   st.markdown("<h6>Feature engineering spécifique</h6>", unsafe_allow_html=True)
   st.markdown("Dans le cas du perceptron multicouches qui séparent les données linéairement, afin d’éviter l’influence des variables ayant une amplitude forte"
               " comme « flight_distance » ou « d_a_time_convenient », il est préférable de normaliser les données."
               " On utilisera la fonction <strong>StandardScaler</strong>.", unsafe_allow_html=True)
   st.write("")
   st.markdown("Les différents modèles de perceptron ont les paramètres suivants :", unsafe_allow_html=True)
   st.markdown("<table>"
               "<thead>"
               "<tr>"
               "<td><strong>Layer</strong></td>"
               "<td><strong>Model 1</strong></td>"
               "<td><strong>Model 2</strong></td>"
               "<td><strong>Model 3</strong></td>"
               "<td><strong>Model 4</strong></td>"
               "</tr>"
               "</thead>"
               "<tr>"
               "<td>1</td>"
               "<td>10</td>"
               "<td>14</td>"
               "<td>32</td>"
               "<td>32</td>"
               "</tr>"
               "<tr>"
               "<td>2</td>"
               "<td>8</td>"
               "<td>10</td>"
               "<td>16</td>"
               "<td>16</td>"
               "</tr>"
               "<tr>"
               "<td>3</td>"
               "<td>6</td>"
               "<td>6</td>"
               "<td>8</td>"
               "<td>8</td>"
               "</tr>"
               "<tr>"
               "<td>4</td>"
               "<td>2</td>"
               "<td>2</td>"
               "<td>2</td>"
               "<td>6</td>"
               "</tr>"
               "<td>5</td>"
               "<td>-</td>"
               "<td>-</td>"
               "<td>-</td>"
               "<td>2</td>"
               "</tr>"
               "</table>", unsafe_allow_html=True)
   st.write("")

   st.subheader("Résultats d'entrainement")

   if dummies == 'sans_dummies':
       if df_brut_ou_0 == 'df_brut':
           image_perceptron = Image.open(r"img\kernel_df.png")
       else:
           image_perceptron = Image.open(r"img\kernel_df_0.png")
   else:
       if df_brut_ou_0 == 'df_brut':
           image_perceptron = Image.open(r"img\kernel_df_dummies.png")
       else:
           image_perceptron = Image.open(r"img\kernel_df_0_dummies.png")


   st.image(image_perceptron, width=800, caption="Entrainement des perceptrons")

   st.write("")
   st.subheader("Résultats")
   col_dl1, col_dl2 = st.columns(2)

   with col_dl1:
        st.markdown("<h6>Matrice de confusion</h6>",unsafe_allow_html=True)
        st.write(pd.crosstab(y_test_class, y_pred_class, rownames=['Observé'], colnames=['Prédit']))
   with col_dl2:
        st.markdown("<h6>Principales métriques</h6>", unsafe_allow_html=True)
        st.write(round(pd.DataFrame(classification_report(y_test_class,y_pred_class,output_dict=True)).iloc[0:3,0:2],4))

   st.write("")



with tab4:
    st.header("Dataframe alternatifs")
    st.markdown("Des modèles ont été entrainés sur des jeux de données alternatifs"
                " en utilisant <strong>un modèle random forest</strong>.", unsafe_allow_html=True)
    st.write("")

    if (dummies == 'avec_dummies') & (df_brut_ou_0 == 'df_brut'):
        st.subheader("df business")
        st.markdown("Ce df correspondant aux clients business", unsafe_allow_html=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.markdown("<h6>Matrice de confusion</h6>", unsafe_allow_html=True)
            matrice_confusion_business = pd.DataFrame([[7252,287],[330,10070]],columns = ['0','1'])
            st.dataframe(matrice_confusion_business)
        with col_dl2:
            st.markdown("<h6>Principales métriques</h6>", unsafe_allow_html=True)
            metrique_business = pd.DataFrame([[0.9565, 0.9723], [0.9619, 0.9683],[0.9592, 0.9703]], columns=['0', '1'], index = ['precision','recall','f1-score'])
            st.dataframe(metrique_business)

        st.subheader("df sans modalité à 0")
        st.markdown("Ce df correspondant aux lignes tronquées des modalités à 0", unsafe_allow_html=True)

        st.write("")

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.markdown("<h6>Matrice de confusion</h6>", unsafe_allow_html=True)
            matrice_confusion_business = pd.DataFrame([[10680, 391], [606, 12246]], columns=['0', '1'])
            st.dataframe(matrice_confusion_business)
        with col_dl2:
            st.markdown("<h6>Principales métriques</h6>", unsafe_allow_html=True)
            metrique_business = pd.DataFrame([[0.9463, 0.9691], [0.9647, 0.9528], [0.9554, 0.9609]], columns=['0', '1'],
                                             index=['precision', 'recall', 'f1-score'])
            st.dataframe(metrique_business)
    else:
        st.write("Sélectionner le cas avec dummies et df brut")