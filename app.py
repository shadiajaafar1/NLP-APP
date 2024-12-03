import pandas as pd
import streamlit as st
import altair as alt
import warnings
warnings.filterwarnings('ignore')
from streamlit_option_menu import option_menu
from joblib import load
import os
from PIL import Image
    
#Datos-------
def leer_archivo(nombre_archivo):
    try:
        with open(nombre_archivo, "r", encoding="utf-8") as file:
            lineas = file.readlines()
    except UnicodeDecodeError:
        with open(nombre_archivo, "r", encoding="latin1") as file:
            lineas = file.readlines()

    # Convertir a DataFrame y limpiar contenido
    df = pd.DataFrame(lineas, columns=["Contenido"])
    df["Contenido"] = df["Contenido"].str.strip()
    return df

try:
    train = leer_archivo("thai_nlp/train.txt")
    train_label = leer_archivo("thai_nlp/train_label.txt")
    test = leer_archivo("thai_nlp/test.txt")
    test_label = leer_archivo("thai_nlp/test_label.txt")
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()
    
def corpus(text):
    text_list = text.split()
    return text_list

train["Label"] = train_label["Contenido"]
test["Label"] = test_label["Contenido"]

df = pd.concat([train, test], ignore_index=True)


df['comment_length'] = df["Contenido"].astype(str).apply(len)
df['Word_count'] = df["Contenido"].apply(lambda x: len(str(x).split()))


#APP----------

custom_style = """
<style>
/* Cambiar el color de fondo de todo el dashboard */
[data-testid="stAppViewContainer"] {
    background-color: #E5EFF0;
}

/* Ajustar el fondo del contenido principal */
[data-testid="stApp"] {
    background-color: #E5EFF0;
    padding: 0;
    display: flex;
    justify-content: center;
    align-items: center;
}

.stAppViewBlockContainer {
        padding-top: 0rem;
            }

.sidebar .sidebar-content h2 {
        color: #ff6347; /* Cambia el color aquí */
        font-size: 20px; /* Ajusta el tamaño */
        font-weight: bold; /* Negrita */
    }

</style>
"""

# Aplicar el CSS combinado
st.markdown(custom_style, unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",  # Título del menú
        options=["Dashboard"],  # Opciones del menú
        icons=["bar-chart"],  # Iconos (de https://icons.getbootstrap.com/)
        menu_icon="cast",  # Icono del menú principal
        default_index=0,  # Página por defecto
    )

options = ["Positivo", "Negativo", "Neutral", "Pregunta"]
selection = st.sidebar.multiselect(
    "Selecciona el Sentimiento", options, default=options
)


#Filtros-----
label_mapping = {
    "Positivo": "pos",
    "Negativo": "neg",
    "Neutral": "neu",
    "Pregunta": "q",
}

if selected == "Dashboard":
    
    st.title("Análisis de Sentimiento de Comentarios")

    if selection:
        mapped_selection = [label_mapping[opt] for opt in selection]
        filtered_df = df[df["Label"].isin(mapped_selection)]
    else:
        filtered_df = df

    #Métricas--------
    puntos = {
    'pos': 2,
    'neu': 1,
    'neg': 0,
    'q': 0 
    }
    filtered_df['Points'] = filtered_df['Label'].map(puntos)
    grade = round(filtered_df['Points'].sum() / (len(filtered_df) * 2), 2)
    moda = filtered_df['Label'].mode().iloc[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Número de comentarios", len(filtered_df))
    col2.metric("Score de sentimiento", grade)
    col3.metric('Sentimiento Modal', moda)


    # Gráfico de distribución de etiquetas (dona) en la primera columna
    label_counts = filtered_df["Label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]
    fig_pie = alt.Chart(label_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Label", type="nominal"),
            tooltip=["Label", "Count"]
        ).properties(
            title="Distribución de Etiquetas"
        )
    st.altair_chart(fig_pie, use_container_width=True)

    # Histograma de longitud de comentarios en la segunda columna
    fig_hist = alt.Chart(filtered_df).mark_bar().encode(
            alt.X("comment_length", bin=alt.Bin(maxbins=30), title="Longitud del Comentario (caracteres)"),
            alt.Y("count()", title="Frecuencia"),
            tooltip=["count()"]
        ).properties(
            title="Distribución de la longitud de los comentarios"
        )
    st.altair_chart(fig_hist, use_container_width=True)

    # Histograma del número de palabras en la tercera columna

    fig_word_hist = alt.Chart(filtered_df).mark_bar().encode(
            alt.X("Word_count", bin=alt.Bin(maxbins=30), title="Número de palabras"),
            alt.Y("count()", title="Frecuencia"),
            tooltip=["count()"]
        ).properties(
            title="Distribución de comentarios según el número de palabras"
        )
    st.altair_chart(fig_word_hist, use_container_width=True)



    def visualize(col):
        # Boxplot
        boxplot = alt.Chart(df).mark_boxplot().encode(
            x=alt.X('Label:N', title="Sentiment Label"),
            y=alt.Y(f'{col}:Q', title=col),
            color='Label:N'
        ).properties(
            title=f"Boxplot of {col} by Sentiment",
            width=400,
            height=300
        )
        
        # KDE plot (Density Plot)
        kdeplot = alt.Chart(df).transform_density(
            col, 
            groupby=['Label'],
            as_=[col, 'density']
        ).mark_area(opacity=0.5).encode(
            x=alt.X(f'{col}:Q', title=col),
            y=alt.Y('density:Q', title='Density'),
            color='Label:N'
        ).properties(
            title=f"KDE Plot of {col} by Sentiment",
            width=400,
            height=300
        )
        
        return boxplot, kdeplot

    features = ['comment_length', 'Word_count']

    for feature in features:
        
        boxplot, kdeplot = visualize(feature)

        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(boxplot, use_container_width=True)
        with col2:
            st.altair_chart(kdeplot, use_container_width=True)


    





