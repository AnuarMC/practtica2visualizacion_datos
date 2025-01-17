import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Análisis Extensivo e Interactivo de la Depresión en Estudiantes",
    layout="wide"
)

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

data_path = "Student Depression Dataset.csv"
data = load_data(data_path)


def show_initial_data_overview(df):
    st.subheader("Información General del Dataset")
    st.write(f"**Total de Filas:** {df.shape[0]} | **Total de Columnas:** {df.shape[1]}")
    
    st.write("**Valores nulos por columna:**")
    missing_values = df.isna().sum()
    st.write(missing_values)
    
    st.write("**Descripción Estadística (variables numéricas):**")
    st.write(df.describe())
    
    st.write("**Vista de los primeros registros:**")
    st.dataframe(df.head(10))

def sidebar_filters(df):
    st.sidebar.title("Filtros Adicionales")

    if "City" in df.columns:
        unique_cities = df["City"].dropna().unique().tolist()
        selected_cities = st.sidebar.multiselect(
            "Selecciona Ciudad(es):",
            options=unique_cities,
            default=unique_cities
        )
    else:
        selected_cities = []
    
    if "Gender" in df.columns:
        unique_genders = df["Gender"].dropna().unique().tolist()
        selected_genders = st.sidebar.multiselect(
            "Selecciona Género(s):",
            options=unique_genders,
            default=unique_genders
        )
    else:
        selected_genders = []
    
    if "Age" in df.columns:
        min_age = int(df["Age"].min()) if not df["Age"].isna().all() else 0
        max_age = int(df["Age"].max()) if not df["Age"].isna().all() else 100
        age_range = st.sidebar.slider(
            "Rango de Edad:",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )
    else:
        age_range = (0, 100)
    
    df_filtered = df.copy()
    
    if "City" in df.columns:
        df_filtered = df_filtered[df_filtered["City"].isin(selected_cities)]
    if "Gender" in df.columns:
        df_filtered = df_filtered[df_filtered["Gender"].isin(selected_genders)]
    if "Age" in df.columns:
        df_filtered = df_filtered[(df_filtered["Age"] >= age_range[0]) & (df_filtered["Age"] <= age_range[1])]
    
    if "Depression" in df.columns:
        depression_option = st.sidebar.selectbox("Filtrar por Depresión:", ["Todos", "Con Depresión", "Sin Depresión"])
        if depression_option == "Con Depresión":
            df_filtered = df_filtered[df_filtered["Depression"] == 1]
        elif depression_option == "Sin Depresión":
            df_filtered = df_filtered[df_filtered["Depression"] == 0]
    
    return df_filtered

def plot_general_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Depression", data=df, palette="coolwarm")
    plt.title("Distribución General de Depresión")
    plt.xlabel("Depresión (0: No, 1: Sí)")
    plt.ylabel("Cantidad de Estudiantes")
    st.pyplot(plt.gcf())
    plt.close()

def plot_gender_depression(df):
    plt.figure(figsize=(8, 5))
    gender_counts = df.groupby(["Gender", "Depression"]).size().unstack(fill_value=0)
    gender_counts.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="coolwarm")
    plt.title("Distribución de Depresión por Género")
    plt.xlabel("Género")
    plt.ylabel("Cantidad de Estudiantes")
    st.pyplot(plt.gcf())
    plt.close()

def plot_age_depression(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Depression", y="Age", data=df, palette="coolwarm")
    plt.title("Distribución de Edad por Niveles de Depresión")
    plt.xlabel("Depresión (0: No, 1: Sí)")
    plt.ylabel("Edad")
    st.pyplot(plt.gcf())
    plt.close()

def plot_sleep_depression(df):
    plt.figure(figsize=(8, 5))
    sleep_counts = df.groupby(["Sleep Duration", "Depression"]).size().unstack(fill_value=0)
    sleep_counts.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="coolwarm")
    plt.title("Duración del Sueño y Depresión")
    plt.xlabel("Duración del Sueño")
    plt.ylabel("Cantidad de Estudiantes")
    st.pyplot(plt.gcf())
    plt.close()

def plot_satisfaction_depression(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Depression", y="Study Satisfaction", data=df, palette="coolwarm")
    plt.title("Satisfacción Académica frente a Depresión")
    plt.xlabel("Depresión (0: No, 1: Sí)")
    plt.ylabel("Satisfacción Académica")
    st.pyplot(plt.gcf())
    plt.close()

def plot_financial_stress_depression(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Depression", y="Financial Stress", data=df, palette="coolwarm")
    plt.title("Estrés Financiero frente a Depresión")
    plt.xlabel("Depresión (0: No, 1: Sí)")
    plt.ylabel("Estrés Financiero")
    st.pyplot(plt.gcf())
    plt.close()

def plot_diet_depression(df):
    plt.figure(figsize=(8, 5))
    diet_counts = df.groupby(["Dietary Habits", "Depression"]).size().unstack(fill_value=0)
    diet_counts.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="coolwarm")
    plt.title("Hábitos Dietéticos y Depresión")
    plt.xlabel("Hábitos Dietéticos")
    plt.ylabel("Cantidad de Estudiantes")
    st.pyplot(plt.gcf())
    plt.close()

def plot_family_history_depression(df):
    plt.figure(figsize=(8, 5))
    family_counts = df.groupby(["Family History of Mental Illness", "Depression"]).size().unstack(fill_value=0)
    family_counts.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="coolwarm")
    plt.title("Antecedentes Familiares de Enfermedad Mental y Depresión")
    plt.xlabel("Antecedentes Familiares (No/Yes)")
    plt.ylabel("Cantidad de Estudiantes")
    st.pyplot(plt.gcf())
    plt.close()

def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu", center=0)
    plt.title("Mapa de Calor de Correlaciones entre Variables Numéricas")
    st.pyplot(plt.gcf())
    plt.close()

def plot_age_histogram(df):
    plt.figure(figsize=(8,5))
    df["Age"].dropna().hist(bins=10, color="skyblue", edgecolor="black")
    plt.title("Distribución de Edades en el Dataset")
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")
    st.pyplot(plt.gcf())
    plt.close()

def plot_work_study_hours_depression(df):
    plt.figure(figsize=(8, 5))
    hours_counts = df.groupby(["Work/Study Hours", "Depression"]).size().unstack(fill_value=0)
    hours_counts.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="coolwarm")
    plt.title("Distribución de Horas de Estudio/Trabajo vs Depresión")
    plt.xlabel("Horas de Estudio/Trabajo")
    plt.ylabel("Cantidad de Estudiantes")
    st.pyplot(plt.gcf())
    plt.close()

def plot_suicidal_depression(df):
    plt.figure(figsize=(8, 5))
    suicidal_counts = df.groupby(["Have you ever had suicidal thoughts ?", "Depression"]).size().unstack(fill_value=0)
    suicidal_counts.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="coolwarm")
    plt.title("Relación entre Pensamientos Suicidas y Depresión")
    plt.xlabel("Pensamientos Suicidas (No/Yes)")
    plt.ylabel("Cantidad de Estudiantes")
    st.pyplot(plt.gcf())
    plt.close()

def plot_scatter_age_cgpa_depression(df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="Age", y="CGPA", hue="Depression", palette="coolwarm", alpha=0.7)
    plt.title("Relación Edad vs CGPA")
    plt.xlabel("Edad")
    plt.ylabel("CGPA")
    st.pyplot(plt.gcf())
    plt.close()


def main():
    st.title("Análisis Extensivo e Interactivo de la Depresión en Estudiantes")
    show_initial = st.checkbox("Mostrar Información General del Dataset", value=True)
    if show_initial:
        show_initial_data_overview(data)

    filtered_data = sidebar_filters(data)

    st.subheader("Distribución de Edades (Histograma)")
    plot_age_histogram(filtered_data)
    st.write("Observa la dispersión de edades de los estudiantes en el dataset.")

    st.subheader("Distribución General de la Depresión")
    plot_general_distribution(filtered_data)
    st.write("Cantidad total de estudiantes que reportaron síntomas de depresión vs quienes no.")

    st.subheader("Distribución de Depresión por Género")
    plot_gender_depression(filtered_data)
    st.write("Prevalencia de la depresión entre géneros, para detectar posibles disparidades.")

    st.subheader("Relación entre Edad y Depresión (Boxplot)")
    plot_age_depression(filtered_data)
    st.write("Cómo se distribuye la Edad según presencia (1) o ausencia (0) de depresión.")

    st.subheader("Duración del Sueño y Niveles de Depresión")
    plot_sleep_depression(filtered_data)
    st.write("Relación entre la duración del sueño y la prevalencia de depresión.")

    st.subheader("Satisfacción Académica Frente a Depresión")
    plot_satisfaction_depression(filtered_data)
    st.write("Cómo la satisfacción con los estudios se correlaciona con la depresión.")

    st.subheader("Estrés Financiero y Depresión")
    plot_financial_stress_depression(filtered_data)
    st.write("Relación entre el estrés financiero y la prevalencia de depresión.")

    st.subheader("Hábitos Dietéticos y Depresión")
    plot_diet_depression(filtered_data)
    st.write("Comparación de la prevalencia de depresión según los hábitos alimenticios.")

    st.subheader("Antecedentes Familiares y Depresión")
    plot_family_history_depression(filtered_data)
    st.write("Influencias de antecedentes familiares de enfermedades mentales en la depresión.")

    st.subheader("Distribución de Horas de Estudio/Trabajo vs Depresión")
    plot_work_study_hours_depression(filtered_data)
    st.write("Relación entre el número de horas de estudio/trabajo y la depresión.")

    st.subheader("Relación entre Pensamientos Suicidas y Depresión")
    plot_suicidal_depression(filtered_data)
    st.write("Comparación de la prevalencia de depresión según la presencia de pensamientos suicidas.")

    st.subheader("Relación Edad vs CGPA")
    plot_scatter_age_cgpa_depression(filtered_data)
    st.write("Visualiza cómo se relacionan la Edad y el CGPA, coloreando por depresión.")

    st.subheader("Mapa de Calor de Correlaciones")
    plot_correlation_heatmap(filtered_data)
    st.write("Visualización de la correlación entre las variables numéricas del dataset.")


if __name__ == "__main__":
    main()
