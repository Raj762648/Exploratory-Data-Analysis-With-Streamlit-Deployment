import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os

## Load the model and dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "raw", "Iris.csv")

df = pd.read_csv(file_path)
df.drop("Id",axis=1,inplace=True)
target_names = df["Species"].unique().tolist()
model = joblib.load("model/iris-model.pkl")

# Sidebar Controls
st.sidebar.header("Controls")

# Mode selection
mode = st.sidebar.radio("Choose Mode:", ["EDA", "Predict"])

if mode == "EDA":
    # EDA Controls
    hist_feature = st.sidebar.selectbox("Choose feature for histogram:", df.columns[:-1])
    box_feature = st.sidebar.selectbox("Choose feature for boxplot:", df.columns[:-1])
    x_axis = st.sidebar.selectbox("X-axis for scatter plot:", df.columns[:-1])
    y_axis = st.sidebar.selectbox("Y-axis for scatter plot:", df.columns[:-1])

    # Main Page
    st.title("Iris Dataset EDA Dashboard")

    st.subheader("Raw Data")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Histogram
    st.subheader(f"Histogram of {hist_feature}")
    fig, ax = plt.subplots()
    sns.histplot(df[hist_feature], kde=True, ax=ax,color="#32a8a0")
    st.pyplot(fig)

    # Boxplot
    st.subheader(f"Boxplot of {box_feature} by Species")
    fig, ax = plt.subplots()
    sns.boxplot(x=df["Species"], y=df[box_feature], ax=ax,hue=df["Species"])
    ax.set_xticklabels(target_names)
    st.pyplot(fig)

    # Scatter Plot
    st.subheader(f"Scatter Plot: {x_axis} vs {y_axis}")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[x_axis], y=df[y_axis], hue=df["Species"], ax=ax, palette="deep")
    ax.legend(title="Species")
    st.pyplot(fig)

elif mode == "Predict":
    st.title("Iris Flower Predictor")

    st.sidebar.subheader("Input Features")
    sepal_length = st.sidebar.slider("Sepal Length", float(df["SepalLengthCm"].min()), float(df["SepalLengthCm"].max()), 5.0)
    sepal_width = st.sidebar.slider("Sepal Width", float(df["SepalWidthCm"].min()), float(df["SepalWidthCm"].max()), 3.0)
    petal_length = st.sidebar.slider("Petal Length", float(df["PetalLengthCm"].min()), float(df["PetalLengthCm"].max()), 1.5)
    petal_width = st.sidebar.slider("Petal Width", float(df["PetalWidthCm"].min()), float(df["PetalWidthCm"].max()), 0.2)

  
    # Make prediction
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)

    predit =st.sidebar.button("Predict")
    if predit:

        # st.subheader("Prediction Result")
        predicted_species = target_names[prediction]
        st.write(f"Predicted Species: **{target_names[prediction]}**")

        # Display image of predicted species
        species_images = {
            "Iris-setosa": "images/iris-setosa.jpg",
            "Iris-versicolor": "images/iris-versicolor.jpg",
            "Iris-virginica": "images/iris-virginica.jpg"
            }

        # Create 2 columns
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction Result")
            st.image(species_images[predicted_species], caption=f"Iris {predicted_species}",width=350)
        with col2:
            st.subheader("Prediction Probabilities")
            st.bar_chart(pd.DataFrame(prediction_proba, columns=target_names),width=300)



