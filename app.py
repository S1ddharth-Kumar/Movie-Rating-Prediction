import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

@st.cache_data
def load_data():
    df = pd.read_csv("Movies Dataset.csv")
    X_columns = joblib.load("X_columns.pkl")
    return df, X_columns

@st.cache_resource
def load_models():
    model1 = joblib.load("reg_model.pkl")
    model2 = joblib.load("gb_model.pkl")
    model3 = joblib.load("xgb_model.pkl")
    return model1, model2, model3

df, X_columns = load_data()
model1, model2, model3 = load_models()


st.title("🎬 Movie Rating Predictor")

movie_titles = df['title'].dropna().unique().tolist()
selected_title = st.selectbox("Select a movie title", sorted(movie_titles))

movie_row = df[df['title'] == selected_title]

if movie_row.empty:
    st.error("Movie not found.")
else:
    features = movie_row[X_columns].values

    rating1 = model1.predict(features)[0]
    rating2 = model2.predict(features)[0]
    rating3 = model3.predict(features)[0]

    st.subheader("📊 Predicted Ratings")
    st.markdown(f"**Random Forest:** {rating1:.2f}")
    st.markdown(f"**Gradient Boost:** {rating2:.2f}")
    st.markdown(f"**XG Boost:** {rating3:.2f}")

    st.subheader("📁 Movie Details")
    st.markdown(f"**Year:** {movie_row.iloc[0]['year']}")
    st.markdown(f"**Director:** {movie_row.iloc[0]['directors']}")
    st.markdown(f"**Writers:** {movie_row.iloc[0]['writers']}")
    st.markdown(f"**Cast:** {movie_row.iloc[0]['cast']}")
    st.markdown(f"**Genres:** {movie_row.iloc[0]['genres']}")
    st.markdown(f"**Actual Rating:** {movie_row.iloc[0]['rating']}")
