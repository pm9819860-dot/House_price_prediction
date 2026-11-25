import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="wide")

@st.cache_resource
def train_model():
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=120, random_state=42, n_jobs=-1, min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
    }
    return model, metrics, list(X.columns)

st.title("🏡 House Price Prediction")
st.caption("Machine Learning regression project | Random Forest Regressor")
st.write(
    "Estimate a California district's median house value using demographic "
    "and housing features. This is an educational demo, not a property appraisal."
)

with st.spinner("Loading dataset and training the model for the first time..."):
    model, metrics, feature_names = train_model()

m1, m2, m3 = st.columns(3)
m1.metric("Mean Absolute Error", f"{metrics['MAE']:.3f}")
m2.metric("Root Mean Squared Error", f"{metrics['RMSE']:.3f}")
m3.metric("R² Score", f"{metrics['R2']:.3f}")

st.divider()
st.subheader("Enter property and area details")
left, right = st.columns(2)

with left:
    med_inc = st.slider("Median income (in $10,000s)", 0.5, 15.0, 5.0, 0.1)
    house_age = st.slider("Median house age (years)", 1, 52, 20)
    ave_rooms = st.slider("Average rooms per household", 1.0, 15.0, 6.0, 0.1)
    ave_bedrms = st.slider("Average bedrooms per household", 0.5, 5.0, 1.2, 0.1)

with right:
    population = st.slider("District population", 3, 40000, 1000, 100)
    ave_occup = st.slider("Average household occupancy", 1.0, 10.0, 3.0, 0.1)
    latitude = st.slider("Latitude", 32.5, 42.0, 34.0, 0.1)
    longitude = st.slider("Longitude", -124.3, -114.3, -118.0, 0.1)

if st.button("Predict House Price", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }], columns=feature_names)
    prediction = float(model.predict(input_data)[0])
    st.success(f"Estimated median house value: ${prediction * 100000:,.0f}")
    st.caption("The source dataset target is expressed in units of $100,000.")

with st.expander("About this project"):
    st.markdown("""
    **Workflow:** Data loading → train/test split → Random Forest training →
    evaluation → interactive prediction.

    **Dataset:** California Housing dataset from scikit-learn.
    **Target:** Median house value per district, represented in units of $100,000.
    **Metrics:** MAE, RMSE and R².
    """)

st.info("Disclaimer: The model uses an older California dataset and is for learning purposes only. It should not be used for real estate, lending, or investment decisions.")
