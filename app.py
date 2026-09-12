import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)


# -----------------------------
# Load Model & Data
# -----------------------------
model = joblib.load("sales_forecast_model.pkl")

df = pd.read_csv("monthly_sales.csv")

df["Date"] = pd.to_datetime(df["Date"])



# -----------------------------
# Title
# -----------------------------
st.title("📈 Sales Forecast Dashboard")

st.write(
    "This dashboard predicts future monthly sales using Machine Learning."
)



# -----------------------------
# Sidebar Prediction
# -----------------------------
st.sidebar.header("Sales Prediction")


year = st.sidebar.selectbox(
    "Select Year",
    [2018, 2019, 2020, 2021]
)


month = st.sidebar.selectbox(
    "Select Month",
    range(1, 13)
)



if st.sidebar.button("Predict Sales"):

    prediction = model.predict([[year, month]])

    st.sidebar.success(
        f"Predicted Sales: ${prediction[0]:,.2f}"
    )



# -----------------------------
# Dataset
# -----------------------------
st.subheader("Monthly Sales Dataset")

st.dataframe(df)



# -----------------------------
# Historical Sales Trend
# -----------------------------
st.subheader("Historical Sales Trend")


fig, ax = plt.subplots(figsize=(10,5))


ax.plot(
    df["Date"],
    df["Sales"],
    marker="o"
)


ax.set_title(
    "Monthly Sales Trend"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Sales"
)

ax.grid(True)


st.pyplot(fig)



# -----------------------------
# Model Performance
# -----------------------------
st.subheader("Model Performance")


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "MAE",
        "11,599.75"
    )


with col2:
    st.metric(
        "RMSE",
        "17,017.19"
    )


with col3:
    st.metric(
        "R² Score",
        "0.49"
    )



# -----------------------------
# Dataset Information
# -----------------------------
st.subheader("Dataset Information")


st.write(
    "Number of Records:",
    len(df)
)


st.write(
    "Date Range:"
)


st.write(
    f"{df['Date'].min().date()} ➜ {df['Date'].max().date()}"
)



