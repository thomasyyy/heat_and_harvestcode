import streamlit as st
import pandas as pd
import plotly.express as px

# Sample data
df = pd.DataFrame({
    "Country": ["USA", "Canada", "Mexico"],
    "Cases": [1000, 500, 700]
})

st.set_page_config(page_title="COVID-19 Dashboard", layout="centered")

st.title("COVID-19 Dashboard")

fig = px.bar(df, x="Country", y="Cases", title="Sample Dashboard")

st.plotly_chart(fig, use_container_width=True)

