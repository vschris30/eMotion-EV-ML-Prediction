import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("eMotion_EV_Performance_and_Price_Dataset.csv")

st.set_page_config(page_title="eMotion EV Dashboard", layout="wide")
st.title("eMotion: EV Performance & Price Dashboard")

# Tabs for separate sections
tabs = st.tabs(["Performance Analysis", "Price Prediction"])

# ---------------- Performance Tab ----------------
with tabs[0]:
    st.header("EV Performance Overview")
    st.markdown("Analyze the performance characteristics of various EVs in the Indian market.")

    if st.checkbox("Show Dataset"):
        st.dataframe(df)

    col1, col2 = st.columns(2)

    with col1:
        feature = st.selectbox("Select Feature for Distribution Plot", df.select_dtypes(include='number').columns)
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        x_feat = st.selectbox("X-axis Feature (Scatter)", df.select_dtypes(include='number').columns)
        y_feat = st.selectbox("Y-axis Feature (Scatter)", df.select_dtypes(include='number').columns, index=1)
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=df[x_feat], y=df[y_feat], ax=ax2)
        st.pyplot(fig2)

# ---------------- Price Prediction Tab ----------------
with tabs[1]:
    st.header("Price Prediction Model")
    st.markdown("Estimate the price of an EV based on key specifications.")

    features = ['Top_Speed_kmph', 'Range_km', 'Efficiency_Wh_km',
                'Fast_Charge_km_h', 'Battery_Capacity_kWh',
                'Acceleration_0_100_kmph_sec']

    df_model = df.dropna(subset=features + ['Price_Euros'])

    X = df_model[features]
    y = df_model['Price_Euros']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    st.subheader("Input Features")
    input_data = {feature: st.slider(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean())) for feature in features}

    input_df = pd.DataFrame([input_data])
    predicted_price = model.predict(input_df)[0]
    st.success(f"Estimated Price: €{predicted_price:,.2f}")

    st.markdown("---")
    st.subheader("Model Performance")
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    st.write(f"Test RMSE: €{rmse:,.2f}")

