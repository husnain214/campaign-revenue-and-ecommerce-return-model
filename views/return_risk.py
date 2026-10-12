from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="E-commerce Return Risk Predictor", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(
        Path(__file__).resolve().parent.parent / "ecommerce_returns_model.pkl"
    )


model = load_model()

st.title("E-commerce Return Risk Predictor")
st.write("Enter the order details to estimate whether it is likely to be returned.")

st.subheader("Order")
col1, col2 = st.columns(2)
with col1:
    cart_value_usd = st.slider("Cart value (USD)", 25.0, 450.0, 240.0, step=0.5)
    delivery_days_taken = st.slider("Delivery days taken", 1, 11, 6)
    discount_pct_applied = st.select_slider(
        "Discount applied (%)", options=[0, 10, 20, 30, 40, 50], value=20
    )
with col2:
    product_category = st.selectbox(
        "Product category", ["Footwear", "Accessories", "Formal-Eveningwear"]
    )
    fit_review_sentiment = st.selectbox(
        "Fit review sentiment", ["True-to-Size", "Inconsistent-Sizing"]
    )
    packaging_color = st.selectbox("Packaging color", ["Kraft-Brown", "Branded-Teal"])

st.subheader("Customer")
col3, col4 = st.columns(2)
with col3:
    past_return_rate = st.slider("Past return rate", 0.0, 0.9, 0.28, step=0.005)
    loyalty_card_color = st.selectbox("Loyalty card", ["Silver", "Gold", "Platinum"])
    size_bracketing = st.checkbox("Size bracketing (ordered multiple sizes)")
with col4:
    browser_used = st.selectbox("Browser", ["Safari", "Firefox", "Edge", "Chrome"])
    checkout_device = st.selectbox(
        "Checkout device", ["Android-App", "iOS-App", "Desktop"]
    )
    newsletter_subscriber = st.checkbox("Newsletter subscriber")

input_df = pd.DataFrame(
    [
        {
            "cart_value_usd": cart_value_usd,
            "past_return_rate": past_return_rate,
            "delivery_days_taken": delivery_days_taken,
            "discount_pct_applied": discount_pct_applied,
            "product_category": product_category,
            "fit_review_sentiment": fit_review_sentiment,
            "browser_used": browser_used,
            "checkout_device": checkout_device,
            "packaging_color": packaging_color,
            "loyalty_card_color": loyalty_card_color,
            "size_bracketing_flag": int(size_bracketing),
            "newsletter_subscriber": int(newsletter_subscriber),
        }
    ]
)

if st.button("Predict return risk", type="primary"):
    label = int(model.predict(input_df)[0])

    if label == 1:
        st.error("High return risk")
    else:
        st.success("Low return risk")

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(input_df)[0][1])
        st.metric("Probability of return", f"{proba:.0%}")

    st.caption(
        "Predictions are most reliable inside the ranges the model was trained on."
    )
