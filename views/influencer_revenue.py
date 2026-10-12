from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Influencer Campaign Revenue Predictor", layout="centered"
)


@st.cache_resource
def load_model():
    return joblib.load(
        Path(__file__).resolve().parent.parent / "influencer_revenue_model.pkl"
    )


model = load_model()

st.title("Influencer Campaign Revenue Predictor")
st.write("Enter the campaign details to estimate revenue (USD).")

st.subheader("Creator and audience")
col1, col2 = st.columns(2)
with col1:
    creator_followers_k = st.slider(
        "Creator followers (thousands)",
        min_value=15.0,
        max_value=450.0,
        value=230.0,
        step=0.1,
    )
    engagement_rate_pct = st.slider(
        "Engagement rate (%)", min_value=0.8, max_value=9.5, value=5.2, step=0.1
    )
    creator_phone_os = st.selectbox("Creator phone OS", ["Android", "iOS"])
with col2:
    audience_intent_tier = st.selectbox(
        "Audience intent tier",
        ["Broad-Entertainment", "Lifestyle-Warm", "High-Intent-Niche"],
    )
    avg_watch_time_sec = st.slider(
        "Average watch time (seconds)",
        min_value=6.0,
        max_value=55.0,
        value=30.0,
        step=0.1,
    )

st.subheader("Content")
col3, col4 = st.columns(2)
with col3:
    content_format = st.selectbox(
        "Content format", ["Short-Video-Reel", "Static-Carousel", "YouTube-Deep-Dive"]
    )
    caption_word_count = st.slider(
        "Caption word count", min_value=12, max_value=159, value=86
    )
with col4:
    hashtags_used_count = st.slider(
        "Hashtags used", min_value=1, max_value=24, value=13
    )
    posting_day_weather = st.selectbox(
        "Posting day weather", ["Sunny", "Cloudy", "Rainy"]
    )

st.subheader("Campaign and contract")
col5, col6 = st.columns(2)
with col5:
    paid_boost_spend_usd = st.slider(
        "Paid boost spend (USD)", min_value=100, max_value=4500, value=2370, step=10
    )
    promo_discount_pct = st.slider(
        "Promo discount (%)", min_value=0, max_value=25, value=10
    )
with col6:
    contract_payment_terms = st.selectbox(
        "Contract payment terms", ["Net-15", "Net-30", "Upfront"]
    )

input_df = pd.DataFrame(
    [
        {
            "creator_followers_k": creator_followers_k,
            "engagement_rate_pct": engagement_rate_pct,
            "paid_boost_spend_usd": paid_boost_spend_usd,
            "avg_watch_time_sec": avg_watch_time_sec,
            "promo_discount_pct": promo_discount_pct,
            "content_format": content_format,
            "audience_intent_tier": audience_intent_tier,
            "caption_word_count": caption_word_count,
            "hashtags_used_count": hashtags_used_count,
            "creator_phone_os": creator_phone_os,
            "contract_payment_terms": contract_payment_terms,
            "posting_day_weather": posting_day_weather,
        }
    ]
)

if st.button("Predict revenue", type="primary"):
    prediction = float(model.predict(input_df)[0])
    st.metric("Predicted campaign revenue", f"${max(prediction, 0):,.0f}")
    st.caption(
        "Typical error on held-out data is roughly $800 (RMSE). "
        "Predictions are most reliable inside the ranges the model was trained on."
    )
