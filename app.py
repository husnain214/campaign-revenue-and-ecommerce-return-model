import streamlit as st

st.set_page_config(page_title="ML Portfolio", layout="centered")

pages = [
    st.Page(
        "views/influencer_revenue.py",
        title="Influencer Revenue",
        icon="📈",
        default=True,
    ),
    st.Page("views/return_risk.py", title="E-commerce Return Risk", icon="📦"),
]

pg = st.navigation(pages)
st.sidebar.caption("ML Portfolio Project")
pg.run()
