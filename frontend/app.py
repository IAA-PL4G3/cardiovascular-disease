import requests
import streamlit as st

API = "http://localhost:8000/predict"

st.set_page_config(page_title="Cardiovascular Risk", layout="centered")
st.title("Cardiovascular Risk Assessment")
st.caption("Clinical fields are optional — missing values are estimated from population data.")

with st.form("form"):
    c1, c2 = st.columns(2)
    age    = c1.number_input("Age (years)", 1, 120, 45)
    gender = c2.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    height = c1.number_input("Height (cm)", 50.0, 300.0, 170.0)
    weight = c2.number_input("Weight (kg)", 10.0, 300.0, 70.0)

    st.markdown("**Lifestyle**")
    c3, c4, c5 = st.columns(3)
    smoke  = int(c3.checkbox("Smoker"))
    alco   = int(c4.checkbox("Drinks alcohol"))
    active = int(c5.checkbox("Physically active", value=True))

    st.markdown("**Clinical** *(optional — leave at 0 if unknown)*")
    c6, c7 = st.columns(2)
    ap_hi = c6.number_input("Systolic BP", 0, 370, 0)
    ap_lo = c7.number_input("Diastolic BP", 0, 250, 0)

    c8, c9 = st.columns(2)
    opts = {0: "Unknown", 1: "Normal", 2: "Above normal", 3: "Well above normal"}
    cholesterol = c8.selectbox("Cholesterol", list(opts), format_func=opts.get)
    gluc        = c9.selectbox("Glucose",     list(opts), format_func=opts.get)

    submitted = st.form_submit_button("Assess risk", use_container_width=True)

if submitted:
    payload = {
        "age_years": age, "gender": gender, "height": height, "weight": weight,
        "smoke": smoke, "alco": alco, "active": active,
        "ap_hi": ap_hi or None, "ap_lo": ap_lo or None,
        "cholesterol": cholesterol or None, "gluc": gluc or None,
    }

    with st.spinner("Calculating..."):
        try:
            r = requests.post(API, json=payload, timeout=30)
            r.raise_for_status()
        except requests.exceptions.ConnectionError:
            st.error("API not reachable. Run: uvicorn api.main:app")
            st.stop()

    d     = r.json()
    pct   = d["risk_percent"]
    label = d["risk_label"]
    icon  = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}[label]

    st.divider()
    st.metric(f"{icon} {label} Risk", f"{pct:.1f}%")
    st.progress(pct / 100)

    if d["estimated"]:
        with st.expander("Estimated missing values"):
            for k, v in d["estimated"].items():
                st.write(f"**{k}:** {v}")

    with st.expander("Per-model breakdown"):
        for name, p in d["models"].items():
            st.write(f"**{name}:** {p * 100:.1f}%")

    st.caption("⚠️ This is a decision-support tool, not a medical diagnosis.")
