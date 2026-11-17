import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_utils import load_crop_data, DATA
from src.models import FEATURES, compare_models, feature_importance
from src.agri_logic import irrigation_estimate, climate_screen

st.set_page_config(page_title="AGROGUARD AI", page_icon="🌱", layout="wide")
st.title("🌱 AGROGUARD AI")
st.caption("Explainable, water-aware agricultural decision support — research prototype")
st.warning("Research prototype only. Not an official agronomic advisory. Verify decisions with local soil tests and qualified agricultural experts.")

try:
    df, provenance = load_crop_data()
except Exception as e:
    st.error(f"Could not load crop dataset: {e}")
    st.stop()

is_demo = "ILLUSTRATIVE DEMO" in provenance
if is_demo:
    st.info(provenance + ". Demo predictions are for interface testing only; do not report their metrics as research findings.")
else:
    st.success(provenance)

with st.spinner("Training and evaluating candidate models..."):
    try:
        best, best_name, scores, X_test, y_test, y_pred = compare_models(df)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["🌾 Crop suitability", "💧 Irrigation", "🌦️ Climate screening", "📊 Research evaluation"])

with tab1:
    st.subheader("Crop suitability assessment")
    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.number_input("Nitrogen (N)", 0.0, 140.0, 70.0)
        p = st.number_input("Phosphorus (P)", 0.0, 145.0, 45.0)
        k = st.number_input("Potassium (K)", 0.0, 205.0, 40.0)
    with c2:
        temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
        humidity = st.number_input("Relative humidity (%)", 0.0, 100.0, 60.0)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    with c3:
        rainfall = st.number_input("Rainfall (mm, dataset time basis)", 0.0, 500.0, 100.0)
        water = st.number_input("Available irrigation water (mm/day)", 0.0, 100.0, 5.0)
        state = st.selectbox("Broad region (context only)", ["North", "South", "East", "West", "Central", "Northeast"])
    if st.button("Analyse crop suitability", type="primary"):
        sample = pd.DataFrame([[n,p,k,temp,humidity,ph,rainfall]], columns=FEATURES)
        prediction = best.predict(sample)[0]
        st.metric("Model's most suitable class", str(prediction).title())
        st.caption(f"Selected model: {best_name}. This is dataset-label similarity, not a guarantee of yield or profitability.")
        core = best.named_steps["model"] if hasattr(best, "named_steps") else best
        if hasattr(best, "predict_proba"):
            probs = best.predict_proba(sample)[0]
            ranking = pd.DataFrame({"Crop class": best.classes_, "Model score": probs}).sort_values("Model score", ascending=False)
            st.plotly_chart(px.bar(ranking.head(8), x="Model score", y="Crop class", orientation="h",
                                   title="Model class scores (not calibrated success probabilities)"), use_container_width=True)
        else:
            st.write("Class scores unavailable for this model.")
        st.write("**Input summary**")
        st.dataframe(sample, use_container_width=True)
        st.write("**Water-aware note:** Available water entered:", water, "mm/day. Compare this with the crop-specific irrigation estimate in the Irrigation tab; this prototype does not yet train a joint crop-water ranking model.")

with tab2:
    st.subheader("Daily crop water requirement estimate")
    st.write("Uses a simplified FAO-style crop water balance. Obtain ET₀ and Kc from a reliable local source for the crop and growth stage.")
    a,b,c = st.columns(3)
    with a:
        et0 = st.number_input("Reference ET₀ (mm/day)", 0.0, 15.0, 4.0)
        kc = st.number_input("Crop coefficient Kc", 0.1, 2.0, 1.0)
    with b:
        rain_eff = st.number_input("Effective rainfall (mm/day)", 0.0, 100.0, 0.0)
        soil_store = st.number_input("Available soil water contribution (mm/day)", 0.0, 50.0, 0.0)
    with c:
        efficiency = st.slider("Irrigation efficiency", 0.1, 1.0, 0.75)
    if st.button("Estimate irrigation"):
        result = irrigation_estimate(et0, kc, rain_eff, soil_store, efficiency)
        m1,m2,m3 = st.columns(3)
        m1.metric("Crop ETc", f"{result['ETc_mm_day']:.2f} mm/day")
        m2.metric("Net irrigation", f"{result['net_irrigation_mm_day']:.2f} mm/day")
        m3.metric("Gross irrigation estimate", f"{result['gross_irrigation_mm_day']:.2f} mm/day")
        st.caption("Illustrative estimate. Field application requires soil-water measurements, rooting depth, rainfall accounting, crop stage and local calibration.")

with tab3:
    st.subheader("Climate threshold screening")
    a,b,c = st.columns(3)
    with a:
        t = st.number_input("Current temperature (°C)", -5.0, 60.0, 30.0)
    with b:
        r = st.number_input("Daily rainfall (mm)", 0.0, 300.0, 5.0)
    with c:
        h = st.number_input("Relative humidity (%)", 0.0, 100.0, 65.0)
    tmin = st.number_input("Configured crop lower temperature threshold (°C)", -5.0, 40.0, 10.0)
    tmax = st.number_input("Configured crop upper temperature threshold (°C)", 5.0, 60.0, 35.0)
    if st.button("Screen climate conditions"):
        result = climate_screen(t, r, h, tmin, tmax)
        st.metric("Screening indicator", result["level"])
        for flag in result["flags"]:
            st.write("•", flag)
        st.caption("These are configurable screening rules, not validated predictions of crop damage, disease or yield loss.")

with tab4:
    st.subheader("Candidate model comparison")
    st.dataframe(scores.style.format({c:"{:.3f}" for c in scores.columns if c != "Model"}), use_container_width=True)
    st.caption(f"Best candidate by hold-out macro-F1: {best_name}. Use repeated stratified cross-validation and external validation before making research claims.")
    imp = feature_importance(best)
    if imp["Importance"].sum() > 0:
        st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", title="Random Forest feature importance"), use_container_width=True)
    st.write("**Evaluation limitations**")
    st.write("- Metrics depend on the dataset and split; they do not establish field performance.")
    st.write("- Compare models on the same untouched test set.")
    st.write("- For pan-India claims, validate across multiple agro-climatic zones and seasons.")
    report = scores.to_csv(index=False).encode("utf-8")
    st.download_button("Download model comparison CSV", report, "model_comparison.csv", "text/csv")

st.divider()
st.subheader("Data and validation")
st.write("Dataset rows:", len(df), "| Crop classes:", df["label"].nunique())
st.write("Expected real dataset location: `data/Crop_recommendation.csv`")
template = pd.DataFrame(columns=["observation_id","date","state","district","crop","soil_N","soil_P","soil_K","soil_pH","soil_moisture_pct","temperature_C","relative_humidity_pct","rainfall_mm","observed_stress","notes"])
st.download_button("Download local field-validation template", template.to_csv(index=False).encode(), "field_validation_template.csv", "text/csv")
