import os
import json
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

# -----------------------
# Page
# -----------------------
st.set_page_config(page_title="Milan Housing Dashboard", layout="wide")
st.title("🏙️ Milan Housing Dashboard")
st.caption("Explore listings + map + model metrics + SHAP explainability")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data_final.xlsx")
MAP_HTML_PATH = os.path.join(BASE_DIR, "milan_area_map_lnprice.html")

# Optional artifacts (created from notebook)
METRICS_JSON_PATH = os.path.join(BASE_DIR, "model_metrics.json")
SHAP_SUMMARY_BAR_PNG = os.path.join(BASE_DIR, "shap_summary_bar.png")
SHAP_BEESWARM_PNG = os.path.join(BASE_DIR, "shap_beeswarm.png")
SHAP_DEP_TRANSPORT_PNG = os.path.join(BASE_DIR, "shap_dependence_transport.png")


# -----------------------
# Helpers
# -----------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def exists(path: str) -> bool:
    return os.path.isfile(path)

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_mean(df: pd.DataFrame, col: str) -> str:
    if col and col in df.columns:
        v = to_num(df[col]).dropna()
        if not v.empty:
            return f"{v.mean():,.0f}" if v.mean() > 10 else f"{v.mean():.3f}"
    return "—"

def safe_median(df: pd.DataFrame, col: str) -> str:
    if col and col in df.columns:
        v = to_num(df[col]).dropna()
        if not v.empty:
            return f"{v.median():,.0f}" if v.median() > 10 else f"{v.median():.3f}"
    return "—"


# -----------------------
# Load data
# -----------------------
if not exists(DATA_PATH):
    st.error(f"❌ data file not found: {DATA_PATH}")
    st.stop()

df = load_data(DATA_PATH)

# Detect columns (flexible)
area_col = pick_col(df, ["Area", "area", "neighborhood", "Neighbourhood", "quartiere", "Quartiere"])
bed_col = pick_col(df, ["Bedroom", "Bedrooms", "bedroom", "bedrooms", "beds"])
energy_col = pick_col(df, ["Energy_score", "energy_score", "energy", "EnergyScore", "energyScore"])
transport_col = pick_col(df, ["transport", "Transport", "transport_score", "Transport_score"])

price_col = pick_col(df, ["price", "Price"])
priceperm_col = pick_col(df, ["priceperm", "price_per_m2", "price_per_sqm", "PricePerm", "price_per_meter"])
ln_price_col = pick_col(df, ["ln_price", "lnprice", "Ln_price"])


# -----------------------
# Sidebar filters
# -----------------------
st.sidebar.header("Filters")
filtered = df.copy()

# Area
if area_col:
    areas = sorted([x for x in filtered[area_col].dropna().unique().tolist()])
    default_areas = areas[:10] if len(areas) > 10 else areas
    selected_areas = st.sidebar.multiselect("Area", areas, default=default_areas)
    if selected_areas:
        filtered = filtered[filtered[area_col].isin(selected_areas)]

# Bedroom
if bed_col:
    b = to_num(filtered[bed_col]).dropna()
    if not b.empty:
        bmin, bmax = int(b.min()), int(b.max())
        if bmin != bmax:
            br = st.sidebar.slider("Bedroom range", bmin, bmax, (bmin, bmax))
            filtered = filtered[to_num(filtered[bed_col]).between(br[0], br[1])]

# Energy score
if energy_col:
    e = to_num(filtered[energy_col]).dropna()
    if not e.empty:
        emin, emax = float(e.min()), float(e.max())
        if emin != emax:
            er = st.sidebar.slider("Energy_score range", float(emin), float(emax), (float(emin), float(emax)))
            filtered = filtered[to_num(filtered[energy_col]).between(er[0], er[1])]

# Transport (optional)
if transport_col:
    t = to_num(filtered[transport_col]).dropna()
    if not t.empty:
        tmin, tmax = float(t.min()), float(t.max())
        if tmin != tmax:
            tr = st.sidebar.slider("Transport range (optional)", float(tmin), float(tmax), (float(tmin), float(tmax)))
            filtered = filtered[to_num(filtered[transport_col]).between(tr[0], tr[1])]

st.sidebar.markdown("---")
st.sidebar.metric("Rows after filters", f"{len(filtered):,}")

# Download filtered data
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="⬇️ Download filtered data (CSV)",
    data=csv_bytes,
    file_name="milan_filtered.csv",
    mime="text/csv",
)


# -----------------------
# Tabs
# -----------------------
tab_explore, tab_model, tab_about = st.tabs(["📊 Explore", "🤖 Model", "ℹ️ About"])


# -----------------------
# Explore tab
# -----------------------
with tab_explore:
    st.success(f"✅ Dataset loaded | Rows: {len(df):,} | Columns: {len(df.columns)}")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Listings (filtered)", f"{len(filtered):,}")
    k2.metric("Median price", safe_median(filtered, price_col))
    k3.metric("Median price / m²", safe_median(filtered, priceperm_col))
    k4.metric("Mean ln_price", safe_mean(filtered, ln_price_col))

    st.markdown("### Preview & Map")
    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("📄 Dataset preview")
        st.dataframe(filtered.head(50), use_container_width=True)

    with right:
        st.subheader("🗺️ Milan map (saved Folium HTML)")
        if exists(MAP_HTML_PATH):
            with open(MAP_HTML_PATH, "r", encoding="utf-8") as f:
                map_html = f.read()
            st_html(map_html, height=560, scrolling=True)
            st.caption("Map rendered from milan_area_map_lnprice.html")
        else:
            st.warning("Map file not found: milan_area_map_lnprice.html")

    # Simple chart (optional but nice)
    if price_col and price_col in filtered.columns:
        st.markdown("### Price distribution (filtered)")
        hist_data = to_num(filtered[price_col]).dropna()
        if len(hist_data) > 0:
            st.bar_chart(hist_data.value_counts(bins=25).sort_index())
        else:
            st.info("No numeric price values to plot.")


# -----------------------
# Model tab
# -----------------------
with tab_model:
    st.subheader("🤖 Model results (XGBoost)")

    m1, m2, m3 = st.columns(3)

    if exists(METRICS_JSON_PATH):
        try:
            with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            rmse = float(metrics.get("rmse", 0))
            mae = float(metrics.get("mae", 0))
            r2 = float(metrics.get("r2", 0))

            m1.metric("RMSE", f"{rmse:.4f}")
            m2.metric("MAE", f"{mae:.4f}")
            m3.metric("R²", f"{r2:.4f}")

            st.caption("Metrics loaded from model_metrics.json")
        except Exception as e:
            m1.metric("RMSE", "—")
            m2.metric("MAE", "—")
            m3.metric("R²", "—")
            st.warning(f"Couldn't read model_metrics.json: {e}")
    else:
        m1.metric("RMSE", "—")
        m2.metric("MAE", "—")
        m3.metric("R²", "—")
        st.info("Create model_metrics.json in your notebook to show metrics here.")

    st.markdown("---")
    st.subheader("🔎 SHAP (Explainability)")

    shap_tabs = st.tabs(["Summary bar", "Beeswarm", "Dependence (transport)"])

    with shap_tabs[0]:
        if exists(SHAP_SUMMARY_BAR_PNG):
            st.image(SHAP_SUMMARY_BAR_PNG, use_container_width=True)
        else:
            st.info("Missing: shap_summary_bar.png")

    with shap_tabs[1]:
        if exists(SHAP_BEESWARM_PNG):
            st.image(SHAP_BEESWARM_PNG, use_container_width=True)
        else:
            st.info("Missing: shap_beeswarm.png")

    with shap_tabs[2]:
        if exists(SHAP_DEP_TRANSPORT_PNG):
            st.image(SHAP_DEP_TRANSPORT_PNG, use_container_width=True)
        else:
            st.info("Missing: shap_dependence_transport.png")


# -----------------------
# About tab
# -----------------------
with tab_about:
    st.markdown(
        """
### What is this?
A compact dashboard to explore Milan housing listings, visualize a Folium map, and show an XGBoost model with SHAP explainability.

### Tech stack
- Streamlit (dashboard)
- Pandas (data)
- Folium (map exported as HTML)
- XGBoost (regression)
- SHAP (model explainability)

### Notes
This dashboard is designed as a **portfolio-friendly** project (fast, clean, and easy to deploy).
        """.strip()
    )
