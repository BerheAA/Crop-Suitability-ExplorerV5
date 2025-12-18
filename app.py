import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Ethiopia Crop Suitability Explorer", layout="wide")
DATA_DIR = Path(__file__).parent / "data"

# Robust file detection (avoids failures due to special characters in filenames)
def find_rain_yield():
    candidates = [
        "rainfall_chirps_mean_1990_2020_crop_yield.csv",
        "rainfall_with_chirps_mean_1990_2020_&_crop_yield.csv",
    ]
    for c in candidates:
        p = DATA_DIR / c
        if p.exists():
            return p
    for p in DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        cols = set(df.columns)
        if "mean_rain_mm" in cols and any(col.endswith("_a") for col in cols):
            return p
    return None

def find_et():
    candidates = [
        "gleam_total_et_apr_sep_1990_2020.csv",
        "climate_with_GLEAM_ET_apr_sep.csv",
    ]
    for c in candidates:
        p = DATA_DIR / c
        if p.exists():
            return p
    for p in DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        cols = [c.lower() for c in df.columns]
        if "point_id" in cols and any("et" in c and ("apr" in c or "sep" in c) for c in cols):
            return p
    return None

def find_temp():
    candidates = [
        "era5land_t2m_apr_sep_by_point.csv",
        "ERA5Land_T2M_AprSep_seasonal_mean_by_point.csv",
    ]
    for c in candidates:
        p = DATA_DIR / c
        if p.exists():
            return p
    for p in DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        cols = [c.lower() for c in df.columns]
        if "point_id" in cols and any(("t2m" in c) or ("temp" in c) for c in cols):
            return p
    return None

RAIN_YIELD = find_rain_yield()
ET_FILE    = find_et()
TEMP_FILE  = find_temp()

CROP_PARAMS = {
    "maiz_a": {"label":"Maize",   "tmin":10, "topt":24, "tmax":35},
    "whea_a": {"label":"Wheat",   "tmin":2,  "topt":16, "tmax":28},
    "barl_a": {"label":"Barley",  "tmin":2,  "topt":15, "tmax":26},
    "sorg_a": {"label":"Sorghum", "tmin":12, "topt":27, "tmax":38},
    "soyb_a": {"label":"Soybean",   "tmin":10, "topt":25, "tmax":35},
    "grou_a": {"label":"Groundnut", "tmin":12, "topt":26, "tmax":36},
    "bean_a": {"label":"Beans",     "tmin":8,  "topt":22, "tmax":32},
}
CEREALS = ["maiz_a","whea_a","barl_a","sorg_a"]
LEGUMES = ["soyb_a","grou_a","bean_a"]

def tri(x,a,b,c):
    if pd.isna(x) or x<=a or x>=c: return 0.0
    if x==b: return 1.0
    return (x-a)/(b-a) if x<b else (c-x)/(c-b)

def moisture_score(mi, ai):
    if pd.isna(mi) or pd.isna(ai): return np.nan
    mi_s = np.clip((mi + 200)/400, 0, 1)
    ai_s = np.clip((ai - 0.6)/0.6, 0, 1)
    return 0.5*mi_s + 0.5*ai_s

def thermal_zone(T):
    if pd.isna(T): return "unknown"
    if T < 15: return "cool"
    if T < 20: return "temperate"
    if T < 25: return "warm"
    return "hot"

def moisture_zone(AI):
    if pd.isna(AI): return "unknown"
    if AI < 0.6: return "arid"
    if AI < 0.9: return "semi_arid"
    if AI < 1.2: return "sub_humid"
    return "humid"

def default_thresholds():
    return (0.75, 0.50, 0.25)

def classify(score, thr):
    if pd.isna(score): return "No data"
    hi, md, lo = thr
    if score >= hi: return "High"
    if score >= md: return "Moderate"
    if score >= lo: return "Low"
    return "Unsuitable"

def yield_band(y):
    if pd.isna(y) or y <= 0: return "not_planted"
    if y < 500:  return "failure"
    if y < 1500: return "low"
    if y < 3000: return "moderate"
    return "high"

def extension_summary(aez, crop_label, suit_class):
    thermo, moist = (aez.split("__") + ["unknown","unknown"])[:2]
    if suit_class == "High":
        return f"{crop_label}: recommended for this {thermo}/{moist} zone (Apr–Sep). Promote improved seed, timely planting, and legume rotations for fertility."
    if suit_class == "Moderate":
        return f"{crop_label}: feasible but with risk in {thermo}/{moist}. Emphasize planting date, moisture conservation, adapted varieties, and rotation with legumes."
    if suit_class == "Low":
        return f"{crop_label}: marginal in {thermo}/{moist}. Recommend drought/heat tolerant varieties, soil-water conservation, and diversification."
    if suit_class == "Unsuitable":
        return f"{crop_label}: not recommended in {thermo}/{moist} for Apr–Sep under typical rainfed farmer management. Consider alternatives or irrigation."
    return "Insufficient data for a recommendation."

@st.cache_data(show_spinner=False)
def load_inputs(rain_path, et_path, temp_path):
    rain = pd.read_csv(rain_path)
    et   = pd.read_csv(et_path)
    temp = pd.read_csv(temp_path)

    temp_clim = (temp.groupby("point_id", as_index=False)["T2M_AprSep_mean_C"]
                 .mean()
                 .rename(columns={"T2M_AprSep_mean_C":"T2M_AprSep_mean_C_1990_2020"}))

    base = (rain
            .merge(et[["point_id","mean_ET_apr_sep_mm_1990_2020"]], on="point_id", how="left")
            .merge(temp_clim, on="point_id", how="left"))

    base["Rain_AprSep_mm"] = base["mean_rain_mm"]
    base["ET_AprSep_mm"]   = base["mean_ET_apr_sep_mm_1990_2020"]  # TOTAL ET (GLEAM E)
    return base

def compute(base, rain_delta_pct=0.0, temp_delta_c=0.0):
    df = base.copy()
    df["Rain_AprSep_mm_scn"] = df["Rain_AprSep_mm"] * (1.0 + rain_delta_pct/100.0)
    df["T2M_scn_C"] = df["T2M_AprSep_mean_C_1990_2020"] + temp_delta_c
    df["MI_RainMinusET_mm_scn"] = df["Rain_AprSep_mm_scn"] - df["ET_AprSep_mm"]
    df["AI_RainOverET_scn"] = df["Rain_AprSep_mm_scn"] / df["ET_AprSep_mm"]
    df["AEZ"] = df["T2M_scn_C"].apply(thermal_zone) + "__" + df["AI_RainOverET_scn"].apply(moisture_zone)

    yield_cols = [c for c in CROP_PARAMS if c in df.columns]
    p95 = {c: (np.nanpercentile(df.loc[df[c]>0, c], 95) if (df[c]>0).any() else np.nan) for c in yield_cols}

    def yscore(y, crop):
        if pd.isna(y) or y == 0:
            return np.nan
        denom = p95.get(crop, np.nan)
        if pd.isna(denom) or denom <= 0:
            return np.nan
        s = float(np.clip(y/denom, 0, 1))
        if 0 < y < 500:
            s *= 0.5
        return s

    moist_s = np.array([moisture_score(mi, ai) for mi, ai in zip(df["MI_RainMinusET_mm_scn"], df["AI_RainOverET_scn"])], dtype=float)
    T = df["T2M_scn_C"].to_numpy()

    # rotation feasibility
    leg_scores = []
    for leg in LEGUMES:
        p = CROP_PARAMS[leg]
        t_s = np.array([tri(x, p["tmin"], p["topt"], p["tmax"]) for x in T], dtype=float)
        leg_scores.append(0.5*t_s + 0.5*moist_s)
    rotation_possible = (np.nanmax(np.vstack(leg_scores), axis=0) >= 0.6)

    # thresholds
    thr_by_crop_aez = {c:{} for c in CROP_PARAMS.keys()}
    for crop, p in CROP_PARAMS.items():
        t_s = np.array([tri(x, p["tmin"], p["topt"], p["tmax"]) for x in T], dtype=float)
        clim = 0.5*t_s + 0.5*moist_s
        if crop in yield_cols:
            y = df[crop].to_numpy()
            y_s = np.array([yscore(v, crop) for v in y], dtype=float)
            score = np.where(np.isnan(y_s), clim, 0.4*t_s + 0.4*moist_s + 0.2*y_s)
            bands = np.array([yield_band(v) for v in y], dtype=object)
        else:
            score = clim
            bands = np.array(["no_yield"]*len(df), dtype=object)

        for aez in df["AEZ"].unique():
            m = (df["AEZ"].to_numpy() == aez)
            sc = score[m]
            bd = bands[m]
            thr = default_thresholds()
            if crop in yield_cols:
                med = {}
                for b in ["failure","low","moderate","high"]:
                    mm = (bd == b) & np.isfinite(sc)
                    if mm.sum() >= 20:
                        med[b] = float(np.nanmedian(sc[mm]))
                if {"low","moderate","high"}.issubset(med.keys()):
                    cut_mod = (med["low"] + med["moderate"]) / 2
                    cut_high = (med["moderate"] + med["high"]) / 2
                    cut_low = (med.get("failure", med["low"]) + med["low"]) / 2 if "failure" in med else max(0.15, cut_mod - 0.25)
                    thr = (float(np.clip(cut_high,0,1)), float(np.clip(cut_mod,0,1)), float(np.clip(cut_low,0,1)))
            thr_by_crop_aez[crop][aez] = thr

    out = []
    for crop, p in CROP_PARAMS.items():
        t_s = np.array([tri(x, p["tmin"], p["topt"], p["tmax"]) for x in T], dtype=float)
        m_s = moist_s.copy()
        clim = 0.5*t_s + 0.5*m_s
        if crop in yield_cols:
            y = df[crop].to_numpy()
            y_s = np.array([yscore(v, crop) for v in y], dtype=float)
            final = np.where(np.isnan(y_s), clim, 0.4*t_s + 0.4*m_s + 0.2*y_s)
        else:
            y_s = np.full(len(df), np.nan, dtype=float)
            final = clim

        systems = [("crop_only", final)]
        if crop in CEREALS:
            final_rot = np.clip(final + rotation_possible.astype(float)*0.05, 0, 1)
            systems = [("cereal_only", final), ("cereal_plus_legume_rotation", final_rot)]

        for system, sc in systems:
            tmp = df[["point_id","x","lon","lat","name_adm1","name_adm2","AEZ"]].copy()
            tmp["crop"] = crop
            tmp["crop_label"] = p["label"]
            tmp["system"] = system
            tmp["thermal_suitability_0_1"] = t_s
            tmp["moisture_suitability_0_1"] = m_s
            tmp["yield_scaled_0_1"] = y_s
            tmp["yield_kg_ha"] = df[crop] if crop in df.columns else np.nan
            tmp["Rain_AprSep_mm_scn"] = df["Rain_AprSep_mm_scn"]
            tmp["ET_AprSep_mm"] = df["ET_AprSep_mm"]
            tmp["MI_RainMinusET_mm_scn"] = df["MI_RainMinusET_mm_scn"]
            tmp["T2M_scn_C"] = df["T2M_scn_C"]
            tmp["suitability_score_0_1"] = sc
            tmp["suitability_class_AEZ"] = [classify(v, thr_by_crop_aez[crop].get(a, default_thresholds())) for v,a in zip(sc, tmp["AEZ"])]
            out.append(tmp)

    return df, pd.concat(out, ignore_index=True), thr_by_crop_aez, rotation_possible

def best_crop(suit, system_choice):
    sub = suit[suit["system"] == system_choice].copy()
    sub = sub.sort_values(["point_id","suitability_score_0_1"], ascending=[True, False])
    return sub.groupby("point_id", as_index=False).first()

# ---------- UI ----------
st.title("🇪🇹 Ethiopia Crop Suitability Explorer")
st.caption("Apr–Sep climate (1990–2020 means). ET uses GLEAM total ET (E = evaporation + transpiration).")

if RAIN_YIELD is None or ET_FILE is None or TEMP_FILE is None:
    st.error("Missing input file(s) in data/.")
    st.write("Files currently in data/:")
    st.write([p.name for p in sorted(DATA_DIR.glob('*'))])
    st.stop()

base = load_inputs(RAIN_YIELD, ET_FILE, TEMP_FILE)

with st.sidebar:
    st.header("Scenario sliders")
    rain_delta = st.slider("Rainfall change (Apr–Sep) %", -30, 30, 0, 1)
    temp_delta = st.slider("Temperature change (Apr–Sep) °C", -3.0, 3.0, 0.0, 0.1)
    st.divider()
    st.header("Map view")
    crop_sel = st.selectbox("Crop view", ["Best crop per location"] + [CROP_PARAMS[k]["label"] for k in CROP_PARAMS.keys()], index=0)
    system_sel = st.selectbox("System", ["cereal_only","cereal_plus_legume_rotation","crop_only"], index=0)
    color_by = st.selectbox("Color by", ["Suitability class (AEZ)","Suitability score (0–1)","Thermal suitability","Moisture suitability","Yield-adjusted (scaled)"], index=0)
    st.divider()
    st.header("Filters")
    adm1 = st.multiselect("Region (adm1)", sorted(base["name_adm1"].dropna().unique().tolist()))

df_scn, suit, thr_by_crop_aez, rotation_possible = compute(base, rain_delta, temp_delta)

with st.sidebar:
    aez_options = sorted(df_scn["AEZ"].dropna().unique().tolist())
    aezf = st.multiselect("AEZ (scenario)", aez_options)

mask = np.ones(len(suit), dtype=bool)
if adm1: mask &= suit["name_adm1"].isin(adm1).to_numpy()
if aezf: mask &= suit["AEZ"].isin(aezf).to_numpy()
suit_f = suit.loc[mask].copy()

if crop_sel == "Best crop per location":
    map_df = best_crop(suit_f, system_sel).copy()
    map_df["map_label"] = map_df["crop_label"] + " (" + map_df["suitability_class_AEZ"] + ")"
else:
    crop_code = [k for k,v in CROP_PARAMS.items() if v["label"] == crop_sel][0]
    map_df = suit_f[(suit_f["crop"]==crop_code) & (suit_f["system"]==system_sel)].copy()
    map_df["map_label"] = map_df["crop_label"] + " (" + map_df["suitability_class_AEZ"] + ")"

# Color mapping
if color_by == "Suitability class (AEZ)":
    v = map_df["suitability_class_AEZ"].map({"Unsuitable":0,"Low":1,"Moderate":2,"High":3,"No data":-1}).fillna(-1).astype(int).to_numpy()
    colors = np.array([[120,120,120],[165,0,38],[244,109,67],[254,224,139],[26,152,80]], dtype=int)
    idx = (v + 1).clip(0,4)
    map_df["r"] = colors[idx,0]; map_df["g"]=colors[idx,1]; map_df["b"]=colors[idx,2]
elif color_by == "Suitability score (0–1)":
    vv = map_df["suitability_score_0_1"].fillna(0).clip(0,1).to_numpy()
    map_df["r"]=(255*(1-vv)).astype(int); map_df["g"]=(255*vv).astype(int); map_df["b"]=60
elif color_by == "Thermal suitability":
    vv = map_df["thermal_suitability_0_1"].fillna(0).clip(0,1).to_numpy()
    map_df["r"]=(255*(1-vv)).astype(int); map_df["g"]=(255*vv).astype(int); map_df["b"]=120
elif color_by == "Moisture suitability":
    vv = map_df["moisture_suitability_0_1"].fillna(0).clip(0,1).to_numpy()
    map_df["r"]=(255*(1-vv)).astype(int); map_df["g"]=(255*vv).astype(int); map_df["b"]=200
else:
    vv = map_df["yield_scaled_0_1"].fillna(0).clip(0,1).to_numpy()
    map_df["r"]=(255*(1-vv)).astype(int); map_df["g"]=(255*vv).astype(int); map_df["b"]=30

c1, c2 = st.columns([1.15, 0.85], gap="large")

with c1:
    st.subheader("Interactive map")
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_radius=4000,
        pickable=True,
        auto_highlight=True,
        get_fill_color='[r, g, b, 160]',
        get_line_color='[0, 0, 0, 80]',
        line_width_min_pixels=1,
    )
    view_state = pdk.ViewState(
        latitude=float(map_df["lat"].mean()) if len(map_df) else 9.0,
        longitude=float(map_df["lon"].mean()) if len(map_df) else 39.5,
        zoom=5.0,
    )
    tooltip = {
        "html": "<b>{map_label}</b><br/>AEZ: {AEZ}<br/>Score: {suitability_score_0_1}<br/>Thermal: {thermal_suitability_0_1}<br/>Moisture: {moisture_suitability_0_1}<br/>T: {T2M_scn_C} °C<br/>Rain: {Rain_AprSep_mm_scn} mm<br/>ET: {ET_AprSep_mm} mm<br/>MI(R-ET): {MI_RainMinusET_mm_scn} mm",
        "style": {"backgroundColor": "white", "color": "black"}
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip), use_container_width=True)

    st.subheader("Best crop per location (table)")
    if crop_sel == "Best crop per location":
        show = map_df[["point_id","name_adm1","name_adm2","AEZ","crop_label","system","suitability_score_0_1","suitability_class_AEZ"]].copy()
        st.dataframe(show.sort_values(["name_adm1","name_adm2"]).head(300), use_container_width=True, height=360)
    else:
        st.info("Choose 'Best crop per location' to see the max-suitability crop per point.")

with c2:
    st.subheader("Charts and extension guidance")
    st.markdown("**Suitability distribution**")
    hist = alt.Chart(map_df.dropna(subset=["suitability_score_0_1"])).mark_bar().encode(
        x=alt.X("suitability_score_0_1:Q", bin=alt.Bin(maxbins=20)),
        y="count()"
    ).properties(height=140)
    st.altair_chart(hist, use_container_width=True)

    st.markdown("**Rotation feasibility**")
    st.write(f"Locations climate-suitable for ≥1 legume in rotation: **{float(np.mean(rotation_possible)):.1%}**")

    st.markdown("**Policy / extension language (top examples)**")
    if len(map_df):
        top = map_df.sort_values("suitability_score_0_1", ascending=False).head(3)
        for _, r in top[["AEZ","crop_label","suitability_class_AEZ"]].iterrows():
            st.write("• " + extension_summary(r["AEZ"], r["crop_label"], r["suitability_class_AEZ"]))
    else:
        st.warning("No points match the current filters.")

st.divider()
st.caption("If you see a missing-file error in deployment, open the app logs: the app will list what files exist inside data/.")
