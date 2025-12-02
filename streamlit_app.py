import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import streamlit as st
import folium
from streamlit_folium import st_folium

# -------------------------
# Helper: Matplotlib fig -> numpy array (for folium ImageOverlay)
# -------------------------
def fig_to_array(fig, max_w=1400):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", transparent=True)
    buf.seek(0)

    img = Image.open(buf).convert("RGBA")
    w, h = img.size
    if w > max_w:
        img = img.resize((max_w, int(h * (max_w / w))), Image.LANCZOS)
    return np.array(img)


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


# -------------------------
# Idealized background fields（跟你原来的一样）
# -------------------------
lons = np.linspace(85, 135, 160)
lats = np.linspace(30, 53, 140)
LON, LAT = np.meshgrid(lons, lats)
bounds = [[30, 85], [53, 135]]   # [[south, west], [north, east]]

U_bg = 6 + 0.1 * (LAT - 40)
V_bg = -6 - 0.1 * (LON - 110)


def gaussian(lon0, lat0, scale):
    R2 = (LON - lon0) ** 2 + (LAT - lat0) ** 2
    return scale * np.exp(-R2 / 40)


factor = 1 + gaussian(116, 40, 1.5) + gaussian(125, 45, 1.5) + gaussian(100, 38, 1.2)
U = U_bg * factor
V = V_bg * factor
speed = np.sqrt(U**2 + V**2)

# --- Wind streamlines figure ---
fig1, ax1 = plt.subplots(figsize=(7, 5), dpi=120)
strm = ax1.streamplot(
    LON, LAT, U, V,
    color=speed, cmap='viridis',
    density=1.3, linewidth=1.2
)
fig1.colorbar(strm.lines, ax=ax1, label='Wind speed (idealized)')
ax1.set_xlim(85, 135)
ax1.set_ylim(30, 53)
ax1.set_title('Winter Monsoon Streamlines (Idealized)')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
plt.tight_layout()
wind_img = fig_to_array(fig1)
plt.close(fig1)

# --- Temperature field ---
TEMP = (
    10
    - 0.35 * (LAT - 30)
    + 0.08 * (LON - 85)
    - 6.0 * np.exp(-((LON-100)**2 + (LAT-45)**2)/120)
)

fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=120)
im2 = ax2.pcolormesh(LON, LAT, TEMP, cmap="coolwarm", shading="auto")
fig2.colorbar(im2, ax=ax2, label="Temp (°C, idealized)")
ax2.set_xlim(85, 135)
ax2.set_ylim(30, 53)
ax2.set_title("Temperature Heatmap (Idealized)")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
plt.tight_layout()
temp_img = fig_to_array(fig2)
plt.close(fig2)

# --- Precip field ---
RAIN = (
    2
    + 0.12 * (LON - 90)
    - 0.08 * (LAT - 30)
    + 8.0 * np.exp(-((LON-118)**2 + (LAT-33)**2)/60)
    + 5.5 * np.exp(-((LON-110)**2 + (LAT-38)**2)/90)
)
RAIN = np.clip(RAIN, 0, None)

fig3, ax3 = plt.subplots(figsize=(7, 5), dpi=120)
im3 = ax3.contourf(LON, LAT, RAIN, levels=12, cmap="Blues", alpha=0.95)
fig3.colorbar(im3, ax=ax3, label="Precip (mm/day, idealized)")
ax3.set_xlim(85, 135)
ax3.set_ylim(30, 53)
ax3.set_title("Precipitation (Idealized)")
ax3.set_xlabel("Longitude")
ax3.set_ylabel("Latitude")
plt.tight_layout()
rain_img = fig_to_array(fig3)
plt.close(fig3)

# -------------------------
# Region polygons（跟你现在这版 regions 一致）
# -------------------------
regions = [
    # 北方：被拆成 西北 / 华北 / 东北
    {"name": "North China",
     "coords": [
         (112, 34), (135, 34),
         (135, 42), (112, 42),
         (112, 34)
     ]},
    {"name": "Northeast China",
     "coords": [
         (112, 42), (135, 42),
         (135, 54), (112, 54),
         (112, 42)
     ]},
    {"name": "Northwest China",
     "coords": [
         (78, 34), (112, 34),
         (112, 54), (78, 54),
         (78, 34)
     ]},

    # 中下部：24–34°N 这一条纬带拼满
    {"name": "East China",
     "coords": [
         (112, 24), (135, 24),
         (135, 34), (112, 34),
         (112, 24)
     ]},
    {"name": "Central China",
     "coords": [
         (105, 24), (112, 24),
         (112, 34), (105, 34),
         (105, 24)
     ]},
    {"name": "South China",
     "coords": [
         (78, 18), (135, 18),
         (135, 24), (78, 24),
         (78, 18)
     ]},
    {"name": "Southwest China",
     "coords": [
         (97, 24), (105, 24),
         (105, 34), (97, 34),
         (97, 24)
     ]},

    # 青藏高原：贴在 78–97E, 24–34N 这一块
    {"name": "Tibetan Plateau",
     "coords": [
         (78, 24), (97, 24),
         (97, 34), (78, 34),
         (78, 24)
     ]},
]

province_geojson_base = {
    "type": "FeatureCollection",
    "features": []
}
region_centroids = {}

for reg in regions:
    feature = {
        "type": "Feature",
        "id": reg["name"],
        "properties": {"name": reg["name"]},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[list(c) for c in reg["coords"]]]
        }
    }
    province_geojson_base["features"].append(feature)
    lon_c = sum(c[0] for c in reg["coords"]) / len(reg["coords"])
    lat_c = sum(c[1] for c in reg["coords"]) / len(reg["coords"])
    region_centroids[reg["name"]] = (lat_c, lon_c)


# -------------------------
# Fake crop dataset（保持不变）
# -------------------------
province_names = [r["name"] for r in regions]
years = list(range(1995, 2020))
crops = ["rice", "maize", "soybean", "wheat"]
scenarios = ["historical", "1.5C", "2C"]

rows = []
rng = np.random.default_rng(0)
for p in province_names:
    for y in years:
        for c in crops:
            for s in scenarios:
                base = {"rice": 6.5, "maize": 7.2, "soybean": 2.4, "wheat": 4.8}[c]
                warming = {"historical": 0.0, "1.5C": 0.6, "2C": 1.0}[s]
                yield_val = base - 0.25 * warming + 0.05 * rng.normal()
                vuln = max(0, warming * 0.7 + rng.normal(0, 0.05))
                rows.append([p, y, c, s, yield_val, warming, 0.1 * warming, vuln])

df = pd.DataFrame(rows, columns=[
    "province", "year", "crop", "scenario",
    "yield_tpha", "temp_anom", "precip_anom", "vulnerability"
])


# -------------------------
# Vulnerability 颜色映射（代替原来的 Choropleth）
# -------------------------
LUMINOUS_REDS = [
    (255, 235, 230),
    (255, 200, 185),
    (255, 160, 145),
    (255, 120, 110),
    (230, 70, 70),
    (180, 30, 30)
]
LUMINOUS_REDS_HEX = [rgb_to_hex(c) for c in LUMINOUS_REDS]


def get_vuln_color_map(crop, scenario, year):
    sub = df[(df.crop == crop) & (df.scenario == scenario) & (df.year == year)]
    vmap = {r["province"]: float(r["vulnerability"]) for _, r in sub.iterrows()}
    if not vmap:
        return {}, None, None

    vals = np.array(list(vmap.values()))
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmin == vmax:
        vmax = vmin + 1e-6

    color_map = {}
    for name, v in vmap.items():
        norm = (v - vmin) / (vmax - vmin)
        norm = min(max(norm, 0.0), 1.0)
        idx = int(norm * (len(LUMINOUS_REDS_HEX) - 1))
        color_map[name] = LUMINOUS_REDS_HEX[idx]

    return color_map, vmin, vmax


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Heat & Harvest", layout="wide")

st.title("Idealized Winter Monsoon & Crop Vulnerability (Streamlit版)")

# 侧边栏控件（对应你原来的 Dropdown / ToggleButtons / Slider 等）
st.sidebar.header("Controls")

crop = st.sidebar.selectbox(
    "Crop:",
    options=crops,
    index=1
)

scenario = st.sidebar.radio(
    "Scenario:",
    options=scenarios,
    index=0
)

year = st.sidebar.slider(
    "Year:",
    min_value=1995,
    max_value=2019,
    value=2005,
    step=1
)

show_vuln = st.sidebar.checkbox("Vulnerability On", value=True)

layers = st.sidebar.multiselect(
    "Layers:",
    options=["wind", "temp", "rain", "regions"],
    default=["wind", "temp", "rain", "regions"]
)

st.sidebar.markdown("---")
st.sidebar.write("点击左侧地图中的多边形，可在右侧查看该区域的统计信息。")

col_map, col_info = st.columns([2.2, 1])

with col_map:
    # 创建 folium 地图
    m = folium.Map(location=[38, 105], zoom_start=4, tiles="Esri.WorldPhysical")

    # 背景叠加：Wind / Temp / Rain
    if "wind" in layers:
        folium.raster_layers.ImageOverlay(
            name="Wind Streamlines",
            image=wind_img,
            bounds=bounds,
            opacity=0.75
        ).add_to(m)

    if "temp" in layers:
        folium.raster_layers.ImageOverlay(
            name="Temperature Heatmap",
            image=temp_img,
            bounds=bounds,
            opacity=0.55
        ).add_to(m)

    if "rain" in layers:
        folium.raster_layers.ImageOverlay(
            name="Precipitation",
            image=rain_img,
            bounds=bounds,
            opacity=0.55
        ).add_to(m)

    # Vulnerability 颜色
    vuln_color_map, vmin, vmax = get_vuln_color_map(crop, scenario, year) if show_vuln else ({}, None, None)

    # 区域多边形（代替 ipyleaflet 的 GeoJSON + Popup）
    for reg in regions:
        name = reg["name"]

        base_color = "#ff7800"
        fill_color = vuln_color_map.get(name, base_color if "regions" in layers else "#000000")
        fill_opacity = 0.6 if (show_vuln and name in vuln_color_map) else (0.15 if "regions" in layers else 0.0)
        weight = 2 if "regions" in layers else 0

        folium.Polygon(
            locations=[(lat, lon) for (lon, lat) in reg["coords"]],  # 转成 (lat, lon)
            color=base_color,
            weight=weight,
            fill=True,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            popup=name,  # 用 popup 传回被点击的区域名
        ).add_to(m)

    # Vulnerability 图例（右下角，HTML 注入）
    if show_vuln and vmin is not None:
        ticks = np.linspace(vmin, vmax, 6)
        ticks = [round(float(t), 2) for t in ticks]

        legend_html = """
        <div style="
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 9999;
            background: white;
            padding: 10px 12px;
            border-radius: 6px;
            font-size: 14px;
            box-shadow: 0 0 6px rgba(0,0,0,0.3);
            line-height: 1.2;
        ">
        <b>Vulnerability</b><br>
        <span style="font-size:12px;">(deeper red = more severe)</span>
        <hr style="margin:6px 0;">
        """
        for i in range(6):
            legend_html += f"""
            <div style='display:flex;align-items:center;margin:2px 0;'>
                <div style='width:22px;height:12px;background:{LUMINOUS_REDS_HEX[i]};
                            margin-right:6px;border:1px solid #999;'></div>
                <span>{ticks[i]}</span>
            </div>
            """
        legend_html += "</div>"

        m.get_root().html.add_child(folium.Element(legend_html))

    # 渲染到 Streamlit
    map_data = st_folium(m, width=900, height=650)

with col_info:
    st.subheader("Region statistics")

    clicked_name = None
    if map_data and "last_object_clicked_popup" in map_data:
        clicked_name = map_data["last_object_clicked_popup"]

    if clicked_name:
        st.write(f"**Region:** {clicked_name}")
        row = df[
            (df.province == clicked_name) &
            (df.crop == crop) &
            (df.scenario == scenario) &
            (df.year == year)
        ]
        if row.empty:
            st.write("_No data for this selection._")
        else:
            r0 = row.iloc[0]
            st.write(f"**Crop:** {crop}")
            st.write(f"**Scenario:** {scenario}")
            st.write(f"**Year:** {year}")
            st.markdown("---")
            st.write(f"**Yield:** {r0.yield_tpha:.2f} t/ha")
            st.write(f"**Temp anomaly:** {r0.temp_anom:.2f} °C")
            st.write(f"**Precip anomaly:** {r0.precip_anom:.2f}")
            st.write(f"**Vulnerability:** {r0.vulnerability:.2f}")
    else:
        st.write("点击左侧地图中的任何一个区域查看详细信息。")

