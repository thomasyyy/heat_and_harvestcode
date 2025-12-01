import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
import io, base64

from branca.colormap import LinearColormap

from ipyleaflet import (
    Map, basemaps, ImageOverlay, WidgetControl,
    GeoJSON, Popup, Choropleth
)
from ipywidgets import (
    ToggleButton, VBox, HBox, Layout, HTML,
    IntSlider, Dropdown, ToggleButtons
)
from IPython.display import display


# -------------------------
# Helper: Matplotlib fig -> data URI (for Leaflet overlay)
# -------------------------
def fig_to_data_uri(fig, max_w=1400):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", transparent=True)
    buf.seek(0)

    img = Image.open(buf).convert("RGBA")
    w, h = img.size
    if w > max_w:
        img = img.resize((max_w, int(h * (max_w / w))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


# -------------------------
# Idealized background fields
# -------------------------
lons = np.linspace(85, 135, 160)
lats = np.linspace(30, 53, 140)
LON, LAT = np.meshgrid(lons, lats)
bounds = ((30, 85), (53, 135))   # (south, west), (north, east)

U_bg = 6 + 0.1 * (LAT - 40)
V_bg = -6 - 0.1 * (LON - 110)

def gaussian(lon0, lat0, scale):
    R2 = (LON - lon0)**2 + (LAT - lat0)**2
    return scale * np.exp(-R2 / 40)

factor = 1 + gaussian(116, 40, 1.5) + gaussian(125, 45, 1.5) + gaussian(100, 38, 1.2)
U = U_bg * factor
V = V_bg * factor
speed = np.sqrt(U**2 + V**2)

fig1, ax1 = plt.subplots(figsize=(7, 5), dpi=120)
strm = ax1.streamplot(
    LON, LAT, U, V,
    color=speed, cmap='viridis',
    density=1.3, linewidth=1.2
)
fig1.colorbar(strm.lines, ax=ax1, label='Wind speed (idealized)')
ax1.set_xlim(85, 135); ax1.set_ylim(30, 53)
ax1.set_title('Winter Monsoon Streamlines (Idealized)')
ax1.set_xlabel('Longitude'); ax1.set_ylabel('Latitude')
ax1.set_ylabel('Latitude')
plt.tight_layout()

wind_uri = fig_to_data_uri(fig1)
plt.close(fig1)

wind_overlay = ImageOverlay(
    url=wind_uri,
    bounds=bounds,
    opacity=0.75,
    name="Wind Streamlines"
)

TEMP = (
    10
    - 0.35 * (LAT - 30)
    + 0.08 * (LON - 85)
    - 6.0 * np.exp(-((LON-100)**2 + (LAT-45)**2)/120)
)

fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=120)
im2 = ax2.pcolormesh(LON, LAT, TEMP, cmap="coolwarm", shading="auto")
fig2.colorbar(im2, ax=ax2, label="Temp (°C, idealized)")
ax2.set_xlim(85, 135); ax2.set_ylim(30, 53)
ax2.set_title("Temperature Heatmap (Idealized)")
ax2.set_xlabel("Longitude"); ax2.set_ylabel("Latitude")
plt.tight_layout()

temp_uri = fig_to_data_uri(fig2)
plt.close(fig2)

temp_overlay = ImageOverlay(
    url=temp_uri,
    bounds=bounds,
    opacity=0.55,
    name="Temperature Heatmap"
)

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
ax3.set_xlim(85, 135); ax3.set_ylim(30, 53)
ax3.set_title("Precipitation (Idealized)")
ax3.set_xlabel("Longitude"); ax3.set_ylabel("Latitude")
plt.tight_layout()

rain_uri = fig_to_data_uri(fig3)
plt.close(fig3)

rain_overlay = ImageOverlay(
    url=rain_uri,
    bounds=bounds,
    opacity=0.55,
    name="Precipitation"
)


# -------------------------
# Base map
# -------------------------
center_lat, center_lon = 38, 105
m = Map(
    center=(center_lat, center_lon),
    zoom=4,
    basemap=basemaps.Esri.WorldPhysical
)

m.add_layer(temp_overlay)
m.add_layer(rain_overlay)
m.add_layer(wind_overlay)


# -------------------------
# Region polygons
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


def make_geojson_polygon(coords_lonlat):
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon",
                     "coordinates": [[list(c) for c in coords_lonlat]]}
    }

province_geojson = {"type": "FeatureCollection", "features": []}
for reg in regions:
    province_geojson["features"].append({
        "type": "Feature",
        "id": reg["name"],
        "properties": {"name": reg["name"]},
        "geometry": {"type": "Polygon",
                     "coordinates": [[list(c) for c in reg["coords"]]]}
    })


# -------------------------
# Fake crop dataset
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
# UI controls
# -------------------------
crop_dd = Dropdown(
    options=[("Rice", "rice"), ("Maize", "maize"),
             ("Soybean", "soybean"), ("Wheat", "wheat")],
    value="maize",
    description="Crop:",
    layout=Layout(width="220px")
)

scenario_tb = ToggleButtons(
    options=[("Historical", "historical"), ("1.5°C", "1.5C"), ("2°C", "2C")],
    value="historical",
    description="Scenario:",
    layout=Layout(width="340px")
)

year_sl = IntSlider(
    value=2005, min=1995, max=2019, step=1,
    description="Year:",
    continuous_update=False,
    layout=Layout(width="420px")
)

vuln_toggle = ToggleButton(
    value=True,
    description="Vulnerability On",
    tooltip="Show/Hide vulnerability blocks",
    layout=Layout(width="170px")
)

panel_toggle = ToggleButton(
    value=True,
    description="Hide Panel",
    tooltip="Hide/Show this control panel",
    layout=Layout(width="150px")
)

controls_box = VBox([HBox([crop_dd, scenario_tb]), year_sl])

def on_panel_toggle(change):
    if change["name"] != "value":
        return
    if change["new"]:
        controls_box.layout.display = "flex"
        panel_toggle.description = "Hide Panel"
    else:
        controls_box.layout.display = "none"
        panel_toggle.description = "Show Panel"

panel_toggle.observe(on_panel_toggle)

top_row = HBox([vuln_toggle, panel_toggle])
ui = VBox([top_row, controls_box])
m.add_control(WidgetControl(widget=ui, position="bottomleft"))


# -------------------------
# Custom 6-step luminous red colormap + CLEAN legend
# -------------------------
LUMINOUS_REDS = [
    (255, 235, 230),
    (255, 200, 185),
    (255, 160, 145),
    (255, 120, 110),
    (230,  70,  70),
    (180,  30,  30)
]
LUMINOUS_REDS_HEX = [rgb_to_hex(c) for c in LUMINOUS_REDS]

choropleth_layer = None
legend_control = None

def build_choropleth(crop, scenario, year):
    sub = df[(df.crop == crop) & (df.scenario == scenario) & (df.year == year)]
    vmap = {r["province"]: float(r["vulnerability"]) for _, r in sub.iterrows()}

    vals = list(vmap.values()) if vmap else [0, 1]
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmin == vmax:
        vmax = vmin + 1e-6

    # continuous -> step(6)
    cmap = LinearColormap(LUMINOUS_REDS_HEX, vmin=vmin, vmax=vmax).to_step(6)

    ch = Choropleth(
        geo_data=province_geojson,
        choro_data=vmap,
        key_on="id",
        colormap=cmap,
        style={"fillOpacity": 0.60, "weight": 1},
        hover_style={"fillOpacity": 0.90},
        name="Vulnerability"
    )
    return ch, vmin, vmax


def refresh_choropleth(*args):
    global choropleth_layer, legend_control

    # OFF => remove choropleth + legend
    if not vuln_toggle.value:
        if choropleth_layer is not None and choropleth_layer in m.layers:
            m.remove_layer(choropleth_layer)
        if legend_control is not None and legend_control in m.controls:
            m.remove_control(legend_control)
        return

    crop = crop_dd.value
    scenario = scenario_tb.value
    year = year_sl.value

    new_layer, vmin, vmax = build_choropleth(crop, scenario, year)

    # replace old choropleth
    if choropleth_layer is not None and choropleth_layer in m.layers:
        m.remove_layer(choropleth_layer)
    choropleth_layer = new_layer
    m.add_layer(choropleth_layer)

    # replace old legend
    if legend_control is not None and legend_control in m.controls:
        m.remove_control(legend_control)

    # ---- CLEAN LEGEND (no乱码, no重复, round to 2 decimals) ----
    ticks = np.linspace(vmin, vmax, 6)
    ticks = [round(float(t), 2) for t in ticks]

    legend_items = ""
    for i in range(6):
        legend_items += f"""
        <div style='display:flex;align-items:center;margin:2px 0;'>
            <div style='width:22px;height:12px;background:{LUMINOUS_REDS_HEX[i]};
                        margin-right:6px;border:1px solid #999;'></div>
            <span>{ticks[i]}</span>
        </div>
        """

    legend_html = HTML(f"""
    <div style="
        background:white;
        padding:10px 12px;
        border-radius:6px;
        font-size:14px;
        box-shadow:0 0 6px rgba(0,0,0,0.3);
        line-height:1.2;
        ">
        <b>Vulnerability</b><br>
        <span style="font-size:12px;">(deeper red = more severe)</span>
        <hr style="margin:6px 0;">
        {legend_items}
    </div>
    """)

    legend_control = WidgetControl(widget=legend_html, position="bottomright")
    m.add_control(legend_control)


# Hook updates
crop_dd.observe(refresh_choropleth, "value")
scenario_tb.observe(refresh_choropleth, "value")
year_sl.observe(refresh_choropleth, "value")

def on_vuln_toggle(change):
    if change["name"] != "value":
        return
    vuln_toggle.description = "Vulnerability On" if change["new"] else "Vulnerability Off"
    refresh_choropleth()

vuln_toggle.observe(on_vuln_toggle)

refresh_choropleth()


# -------------------------
# Popup on click
# -------------------------
popup = Popup(close_button=True, auto_close=False, close_on_escape_key=True)
state = {"last_region": None}

def make_click_handler(reg, state):
    def _on_click(**kwargs):
        crop = crop_dd.value
        scenario = scenario_tb.value
        year = year_sl.value

        if state["last_region"] == reg["name"] and popup in m.layers:
            m.remove_layer(popup)
            state["last_region"] = None
            return

        row = df[
            (df.province == reg["name"]) &
            (df.crop == crop) &
            (df.scenario == scenario) &
            (df.year == year)
        ]

        if row.empty:
            stats_html = "<i>No data for this selection.</i>"
        else:
            r0 = row.iloc[0]
            stats_html = (
                f"<b>Yield:</b> {r0.yield_tpha:.2f} t/ha<br>"
                f"<b>Temp anomaly:</b> {r0.temp_anom:.2f} °C<br>"
                f"<b>Precip anomaly:</b> {r0.precip_anom:.2f}<br>"
                f"<b>Vulnerability:</b> {r0.vulnerability:.2f}<br>"
            )

        html = (
            f"<h4>{reg['name']}</h4>"
            f"<b>Crop:</b> {crop}<br>"
            f"<b>Scenario:</b> {scenario}<br>"
            f"<b>Year:</b> {year}<br><hr>"
            f"{stats_html}"
        )

        lat_c = sum(c[1] for c in reg["coords"]) / len(reg["coords"])
        lon_c = sum(c[0] for c in reg["coords"]) / len(reg["coords"])

        popup.location = (lat_c, lon_c)
        popup.child = HTML(html)

        if popup not in m.layers:
            m.add_layer(popup)

        state["last_region"] = reg["name"]

    return _on_click


# -------------------------
# Region outline layers
# -------------------------
region_layers = []
for reg in regions:
    feature = make_geojson_polygon(reg["coords"])
    gj = GeoJSON(
        data=feature,
        style={
            "color": "#ff7800",
            "weight": 2,
            "fillColor": "#ff7800",
            "fillOpacity": 0.15
        },
        hover_style={
            "fillOpacity": 0.40,
            "color": "#ff0000"
        },
        name=reg["name"]
    )
    gj.on_click(make_click_handler(reg, state))
    region_layers.append(gj)
    m.add_layer(gj)


# -------------------------
# Top-right layer toggles
# -------------------------
btn_wind = ToggleButton(
    value=True, description="Wind Streamlines",
    tooltip="Show/Hide Wind Streamlines",
    layout=Layout(width="190px")
)
btn_temp = ToggleButton(
    value=True, description="Temperature Heatmap",
    tooltip="Show/Hide Temperature",
    layout=Layout(width="190px")
)
btn_rain = ToggleButton(
    value=True, description="Precipitation",
    tooltip="Show/Hide Precipitation",
    layout=Layout(width="190px")
)

def on_wind(change):
    if change["name"] == "value":
        wind_overlay.opacity = 0.75 if change["new"] else 0.0

def on_temp(change):
    if change["name"] == "value":
        temp_overlay.opacity = 0.55 if change["new"] else 0.0

def on_rain(change):
    if change["name"] == "value":
        rain_overlay.opacity = 0.55 if change["new"] else 0.0

btn_wind.observe(on_wind)
btn_temp.observe(on_temp)
btn_rain.observe(on_rain)

btn_regions = ToggleButton(
    value=True,
    description="Regions On/Off",
    tooltip="Show/Hide ALL Orange Regions",
    layout=Layout(width="190px")
)

def on_regions(change):
    if change["name"] != "value":
        return
    show = change["new"]
    if show:
        for layer in region_layers:
            if layer not in m.layers:
                m.add_layer(layer)
    else:
        if popup in m.layers:
            m.remove_layer(popup)
            state["last_region"] = None
        for layer in region_layers:
            if layer in m.layers:
                m.remove_layer(layer)

btn_regions.observe(on_regions)

buttons_box = VBox([btn_wind, btn_temp, btn_rain, btn_regions])
m.add_control(WidgetControl(widget=buttons_box, position="topright"))


# -------------------------
# Display map
# -------------------------
display(m)
