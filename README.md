# 🌏 Interactive Climate–Crop Explorer
**An interactive ipyleaflet visualization combining climate fields, regional boundaries, and crop vulnerability data**

This project provides a fully interactive geospatial dashboard built with **Python**, **Jupyter notebook widgets**, **Matplotlib**, and **ipyleaflet**.  
It visualizes **idealized climate fields** (wind streamlines, temperature, precipitation) over East Asia and overlays them with **region-level crop yield & vulnerability data** under multiple climate-warming scenarios.

---

## 📁 Project Structure

├── LICENSE
├── README.md
├── app.py             ← main script generating the interactive map
├── data/              ← optional data folder
└── requirements.txt   ← dependencies

---

## ✨ Features

### 🌀 1. Climate Visualization Layers

The script generates and overlays:

- Wind streamlines
- Temperature heatmap
- Precipitation contour map

All generated using Matplotlib → converted to base64 PNG → displayed as Leaflet ImageOverlay layers with opacity toggles.

---

### 🗺️ 2. Interactive Regions

Eight conceptual regions are defined:

- North China
- Northeast China
- Northwest China
- East China
- Central China
- South China
- Southwest China
- Tibetan Plateau

Each region is clickable, showing a popup with:

- Selected crop
- Scenario
- Year
- Yield (t/ha)
- Temperature anomaly
- Precipitation anomaly
- Vulnerability index

---

### 🌾 3. Synthetic Crop Dataset

The script auto-generates a structured dataset:

- Crops: rice, maize, soybean, wheat
- Scenarios: historical, 1.5°C, 2°C
- Years: 1995–2019
- Regions: 8 regions
- Variables: yield, temp anomaly, precip anomaly, vulnerability

Random noise is added for realism.

---

### 🔴 4. Choropleth Vulnerability Map

A custom 6-step luminous red colormap is used.

Includes:

- Auto-scaled vmin/vmax  
- Smooth stepped colormap  
- A clean HTML legend  
- Toggle to hide/show the entire vulnerability map  

---

### 🧰 5. Interactive Controls

Left-bottom control panel:

- Crop selector
- Scenario toggle buttons
- Year slider
- Vulnerability on/off toggle
- Hide/show panel toggle

Top-right control panel:

- Wind layer toggle
- Temperature layer toggle
- Precipitation layer toggle
- Region outline toggle

---

## 🚀 How to Run

### 1. Create environment

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

### 2. Install dependencies

pip install -r requirements.txt

### 3. Launch Jupyter Notebook

jupyter notebook

Open the notebook containing or importing app.py.  
The map will display automatically.

---

## 🧩 Main Dependencies

- numpy
- pandas
- matplotlib
- pillow
- ipyleaflet
- branca
- ipywidgets

Ensure Jupyter is configured to support ipyleaflet widgets.

---

## 📜 License
See the LICENSE file.

---

## 🙌 Acknowledgements

- ipyleaflet – interactive geospatial visualization  
- Matplotlib – climate field rendering  
- branca – colormaps  
- Jupyter widgets – UI controls  
