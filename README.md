# Cyclone Impact & Infrastructure Vulnerability Forecaster — Backend API

FastAPI-powered geospatial analytics and disaster management engine for the Bay of Bengal & Coastal APAC region.

## Features
- **Hydrodynamic Surge Simulation**: Multi-tier coastal inundation modeling with SLOSH & Bathurst SLP scaling.
- **Vulnerability Engine**: Spatial intersection with critical coastal infrastructure (substations, hospitals, ports, roads, bridges, shelters).
- **Google Earth Engine (GEE)**: Direct ingestion of NASADEM, Sentinel-1 SAR flood overlays, and GPM precipitation.
- **Gemini 1.5/2.0 Multi-Stage Reasoning**: Generates prioritized municipal action briefs, vernacular SMS bulletins (Hindi, Odia, Bengali, Telugu), and parametric insurance payouts.
- **TTS Engine**: Real-time multi-lingual emergency audio broadcast generator.

---

## Quickstart

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and provide your credentials:
```bash
cp .env.example .env
```

### 3. Run the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs will be available at: `http://localhost:8000/docs`.
