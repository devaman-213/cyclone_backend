import os
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.services.gee_pipeline import gee_service
from app.services.vulnerability import vulnerability_engine
from app.services.gemini_client import gemini_advisor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Track 5 Cyclone Impact & Infrastructure Vulnerability Forecaster API (Bay of Bengal & Coastal APAC)"
)

# Enable CORS for Frontend UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Schemas
class StormSimulationRequest(BaseModel):
    storm_name: str = Field(default="Cyclone Remal-02B", description="Name of the cyclonic system")
    central_pressure_hpa: float = Field(default=940.0, ge=880.0, le=1010.0, description="Central atmospheric pressure in hPa")
    max_sustained_wind_knots: float = Field(default=115.0, ge=30.0, le=185.0, description="Maximum sustained wind speed in knots")
    landfall_lat: float = Field(default=19.8134, ge=10.0, le=25.0, description="Landfall Latitude (Bay of Bengal)")
    landfall_lon: float = Field(default=85.8312, ge=80.0, le=96.0, description="Landfall Longitude (Bay of Bengal)")
    radius_of_max_winds_km: float = Field(default=42.0, ge=10.0, le=150.0, description="Radius of Maximum Winds in km")
    astronomical_tide_m: float = Field(default=1.3, ge=0.0, le=5.0, description="Astronomical High Tide amplitude in meters")
    landfall_eta_hours: int = Field(default=12, ge=0, le=72, description="Landfall ETA in hours")

class AdvisoryGenerationRequest(BaseModel):
    storm_name: str = Field(default="Cyclone Remal-02B")
    storm_metrics: Dict[str, Any]
    vulnerability_summary: Dict[str, Any]
    impacted_assets: List[Dict[str, Any]]
    satellite_image_base64: Optional[str] = None

@app.get("/")
def root():
    return {
        "system": settings.PROJECT_NAME,
        "status": "OPERATIONAL",
        "region": "Bay of Bengal & Coastal APAC",
        "version": settings.VERSION,
        "gee_connected": gee_service.initialized,
        "docs_url": "/docs"
    }

@app.get(f"{settings.API_V1_PREFIX}/gee/layers")
def get_gee_layers(region: str = Query(default="odisha_bengal", description="Region preset: odisha_bengal, bangladesh_delta, andhra_coromandel")):
    """
    Fetches pre-processed GEE terrain (NASADEM), GPM precipitation, and Sentinel-1 SAR flood tiles.
    """
    try:
        layers_data = gee_service.get_tile_layers_metadata(region_name=region)
        return layers_data
    except Exception as e:
        logger.error(f"Error fetching GEE layer metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_PREFIX}/simulate")
def simulate_cyclone_impact(request: StormSimulationRequest):
    """
    Simulates:
    1. Hydrodynamic surge height & multi-tier inundation polygons.
    2. 48-hour projected storm track & wind cones.
    3. Spatial overlay on coastal critical assets (substations, highways, shelters).
    """
    try:
        # Step 1: Calculate Peak Surge
        peak_surge_m = vulnerability_engine.calculate_peak_surge_height(
            central_pressure_hpa=request.central_pressure_hpa,
            max_sustained_wind_knots=request.max_sustained_wind_knots,
            astronomical_tide_m=request.astronomical_tide_m
        )

        # Step 2: Generate Surge Inundation Polygons (GeoJSON)
        surge_geojson = vulnerability_engine.generate_surge_inundation_polygons(
            landfall_lat=request.landfall_lat,
            landfall_lon=request.landfall_lon,
            peak_surge_height_m=peak_surge_m,
            rmw_km=request.radius_of_max_winds_km
        )

        # Step 3: Generate Cyclone Track & Wind Cones
        track_geojson = vulnerability_engine.generate_cyclone_track_and_wind_cone(
            landfall_lat=request.landfall_lat,
            landfall_lon=request.landfall_lon,
            max_wind_knots=request.max_sustained_wind_knots,
            rmw_km=request.radius_of_max_winds_km
        )

        # Step 4: Vulnerability Spatial Intersections
        vulnerability_assessment = vulnerability_engine.assess_infrastructure_vulnerability(
            surge_polygons_geojson=surge_geojson,
            landfall_lat=request.landfall_lat,
            landfall_lon=request.landfall_lon,
            peak_surge_height_m=peak_surge_m
        )

        return {
            "storm_name": request.storm_name,
            "inputs": request.model_dump(),
            "peak_surge_height_m": peak_surge_m,
            "surge_inundation_geojson": surge_geojson,
            "cyclone_track_geojson": track_geojson,
            "vulnerability_assessment": vulnerability_assessment
        }

    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_PREFIX}/generate-advisories")
def generate_advisories(request: AdvisoryGenerationRequest):
    """
    Invokes Gemini 3.7 Flash on damage telemetry to produce:
    - 4-stage reasoning pipeline (Hazard -> Bottlenecks -> Immediate Actions -> Liquidity).
    - High-priority municipal executive briefs.
    - Local multi-language public advisory SMS bulletins.
    - Parametric insurance payout triggers.
    """
    try:
        advisory_dispatch = gemini_advisor.generate_advisory_dispatch(
            storm_name=request.storm_name,
            storm_metrics=request.storm_metrics,
            vulnerability_summary=request.vulnerability_summary,
            impacted_assets=request.impacted_assets,
            image_base64=request.satellite_image_base64
        )
        return advisory_dispatch
    except Exception as e:
        logger.error(f"Advisory generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class TTSRequest(BaseModel):
    text: str
    language_code: str = "hi" # hi, te, bn, en, ta

@app.post(f"{settings.API_V1_PREFIX}/tts")
def generate_speech_audio(request: TTSRequest):
    """
    Generates realistic MP3 audio stream for Indian languages using gTTS (Google Text-to-Speech).
    Supports Hindi (hi), Bengali (bn), Telugu (te), Tamil (ta), English (en/en-in).
    """
    try:
        import io
        from fastapi.responses import Response
        import importlib
        
        try:
            gtts_module = importlib.import_module("gtts")
            gTTS_cls = getattr(gtts_module, "gTTS")
        except ImportError:
            raise HTTPException(status_code=501, detail="gTTS module not installed in current python environment.")

        lang_map = {
            "odia": "hi", # Fallback to clear Hindi for Odia if direct Odia voice unavailable
            "bengali": "bn",
            "telugu": "te",
            "hindi": "hi",
            "english": "en"
        }
        
        target_lang = "hi"
        req_lang_lower = request.language_code.lower()
        for k, v in lang_map.items():
            if k in req_lang_lower:
                target_lang = v
                break
        if target_lang == "hi" and "en" in req_lang_lower:
            target_lang = "en"

        clean_text = request.text.strip().strip('"').strip("'").strip()
        if not clean_text:
            clean_text = "Emergency cyclone advisory alert. Please evacuate to safety."

        tts = gTTS_cls(text=clean_text, lang=target_lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        return Response(content=fp.read(), media_type="audio/mpeg")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
