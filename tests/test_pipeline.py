import os
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OPERATIONAL"

def test_gee_layers():
    response = client.get("/api/v1/gee/layers?region=odisha_bengal")
    assert response.status_code == 200
    data = response.json()
    assert "layers" in data
    assert len(data["layers"]) >= 3

def test_simulation_pipeline():
    payload = {
        "storm_name": "Test Cyclone Yash-01",
        "central_pressure_hpa": 935.0,
        "max_sustained_wind_knots": 120.0,
        "landfall_lat": 19.8134,
        "landfall_lon": 85.8312,
        "radius_of_max_winds_km": 42.0,
        "astronomical_tide_m": 1.4,
        "landfall_eta_hours": 12
    }
    response = client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "peak_surge_height_m" in data
    assert data["peak_surge_height_m"] > 0
    assert "surge_inundation_geojson" in data
    assert "vulnerability_assessment" in data
    assert len(data["vulnerability_assessment"]["impacted_assets"]) > 0

def test_gemini_advisory_generation():
    payload = {
        "storm_name": "Test Cyclone Yash-01",
        "storm_metrics": {
            "central_pressure_hpa": 935.0,
            "max_wind_knots": 120.0,
            "landfall_lat": 19.8134,
            "landfall_lon": 85.8312,
            "landfall_eta_hours": 12
        },
        "vulnerability_summary": {
            "peak_surge_height_m": 3.4,
            "total_submerged_roads_km": 42.5,
            "blackout_substations_count": 2,
            "households_at_risk": 375000,
            "isolated_shelters_count": 2,
            "total_evacuees_at_shelters": 2050,
            "risk_score": 82,
            "risk_category": "EXTREME / CATASTROPHIC"
        },
        "impacted_assets": []
    }
    response = client.post("/api/v1/generate-advisories", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reasoning_stages" in data
    assert "stage_1_hazard_identification" in data["reasoning_stages"]
    assert "stage_4_parametric_triggers" in data["reasoning_stages"]
    assert "municipal_executive_brief" in data
    assert "public_sms_bulletins" in data
