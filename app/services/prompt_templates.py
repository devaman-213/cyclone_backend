from typing import Dict, Any

def build_cyclone_advisory_prompt(
    storm_name: str,
    storm_metrics: Dict[str, Any],
    vulnerability_summary: Dict[str, Any],
    impacted_assets: list,
    regional_language: str = "Odia / Telugu / Bengali"
) -> str:
    """
    Constructs the structured prompt for Gemini 3.7 Flash multimodal reasoning.
    Enforces the mandatory 4-step reasoning chain:
    1. Hazard Identification
    2. Critical Bottlenecks
    3. Immediate Actions
    4. Liquidity & Parametric Insurance Trigger Metrics
    """
    prompt = f"""
You are the Chief Geospatial & Disaster Response Intelligence System for the Bay of Bengal & Coastal APAC Track 5 Cyclone Initiative.

Analyze the simulated cyclone telemetry, hydrodynamic surge, and coastal infrastructure vulnerability data below:

### CYCLONE METRICS:
- Cyclone System Name: {storm_name}
- Landfall Coordinates: Lat {storm_metrics.get('landfall_lat')}, Lon {storm_metrics.get('landfall_lon')}
- Central Atmospheric Pressure: {storm_metrics.get('central_pressure_hpa')} hPa
- Maximum Sustained Wind Speed: {storm_metrics.get('max_wind_knots')} knots ({round(float(storm_metrics.get('max_wind_knots', 0)) * 1.852, 1)} km/h)
- Radius of Maximum Winds (RMW): {storm_metrics.get('rmw_km')} km
- Modeled Peak Storm Surge: {vulnerability_summary.get('peak_surge_height_m')} meters above mean sea level
- Landfall ETA: {storm_metrics.get('landfall_eta_hours', 12)} Hours

### CRITICAL INFRASTRUCTURE SPATIAL OVERLAY:
- Cutoff / Submerged Highway Network: {vulnerability_summary.get('total_submerged_roads_km')} km
- Inundated Power Substations: {vulnerability_summary.get('blackout_substations_count')} units
- Estimated Households Facing Immediate Blackout: {vulnerability_summary.get('households_at_risk'):,}
- Isolated Disaster Shelters (Road Connectivity Lost): {vulnerability_summary.get('isolated_shelters_count')} shelters
- Trapped / Cutoff Evacuees inside Shelters: {vulnerability_summary.get('total_evacuees_at_shelters'):,} persons
- Composite Parametric Risk Score: {vulnerability_summary.get('risk_score')}/100 ({vulnerability_summary.get('risk_category')})

### IMPACTED ASSETS BREAKDOWN:
{impacted_assets[:8]}

---
### MANDATORY REASONING & OUTPUT INSTRUCTIONS:
1. Conduct the explicit 4-stage reasoning pipeline:
   - STAGE 1 (Hazard Identification): Characterize the compounding threat of wind shear, tidal phase, and hydrodynamic storm surge.
   - STAGE 2 (Critical Bottlenecks): Detail specific arterial highway failures, power outages, and isolated medical/shelter points.
   - STAGE 3 (Immediate Actions): Step-by-step tactical operational priorities for municipal disaster management authorities (NDRF/ODRAF/SDMA) over the next 0-6h and 6-24h.
   - STAGE 4 (Liquidity & Parametric Insurance): Define parametric payout triggers (e.g. pressure < 950hPa, wind > 105 kts, surge > 2.5m) to release immediate emergency relief funds (Disaster Relief Funds / World Bank Catastrophe Drawdown).

2. Provide the Dual-Format Dispatch:
   - High-Priority Municipal Executive Brief (Structured Markdown for Command Center dashboard).
   - Local Public Advisory SMS/Bulletins (Clear, urgent alerts translated/adapted for regional populations in {regional_language} & English).

Output must strictly adhere to the requested JSON schema.
"""
    return prompt
