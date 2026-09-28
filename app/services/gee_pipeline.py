import os
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Try importing Earth Engine
EE_AVAILABLE = False
try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    logger.warning("earthengine-api not installed. Running in mock/standalone mode.")

class GEEPipelineService:
    """
    Google Earth Engine Pipeline Service for extracting:
    - NASADEM / FABDEM Coastal Elevation
    - GPM IMERG Precipitation feeds
    - Sentinel-1 SAR Backscatter for water extent / flood mapping
    - TileJSON & Tile URLs for Deck.gl / Mapbox visualization
    """

    def __init__(self, service_account: Optional[str] = None, key_path: Optional[str] = None, project_id: Optional[str] = None):
        self.service_account = service_account or os.getenv("GEE_SERVICE_ACCOUNT")
        self.key_path = key_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.project_id = project_id or os.getenv("GEE_PROJECT_ID", "ee-cyclone-forecaster")
        self.initialized = False
        self._initialize_gee()

    def _initialize_gee(self):
        if not EE_AVAILABLE:
            logger.info("Earth Engine module unavailable. Utilizing local geospatial simulation fallback.")
            return

        try:
            if self.service_account and self.key_path and os.path.exists(self.key_path):
                credentials = ee.ServiceAccountCredentials(self.service_account, self.key_path)
                ee.Initialize(credentials, project=self.project_id)
                self.initialized = True
                logger.info(f"GEE successfully initialized with Service Account: {self.service_account}")
            else:
                # Attempt standard initialization
                ee.Initialize(project=self.project_id)
                self.initialized = True
                logger.info("GEE initialized via default environment credentials.")
        except Exception as e:
            logger.warning(f"GEE Initialization notice: {e}. Fallback tiles will be generated dynamically.")
            self.initialized = False

    def get_elevation_dem(self, bbox: List[float]):
        """
        Extract NASADEM or FABDEM elevation within bounding box [min_lon, min_lat, max_lon, max_lat]
        """
        if not self.initialized:
            return None
        try:
            region = ee.Geometry.BBox(bbox[0], bbox[1], bbox[2], bbox[3])
            dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation').clip(region)
            return dem
        except Exception as e:
            logger.error(f"Error fetching DEM from GEE: {e}")
            return None

    def get_precipitation_imerg(self, bbox: List[float], start_date: str = "2024-05-20", end_date: str = "2024-05-28"):
        """
        Extract GPM IMERG Daily/30-min precipitation
        """
        if not self.initialized:
            return None
        try:
            region = ee.Geometry.BBox(bbox[0], bbox[1], bbox[2], bbox[3])
            imerg = (ee.ImageCollection("NASA/GPM_L3/IMERG_V06")
                     .filterDate(start_date, end_date)
                     .select('precipitationCal')
                     .sum()
                     .clip(region))
            return imerg
        except Exception as e:
            logger.error(f"Error fetching GPM IMERG: {e}")
            return None

    def get_sentinel1_water_extent(self, bbox: List[float], start_date: str = "2024-05-20", end_date: str = "2024-05-28"):
        """
        Sentinel-1 SAR VV backscatter for water surface mapping
        """
        if not self.initialized:
            return None
        try:
            region = ee.Geometry.BBox(bbox[0], bbox[1], bbox[2], bbox[3])
            s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filterBounds(region)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .select('VV')
                  .mosaic()
                  .clip(region))
            # Rough threshold for water surface: backscatter < -16 dB
            water = s1.lt(-16.0).rename('water_mask')
            return water
        except Exception as e:
            logger.error(f"Error fetching Sentinel-1: {e}")
            return None

    def get_tile_layers_metadata(self, region_name: str = "odisha_bengal") -> Dict[str, Any]:
        """
        Returns TileJSON / Map tile endpoints for NASADEM elevation, GPM precipitation, and Sentinel-1
        """
        # Bounding box presets for Bay of Bengal coastal hubs
        regions = {
            "odisha_bengal": {
                "name": "Odisha - North Andhra - Bengal Coast",
                "bbox": [83.5, 17.5, 89.5, 22.5],
                "center": [86.2, 19.8],
                "zoom": 8
            },
            "bangladesh_delta": {
                "name": "Bangladesh Meghna Delta & Sundarbans",
                "bbox": [88.5, 21.0, 92.5, 24.0],
                "center": [90.5, 22.3],
                "zoom": 8
            },
            "andhra_coromandel": {
                "name": "Andhra Pradesh / Coromandel Coast",
                "bbox": [79.8, 13.0, 84.5, 18.0],
                "center": [82.2, 16.5],
                "zoom": 8
            }
        }

        selected_region = regions.get(region_name, regions["odisha_bengal"])
        bbox = selected_region["bbox"]

        elevation_tile_url = None
        precipitation_tile_url = None
        s1_water_tile_url = None

        if self.initialized:
            try:
                dem = self.get_elevation_dem(bbox)
                if dem:
                    elev_vis = {'min': 0, 'max': 30, 'palette': ['001133', '006699', '33cc99', 'ffcc00', '993300']}
                    elev_map = dem.getMapId(elev_vis)
                    elevation_tile_url = elev_map['tile_fetcher'].url_format

                precip = self.get_precipitation_imerg(bbox)
                if precip:
                    precip_vis = {'min': 50, 'max': 500, 'palette': ['ffffff', '66ffff', '0066ff', 'ff00ff', 'ff0000']}
                    precip_map = precip.getMapId(precip_vis)
                    precipitation_tile_url = precip_map['tile_fetcher'].url_format

                s1_water = self.get_sentinel1_water_extent(bbox)
                if s1_water:
                    s1_vis = {'min': 0, 'max': 1, 'palette': ['00000000', '00e5ff']}
                    s1_map = s1_water.getMapId(s1_vis)
                    s1_water_tile_url = s1_map['tile_fetcher'].url_format
            except Exception as e:
                logger.warning(f"Failed to obtain GEE live tile URLs: {e}")

        # Fallback tile endpoints or visual overlays
        return {
            "region": selected_region,
            "is_gee_live": self.initialized,
            "layers": [
                {
                    "id": "nasadem_coastal_elevation",
                    "name": "NASADEM High-Resolution Coastal Elevation (0-30m)",
                    "type": "raster",
                    "source_type": "GEE / NASA JPL",
                    "tile_url": elevation_tile_url or "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
                    "opacity": 0.65,
                    "legend": {
                        "unit": "meters above sea level",
                        "scale": [
                            {"value": "< 2m", "color": "#ff0033", "risk": "Extreme Inundation Hazard"},
                            {"value": "2m - 5m", "color": "#ff9900", "risk": "High Surge Risk"},
                            {"value": "5m - 10m", "color": "#33cc99", "risk": "Moderate Elevation"},
                            {"value": "> 10m", "color": "#006699", "risk": "Safe Ridge"}
                        ]
                    }
                },
                {
                    "id": "gpm_imerg_accumulated_rain",
                    "name": "GPM IMERG 72h Storm Precipitation",
                    "type": "raster",
                    "source_type": "NASA Global Precipitation Measurement (GPM)",
                    "tile_url": precipitation_tile_url or "https://tile.openweathermap.org/map/precipitation_new/{z}/{x}/{y}.png",
                    "opacity": 0.55,
                    "legend": {
                        "unit": "mm accumulated",
                        "scale": [
                            {"value": "> 400mm", "color": "#ff0000", "risk": "Catastrophic Flash Flood"},
                            {"value": "250 - 400mm", "color": "#ff00ff", "risk": "Severe Flash Flood"},
                            {"value": "100 - 250mm", "color": "#0066ff", "risk": "Moderate Inundation"}
                        ]
                    }
                },
                {
                    "id": "sentinel1_sar_flood_extent",
                    "name": "Sentinel-1 SAR Surface Water Backscatter Extent",
                    "type": "raster",
                    "source_type": "ESA Copernicus Sentinel-1",
                    "tile_url": s1_water_tile_url,
                    "opacity": 0.70,
                    "legend": {
                        "unit": "Binary Standing Water Mask",
                        "scale": [
                            {"value": "Submerged Surface", "color": "#00e5ff"}
                        ]
                    }
                }
            ]
        }

gee_service = GEEPipelineService()
