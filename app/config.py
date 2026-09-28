import os
from dataclasses import dataclass, field

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class Settings:
    PROJECT_NAME: str = "Cyclone Impact & Infrastructure Vulnerability Forecaster"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Earth Engine Configuration
    GEE_SERVICE_ACCOUNT: str = field(default_factory=lambda: os.getenv("GEE_SERVICE_ACCOUNT", ""))
    GEE_PRIVATE_KEY_PATH: str = field(default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json"))
    GEE_PROJECT_ID: str = field(default_factory=lambda: os.getenv("GEE_PROJECT_ID", "cyclone-impact-forecaster"))
    
    # Gemini Configuration
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    GEMINI_MODEL: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    
    # Paths
    ASSETS_GEOJSON_PATH: str = field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "coastal_critical_assets.geojson")))

settings = Settings()

