import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agri_logic import irrigation_estimate, climate_screen

def test_irrigation_zero_need():
    r = irrigation_estimate(4, 1, effective_rain_mm=5, available_soil_water_mm=0, efficiency=.8)
    assert r["ETc_mm_day"] == 4
    assert r["net_irrigation_mm_day"] == 0

def test_irrigation_positive_need():
    r = irrigation_estimate(4, 1.2, effective_rain_mm=1, available_soil_water_mm=1, efficiency=.8)
    assert round(r["net_irrigation_mm_day"], 2) == 2.8
    assert round(r["gross_irrigation_mm_day"], 2) == 3.5

def test_climate_screen():
    r = climate_screen(42, 60, 95)
    assert r["level"] == "High screening signal"
