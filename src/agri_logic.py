def irrigation_estimate(et0_mm_day, crop_coefficient, effective_rain_mm=0.0,
                         available_soil_water_mm=0.0, efficiency=0.75):
    """Simplified daily net/gross irrigation estimate in mm/day."""
    if et0_mm_day < 0 or crop_coefficient <= 0:
        raise ValueError("ET0 must be >= 0 and crop coefficient must be > 0.")
    if not 0 < efficiency <= 1:
        raise ValueError("Efficiency must be between 0 and 1.")
    etc = et0_mm_day * crop_coefficient
    net = max(0.0, etc - max(0.0, effective_rain_mm) - max(0.0, available_soil_water_mm))
    gross = net / efficiency
    return {"ETc_mm_day": etc, "net_irrigation_mm_day": net,
            "gross_irrigation_mm_day": gross}

def climate_screen(temperature_c, rainfall_mm_day, humidity_pct,
                   crop_temp_min=10, crop_temp_max=35, heavy_rain_threshold=50):
    """Transparent screening rules; not a trained/validated crop-loss prediction."""
    flags = []
    if temperature_c < crop_temp_min:
        flags.append("Temperature below the configured crop range")
    if temperature_c > crop_temp_max:
        flags.append("Temperature above the configured crop range")
    if rainfall_mm_day >= heavy_rain_threshold:
        flags.append("Heavy daily rainfall threshold reached")
    if humidity_pct >= 90:
        flags.append("Very high relative humidity")
    if not flags:
        return {"level": "Low screening signal", "flags": ["No configured threshold crossed"]}
    if len(flags) >= 2:
        level = "High screening signal"
    else:
        level = "Moderate screening signal"
    return {"level": level, "flags": flags}
