import os
import numpy as np
import pandas as pd

from data_pipeline import extract_city_features
from models import run_training_pipeline
from simulation.optimizer import UrbanHeatOptimizer

def main():
    print("======================================================")
    print(" BHARATIYA ANTARIKSH HACKATHON 2026")
    print(" Problem Statement 1: Urban Heat Mitigation (Gujarat)")
    print("======================================================\n")

    city_name = "Ahmedabad"
    bbox = [72.45, 22.95, 72.68, 23.12]
    csv_filename = "ahmedabad_heat_data.csv"
    start_date = "2020-01-01"
    end_date = "2025-12-30"

    if not os.path.exists(csv_filename):
        print(f"[Phase 1] Extracting Satellite & Climate Data for {city_name} via GEE")
        df = extract_city_features(
            city_name=city_name,
            bbox=bbox,
            start_date=start_date,
            end_date=end_date
        )
        df.to_csv(csv_filename, index=False)
        print(f"[Phase 1] Data extraction complete. Saved to '{csv_filename}'.\n")
    else:
        print(f"[Phase 1] Dataset '{csv_filename}' found locally. Skipping GEE extraction.\n")

    print("[Phase 2] Initiating PINN Training Pipeline")
    model, scaler = run_training_pipeline(csv_path=csv_filename, epochs=50, batch_size=32)
    print("[Phase 2] Model Training Complete.\n")

    print("[Phase 3] Running Urban Heat Optimization Engine")
    optimizer = UrbanHeatOptimizer(model=model, scaler=scaler)

    #hot_zone_features = [0.08, 0.35, 32.0, 1.2]
    df = pd.read_csv(csv_filename)
    hottest_pixel = df.sort_values(by='LST', ascending=False).iloc[0]

    hot_zone_features = [
        float(hottest_pixel['NDVI']),
        float(hottest_pixel['NDBI']),
        float(hottest_pixel['Air_Temp']),
        float(hottest_pixel['Wind_Speed'])
    ]

    print("\n--- Current High-Risk Zone Baseline ---")
    print(f"NDVI (Greenery): {hot_zone_features[0]}")
    print(f"NDBI (Concrete): {hot_zone_features[1]}")
    print(f"Air Temp: {hot_zone_features[2]}°C")
    print(f"Wind Speed: {hot_zone_features[3]} m/s")

    print("\nCalculating optimal resource deployment...")
    optimization_results = optimizer.first_optimal_intervention(
        base_feature=hot_zone_features,
        max_ndvi_increase=0.20,
        max_ndbi_decrease=0.15
    )

    print("\n--- Optimization Output Metrics ---")
    for key, val in optimization_results.items():
        formatted_key = key.replace('_', ' ').title()

        if "Celsius" in formatted_key or "Reduction" in formatted_key:
            print(f"{formatted_key}: {val} °C")
        else:
            print(f"{formatted_key}: {val}")

    print("\n======================================================")
    print(" Execution Completed")
    print("======================================================")


if __name__ == "__main__":
    main()
