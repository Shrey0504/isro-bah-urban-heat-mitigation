import ee
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geemap

try:
    ee.Initialize(project='bah2026-vector-zero')
    print("Earth Engine Initialized Successfully!")
except Exception as e:
    print("Initialization failed. Triggering re-authentication...")
    ee.Authenticate(auth_mode='localhost', force=True)
    ee.Initialize(project='bah2026-vector-zero')


def extract_city_features(city_name, bbox, start_date, end_date):
    geometry = ee.Geometry.Rectangle(bbox)

    landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(geometry).filterDate(start_date, end_date).filter(ee.Filter.lt('CLOUD_COVER', 10)).median()

    lst = landsat.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')

    ndvi = landsat.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

    ndbi = landsat.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')

    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterBounds(geometry).filterDate(start_date, end_date).median()

    air_temp = era5.select('temperature_2m').subtract(273.15).rename('Air_Temp')
    u_wind = era5.select('u_component_of_wind_10m')
    v_wind = era5.select('v_component_of_wind_10m')
    wind_speed = u_wind.hypot(v_wind).rename('Wind_Speed')

    composite = ee.Image.cat([lst, ndvi, ndbi, air_temp, wind_speed]).clip(geometry)

    sample_points = composite.sample(region=geometry, scale=100, numPixels=2500, geometries=True, tileScale=4)

    print(f"Fetching sampled data from Earth Engine servers for {city_name}")

    raw_data = sample_points.getInfo()

    if 'features' not in raw_data:
        raise ValueError("No features returned from Earth Engine. Check bounding box and date ranges.")

    features = raw_data['features']

    data = []
    for f in features:
        props = dict(f.get('properties', {}))

        if all(k in props for k in ['LST', 'NDVI', 'NDBI', 'Air_Temp', 'Wind_Speed']):
            coords = f.get('geometry', {}).get('coordinates', [None, None])
            props['Longitude'] = coords[0]
            props['Latitude'] = coords[1]
            data.append(props)

    if len(data) == 0:
        raise ValueError("Data was extracted, but none of the pixels contained all required bands.")

    df = pd.DataFrame(data)
    df['City'] = city_name
    return df, composite, geometry


def plot_data(df, city_name, out_prefix="ahmedabad_heat"):
    sns.set_style("whitegrid")
    variables = ['LST', 'NDVI', 'NDBI', 'Air_Temp', 'Wind_Speed']

    print("\n--- Summary statistics ---")
    print(df[variables].describe())

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, var in enumerate(variables):
        sns.histplot(df[var], kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(f"{var} distribution")
    axes[-1].axis('off')
    fig.suptitle(f"{city_name}: Feature Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_distributions.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    corr = df[variables].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax2)
    ax2.set_title(f"{city_name}: Feature Correlations")
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_correlation.png", dpi=150)

    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=df, x='NDVI', y='LST', ax=axes3[0], alpha=0.5, s=15, color="green")
    axes3[0].set_title("NDVI vs LST")
    sns.scatterplot(data=df, x='NDBI', y='LST', ax=axes3[1], alpha=0.5, s=15, color="firebrick")
    axes3[1].set_title("NDBI vs LST")
    fig3.tight_layout()
    fig3.savefig(f"{out_prefix}_scatter_ndvi_ndbi_lst.png", dpi=150)

    fig4, ax4 = plt.subplots(figsize=(7, 7))
    sc = ax4.scatter(df['Longitude'], df['Latitude'], c=df['LST'], cmap='inferno', s=12)
    plt.colorbar(sc, ax=ax4, label='LST (°C)')
    ax4.set_title(f"{city_name}: Spatial LST Distribution")
    ax4.set_xlabel("Longitude")
    ax4.set_ylabel("Latitude")
    fig4.tight_layout()
    fig4.savefig(f"{out_prefix}_spatial_lst.png", dpi=150)

    plt.show()
    print(f"\nPlots saved with prefix '{out_prefix}_*.png'")


def city_map(composite, geometry, df, city_name="Ahmedabad", out_html="ahmedabad_map.html"):
    Map = geemap.Map()
    Map.centerObject(geometry, zoom=11)
    Map.addLayer(geometry, {}, "AOI boundary", opacity=0.0)

    lst_vis = {'min': 20, 'max': 50, 'palette': ['blue', 'yellow', 'orange', 'red']}
    ndvi_vis = {'min': -0.2, 'max': 0.8, 'palette': ['brown', 'yellow', 'green']}
    ndbi_vis = {'min': -0.5, 'max': 0.5, 'palette': ['blue', 'white', 'red']}

    Map.addLayer(composite.select('LST'), lst_vis, "Land Surface Temp (°C)")
    Map.addLayer(composite.select('NDVI'), ndvi_vis, "NDVI (vegetation)", shown=False)
    Map.addLayer(composite.select('NDBI'), ndbi_vis, "NDBI (built-up)", shown=False)

    points_fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([row['Longitude'], row['Latitude']]), {'LST': row['LST']})
        for _, row in df.iterrows()
    ])
    Map.addLayer(points_fc.style(color='black', pointSize=2), {}, "Sample points", shown=False)

    Map.add_colorbar(lst_vis, label="LST (°C)")
    Map.addLayerControl()

    Map.to_html(out_html)
    print(f"Interactive map saved to {out_html} — open it in a browser.")

    return Map


if __name__ == "__main__":
    ahmedabad_bbox = [72.45, 22.95, 72.68, 23.12]
    df_ahmedabad, composite_ahmedabad, geometry_ahmedabad = extract_city_features(
        city_name="Ahmedabad",
        bbox=ahmedabad_bbox,
        start_date="2025-01-01",
        end_date="2025-12-30"
    )
    df_ahmedabad.to_csv("ahmedabad_heat_data.csv", index=False)
    print(f"Successfully collected {len(df_ahmedabad)} data points for Ahmedabad.")

    plot_data(df_ahmedabad, city_name="Ahmedabad")

    city_map(composite_ahmedabad, geometry_ahmedabad, df_ahmedabad, city_name="Ahmedabad")
