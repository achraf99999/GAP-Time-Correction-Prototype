import gpxpy
import pandas as pd
import requests
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# === CONFIGURATION ===
GPX_FILE = "10k.gpx"
API_KEY = "fc6506314f2fd8f4a9b51255069acafe"
TEMP_THRESHOLD = 20
HUMIDITY_THRESHOLD = 60
TEMP_PENALTY_RATE = 0.005
HUMIDITY_PENALTY_RATE = 0.001

# === STEP 1: Extract Data from GPX File ===
with open(GPX_FILE, 'r') as gpx_file:
    gpx = gpxpy.parse(gpx_file)

segments = []
for track in gpx.tracks:
    for segment in track.segments:
        for i in range(1, len(segment.points)):
            p1, p2 = segment.points[i - 1], segment.points[i]
            dist = p2.distance_2d(p1)
            elev_diff = p2.elevation - p1.elevation
            slope = elev_diff / dist if dist else 0
            time_diff = (p2.time - p1.time).total_seconds()
            speed = dist / time_diff if time_diff else 0
            pace = 1000 / speed if speed else 0

            segments.append({
                "lat": p2.latitude,
                "lon": p2.longitude,
                "elev": p2.elevation,
                "dist": dist,
                "elev_diff": elev_diff,
                "slope": slope,
                "time_diff": time_diff,
                "pace": pace
            })

df = pd.DataFrame(segments)

# === STEP 2: Elevation Adjustment (Minetti Model) ===
def minetti_cost(slope):
    return 155.4 * slope**5 - 30.4 * slope**4 - 43.3 * slope**3 + 46.3 * slope**2 + 19.5 * slope + 3.6

df["energy_cost"] = df["slope"].apply(minetti_cost)
flat_cost = minetti_cost(0)
df["adjusted_pace"] = df["pace"] * (df["energy_cost"] / flat_cost)

# === STEP 3: Fetch Weather Data and Apply Penalties ===
lat = df["lat"].mean()
lon = df["lon"].mean()
weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather = response.json()

penalty_factor = 1.0
if "main" in weather:
    temp = weather["main"]["temp"]
    humidity = weather["main"]["humidity"]
    temp_penalty = max(0, temp - TEMP_THRESHOLD) * TEMP_PENALTY_RATE
    humidity_penalty = max(0, humidity - HUMIDITY_THRESHOLD) * HUMIDITY_PENALTY_RATE
    penalty_factor += temp_penalty + humidity_penalty
    print(f"Température: {temp}°C, Humidité: {humidity}%, Facteur pénalité: {penalty_factor:.3f}")
else:
    print("⚠️ Données météo indisponibles. Aucun ajustement météo appliqué.")

df["weather_adjusted_pace"] = df["adjusted_pace"] * penalty_factor

# === STEP 4: Machine Learning (XGBoost) ===
# Simulated temperature/humidity to feed into model
df['temp'] = df['weather_adjusted_pace'] / df['adjusted_pace'] * TEMP_THRESHOLD
df['humidity'] = df['weather_adjusted_pace'] / df['adjusted_pace'] * HUMIDITY_THRESHOLD

features = ['slope', 'temp', 'humidity']
target = 'weather_adjusted_pace'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ RMSE du modèle XGBoost : {rmse:.3f}")

df['ml_predicted_pace'] = model.predict(X)

# === STEP 5: Time Summary ===
flat_time = df["pace"].sum() / 60
gap_time = df["adjusted_pace"].sum() / 60
weather_time = df["weather_adjusted_pace"].sum() / 60
ml_time = df["ml_predicted_pace"].sum() / 60

print("\n========== Résumé des Temps ==========")
print(f"Temps original (plat) :            {flat_time:.2f} min")
print(f"Temps ajusté (dénivelé) :          {gap_time:.2f} min")
print(f"Temps ajusté (météo) :             {weather_time:.2f} min")
print(f"Temps prédit par ML :              {ml_time:.2f} min")

# === Export Final ===
df.to_csv("prototype_gap_complet.csv", index=False)
print("✅ Données complètes sauvegardées dans prototype_gap_complet.csv")
