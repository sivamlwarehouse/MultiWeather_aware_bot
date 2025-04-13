import pandas as pd
import random

# Sample values
locations = ['Vijayawada', 'Hyderabad', 'Chennai', 'Mumbai', 'Bangalore']
conditions = ['Sunny', 'Cloudy', 'Rainy', 'Partly Cloudy', 'Stormy']

# Generate 5 entries
data = []
for i in range(1, 6):
    location = random.choice(locations)
    temp_c = random.randint(18, 35)
    humidity = random.randint(40, 90)
    wind = random.randint(5, 25)
    condition = random.choice(conditions)
    
    # Generate day-wise random temperatures
    days = [f"{random.randint(18, 35)}°C" for _ in range(5)]

    row = {
        "Sno": i,
        "Location": location,
        "Temperature": f"{temp_c}°C",
        "temperature_c": temp_c,
        "humidity_percent": humidity,
        "wind_kph": wind,
        "condition": condition,
        "Day-1": days[0],
        "Day-2": days[1],
        "Day-3": days[2],
        "Day-4": days[3],
        "Day-5": days[4],
    }
    data.append(row)

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv("weather_data.csv", index=False)

print("✅ Weather data saved to 'formatted_weather_data.csv'")
