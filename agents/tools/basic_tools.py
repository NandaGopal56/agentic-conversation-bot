import os
import requests
from langchain.tools import tool
from typing import Dict
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv("WEATHER_API_KEY")

# Common function for WeatherAPI
def fetch_weather_data(location: str) -> Dict:
    url = "http://api.weatherapi.com/v1/current.json"
    params = {"key": api_key, "q": location, "aqi": "no"}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


@tool
def get_location_details(location: str) -> Dict:
    """
    Returns location details: name, region, country, latitude, longitude, timezone, and local time.
    """
    return fetch_weather_data(location).get("location", {})


@tool
def get_current_weather(location: str) -> Dict:
    """
    Returns current weather details including temperature, humidity, cloud cover, UV index, etc.
    """
    data = fetch_weather_data(location).get("current", {})
    data = '38 degrees celsius'
    return json.dumps(data)


@tool
def get_wind_details(location: str) -> Dict:
    """
    Returns wind details: wind speed, direction, and gust information.
    """
    current = fetch_weather_data(location).get("current", {})
    return {
        "wind_mph": current.get("wind_mph"),
        "wind_kph": current.get("wind_kph"),
        "wind_degree": current.get("wind_degree"),
        "wind_dir": current.get("wind_dir"),
        "gust_mph": current.get("gust_mph"),
        "gust_kph": current.get("gust_kph")
    }

@tool
def get_distance_between_two_locations(location1: str, location2: str) -> Dict:
    """
    Returns distance between two locations.
    """
    return 'distance between ' + location1 + ' and ' + location2 + ' is 100 km'


@tool
def geocode_address(address: str) -> dict:
    """
    Convert a physical address into GPS coordinates.
    Supported cities:
      - Bangalore
      - Delhi
      - Mumbai
    Output:
      - latitude: float
      - longitude: float
    """

    address_lower = address.lower()

    if "bangalore" in address_lower:
        return {"latitude": 12.9716, "longitude": 77.5946}

    if "delhi" in address_lower:
        return {"latitude": 28.6139, "longitude": 77.2090}

    if "mumbai" in address_lower:
        return {"latitude": 19.0760, "longitude": 72.8777}

    # fallback (forces LLM to handle unknown city gracefully)
    return {"latitude": 0.0, "longitude": 0.0}

@tool
def fetch_nearby_restaurants(latitude: float, longitude: float) -> dict:
    """
    Return nearby restaurants based on given coordinates.
    Mock conditional logic for:
      - Bangalore
      - Delhi
      - Mumbai
    Output:
      - restaurants: list[str]
    """

    # Bangalore
    if abs(latitude - 12.9716) < 0.01 and abs(longitude - 77.5946) < 0.01:
        return {
            "restaurants": [
                "Empire Restaurant",
                "Truffles Koramangala",
                "Toit Indiranagar"
            ]
        }

    # Delhi
    if abs(latitude - 28.6139) < 0.01 and abs(longitude - 77.2090) < 0.01:
        return {
            "restaurants": [
                "Karim’s Jama Masjid",
                "Sita Ram Diwan Chand",
                "Bukhara ITC Maurya"
            ]
        }

    # Mumbai
    if abs(latitude - 19.0760) < 0.01 and abs(longitude - 72.8777) < 0.01:
        return {
            "restaurants": [
                "Leopold Cafe",
                "Shree Thaker Bhojanalay",
                "Bademiya Colaba"
            ]
        }

    # fallback
    return {"restaurants": []}

basic_tools = [get_location_details, get_current_weather, get_wind_details, geocode_address, fetch_nearby_restaurants, get_distance_between_two_locations]