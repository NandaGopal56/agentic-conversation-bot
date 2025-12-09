import os
import uuid
import requests
from langchain.tools import tool
from typing import Dict
from dotenv import load_dotenv
import json
import cv2
import base64
import tempfile
from PIL import Image
import os
import uuid

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
def capture_camera_image() -> Image:
    """
    Tool Name: capture_camera_image

    Purpose:
    - Captures a single live frame from the system camera
    - Returns a PIL Image object that can be directly passed to LLM vision models
    - Optionally saves the image to ./temporary/ for debugging

    Returns:
    PIL.Image.Image object ready for LLM consumption
    """
    

    max_width = 320
    jpeg_quality = 70
    temp_dir = "temporary"

    # Ensure temp directory exists (for optional debugging saves)
    os.makedirs(temp_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    # Warm up camera
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Could not read frame from camera")

    # Resize while keeping aspect ratio
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))

    # Convert BGR (OpenCV format) to RGB (PIL format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    
    # Optional: Save for debugging (commented out by default)
    debug_path = os.path.join(temp_dir, f"capture_{int(uuid.uuid4())}.jpg")
    pil_image.save(debug_path, quality=jpeg_quality)
    
    return pil_image
basic_tools = [capture_camera_image, get_location_details, get_current_weather, get_wind_details]