import os
import requests
from langchain_core.tools import tool


# 1. 도구 정의 (간단한 날씨 확인 함수 예)
@tool
def get_weather(location: str) -> str:
    """Query current weather information for a given location."""
    # API 키 설정
    api_key = os.getenv("OPEN_WEATHER_MAP_API_KEY")  # 구글 코랩 사용 시
    if not api_key:
        return "OpenWeatherMap API 키가 설정되지 않았습니다."
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    location = "Seoul,KR"
    params = {"q": location, "appid": api_key, "units": "metric"}
    try:
        # API 요청 및 응답 처리
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        weather_description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        return f"{location}의 현재 날씨: {weather_description}, 온도: {temperature}°C"
    except requests.exceptions.RequestException as e:
        return f"날씨 정보를 가져오는데 실패했습니다. 오류: {str(e)}"

