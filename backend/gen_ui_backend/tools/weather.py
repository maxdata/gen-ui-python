from typing import Optional
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

class WeatherInput(BaseModel):
    city: str = Field(..., description="The city name to get weather for")
    state: str = Field(
        ..., description="The two letter state abbreviation to get weather for"
    )
    country: Optional[str] = Field(
        "usa", description="The two letter country abbreviation to get weather for"
    )

@tool("weather-data", args_schema=WeatherInput, return_direct=True)
def weather_data(city: str, state: str, country: str = "usa") -> dict:
    """Get the current temperature for a city using dummy data."""
    # Dummy data for demonstration purposes
    dummy_weather_data = {
        "New York": {"temperature": 72},
        "Los Angeles": {"temperature": 75},
        "Chicago": {"temperature": 68},
        "Houston": {"temperature": 80},
        "Phoenix": {"temperature": 90},
        "Philadelphia": {"temperature": 70},
        "San Antonio": {"temperature": 82},
        "San Diego": {"temperature": 73},
        "Dallas": {"temperature": 85},
        "San Jose": {"temperature": 70},        
    }

    # Default temperature if city is not in dummy data
    default_temperature = 70

    # Get temperature from dummy data or use default
    temperature = dummy_weather_data.get(city, {"temperature": default_temperature})["temperature"]

    return {
        "city": city,
        "state": state,
        "country": country,
        "temperature": temperature,
    }
