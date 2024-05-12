import json
import urllib.request
def get_weather_data(city, api_key):
    base_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{urllib.parse.quote(city)}"
    full_url = f"{base_url}?unitGroup=metric&key={api_key}&contentType=json"
    try:
        response = urllib.request.urlopen(full_url)
        data = json.load(response)
        return data
    except urllib.error.HTTPError as e:
        return f"HTTP Error: {e.code} {e.read().decode()}"
    except urllib.error.URLError as e:
        return f"URL Error: {e.reason}"