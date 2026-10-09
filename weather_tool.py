"""天气查询工具（使用 Open-Meteo，不需要额外天气 API Key）。"""

import requests
from langchain_core.tools import tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "晴朗", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "有雾", 48: "雾凇",
    51: "小毛毛雨", 53: "中等毛毛雨", 55: "浓毛毛雨",
    56: "轻微冻毛毛雨", 57: "较强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "轻微冻雨", 67: "较强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "中阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "强阵雪",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴较强冰雹",
}


def _get_json(url: str, params: dict) -> dict:
    """使用固定的官方 API 地址，避免允许模型传入任意请求 URL。"""
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("error"):
        raise ValueError("天气服务返回异常数据")
    return payload


def _at(values: list, index: int):
    """容错处理长度不一致或缺少的每日天气数据。"""
    return values[index] if isinstance(values, list) and index < len(values) else None


def _show(value, unit: str = "") -> str:
    return "暂无数据" if value is None else f"{value}{unit}"


@tool("get_weather")
def get_weather(city: str) -> str:
    """查询某个城市当前天气及未来七天的天气预报。

    当用户询问城市的当前天气、今天/明天/后天的天气、气温、降水、
    湿度、风速，或未来七天的天气预报时使用本工具。
    city 必须是明确的城市名称，如“北京”“上海”“Denver, Colorado”。
    若用户没有提供城市，应先询问城市，不要猜测其所在地。
    """
    city = city.strip() if isinstance(city, str) else ""
    if not city:
        return "缺少城市名称：请询问用户要查询哪个城市的天气。"

    try:
        location_data = _get_json(
            GEOCODING_URL,
            {"name": city, "count": 1, "language": "zh", "format": "json"},
        )
        places = location_data.get("results") or []
        if not places:
            return f"没有找到城市“{city}”，请用户提供城市和省份或国家。"

        place = places[0]
        if "latitude" not in place or "longitude" not in place:
            return f"无法解析城市“{city}”的经纬度。"

        weather = _get_json(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,weather_code,wind_speed_10m"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "timezone": "auto",
                "forecast_days": 7,
            },
        )

        name = "，".join(
            dict.fromkeys(
                part for part in (
                    place.get("name"), place.get("admin1"), place.get("country")
                ) if part
            )
        ) or city
        current = weather.get("current") or {}
        units = weather.get("current_units") or {}
        daily = weather.get("daily") or {}

        output = [f"地点：{name}（{weather.get('timezone', '当地时区')}）"]
        if current:
            desc = WEATHER_CODES.get(current.get("weather_code"), "天气状况未知")
            output.append(
                f"当前天气（{current.get('time', '时间未知')}）：{desc}；"
                f"气温 {_show(current.get('temperature_2m'), units.get('temperature_2m', '°C'))}；"
                f"体感 {_show(current.get('apparent_temperature'), units.get('apparent_temperature', '°C'))}；"
                f"湿度 {_show(current.get('relative_humidity_2m'), '%')}；"
                f"降水 {_show(current.get('precipitation'), 'mm')}；"
                f"风速 {_show(current.get('wind_speed_10m'), units.get('wind_speed_10m', 'km/h'))}。"
            )
        else:
            output.append("当前天气：接口未返回数据。")

        dates = daily.get("time") or []
        if dates:
            output.append("未来七天预报（日期为当地日期）：")
            for i, date in enumerate(dates[:7]):
                code = _at(daily.get("weather_code"), i)
                desc = WEATHER_CODES.get(code, "天气状况未知")
                high = _show(_at(daily.get("temperature_2m_max"), i), "°C")
                low = _show(_at(daily.get("temperature_2m_min"), i), "°C")
                rain = _show(_at(daily.get("precipitation_probability_max"), i), "%")
                output.append(f"{date}：{desc}，最高{high}，最低{low}，最高降水概率{rain}。")
        else:
            output.append("七天天气预报：接口未返回数据。")

        output.append("数据来源：Open-Meteo（模型预报数据，非实地实时观测）。")
        return "\n".join(output)

    except (requests.RequestException, ValueError, KeyError, TypeError):
        # 不向用户暴露服务内部的 URL、环境变量或错误堆栈。
        return "天气服务暂时不可用，请稍后重试。"
