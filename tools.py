from typing import Dict, Any

# Mock calendar database for apartment viewings
MOCK_CALENDAR = {
    "The Vertex": ["10:00 AM", "2:00 PM", "4:00 PM"],
    "Oakwood Townhomes": ["11:00 AM", "3:00 PM"],
    "Echo Lofts": ["1:00 PM", "5:00 PM"],
    "Riverside Oasis": ["10:30 AM", "1:30 PM", "3:30 PM"]
}

def check_availability(property_name: str) -> str:
    """Checks available tour times for a given property."""
    slots = MOCK_CALENDAR.get(property_name)
    if not slots:
        return f"Sorry, we don't have automated scheduling set up for '{property_name}' yet, or the property was not found."
    return f"Available tour times for {property_name}: {', '.join(slots)}."

def draft_booking(property_name: str, date: str, time: str, user_name: str = "Prospective Tenant") -> Dict[str, Any]:
    """Drafts a tour appointment payload."""
    return {
        "status": "DRAFTED",
        "property_name": property_name,
        "date": date,
        "time": time,
        "user_name": user_name,
        "message": f"Tour drafted for {user_name} at {property_name} on {date} at {time}. Awaiting human confirmation."
    } 