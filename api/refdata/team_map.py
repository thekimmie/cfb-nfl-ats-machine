# api/refdata/team_map.py
TEAMS = {
    "NFL": {
        "Kansas City Chiefs": {
            "stadium": "GEHA Field at Arrowhead",
            "lat": 39.0489, "lon": -94.4839, "tz": "America/Chicago"
        },
        "Green Bay Packers": {
            "stadium": "Lambeau Field",
            "lat": 44.5013, "lon": -88.0622, "tz": "America/Chicago"
        },
        "Chicago Bears": {
            "stadium": "Soldier Field",
            "lat": 41.8623, "lon": -87.6167, "tz": "America/Chicago"
        },
        "New England Patriots": {
            "stadium": "Gillette Stadium",
            "lat": 42.0909, "lon": -71.2643, "tz": "America/New_York"
        },
        "New York Jets": {
            "stadium": "MetLife Stadium",
            "lat": 40.8136, "lon": -74.0745, "tz": "America/New_York"
        },
        "New York Giants": {
            "stadium": "MetLife Stadium",
            "lat": 40.8136, "lon": -74.0745, "tz": "America/New_York"
        },
        "Philadelphia Eagles": {
            "stadium": "Lincoln Financial Field",
            "lat": 39.9008, "lon": -75.1675, "tz": "America/New_York"
        },
        "Dallas Cowboys": {
            "stadium": "AT&T Stadium",
            "lat": 32.7473, "lon": -97.0945, "tz": "America/Chicago"
        }
    },
    "NCAA": {
        # Add as you go (only the teams you need first)
        # "Alabama Crimson Tide": {"stadium":"Bryant–Denny Stadium","lat":33.2083,"lon":-87.5504,"tz":"America/Chicago"}
    }
}

def lookup_team(league: str, team: str):
    return TEAMS.get(league, {}).get(team)
