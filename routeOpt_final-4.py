
import folium
import requests
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import polyline
import concurrent.futures
import osmnx as ox
import networkx as nx
from branca.element import Template, MacroElement
import math


# Enable caching to speed up repeated API requests
ox.settings.use_cache = True
ox.settings.log_console = True


# Google Maps API Key (replace with your key)
GOOGLE_MAPS_API_KEY = "########################"

# Local Data Files
CRIME_DATA_FILE = "/Users/peiyan/Documents/GWU/25_Spring/Advanced Software Paradigms/HW/final/new project/Crashes_in_DC.geojson"
POI_DATA_FILE = "/Users/peiyan/Documents/GWU/25_Spring/Advanced Software Paradigms/HW/final/new project/Points_of_Interest.geojson"

def geocode_address(address):
    """
    Geocode an address string using the OpenStreetMap Nominatim API.
    Returns a tuple: (latitude, longitude)
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json"}
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        results = response.json()
        if results:
            lat = float(results[0]['lat'])
            lon = float(results[0]['lon'])
            return (lat, lon)
    return None

def fetch_crime_data():
    try:
        crime_data = gpd.read_file(CRIME_DATA_FILE)
        crime_data["date"] = pd.to_datetime(crime_data["REPORTDATE"], errors='coerce')
        latest_date = crime_data["date"].max()
        # Filter to the last 30 days
        crime_data = crime_data[crime_data["date"] >= (latest_date - pd.Timedelta(days=30))]
        # Convert to metric projection (EPSG:3857) for display purposes
        crime_data = crime_data.to_crs(epsg=3857)
        print("Filtered crime data to the last 30 days.")
        return crime_data
    except Exception as e:
        print("Error loading crime data:", e)
        return None

def get_scenic_waypoint(start, end):
    """
    Load the POI data and pick the POI that is closest to the midpoint between start and end.
    Returns a tuple: (latitude, longitude, poi_name)
    """
    try:
        poi_data = gpd.read_file(POI_DATA_FILE)
        # Ensure the CRS is defined
        if poi_data.crs is None:
            poi_data.set_crs(epsg=4326, inplace=True)
        
        # Compute the midpoint between start and end (note: start and end are in (lat, lon))
        mid_lat = (start[0] + end[0]) / 2
        mid_lon = (start[1] + end[1]) / 2
        mid_point = Point(mid_lon, mid_lat)
        mid_gs = gpd.GeoSeries([mid_point], crs="EPSG:4326")
        
        # Project both POI data and midpoint to a metric CRS for accurate distance calculations
        poi_projected = poi_data.to_crs(epsg=3857)
        mid_projected = mid_gs.to_crs(epsg=3857)[0]
        
        # Compute the distance between each POI and the midpoint
        poi_projected["dist"] = poi_projected.geometry.distance(mid_projected)
        best_poi = poi_projected.sort_values("dist").iloc[0]
        
        # Use the "NAME" property from your POI data
        poi_name = best_poi["NAME"] if "NAME" in best_poi and pd.notnull(best_poi["NAME"]) else "Unknown POI"
        
        # Reproject the best POI geometry to EPSG:4326 for mapping (lat, lon)
        best_poi_geo = gpd.GeoSeries([best_poi.geometry], crs="EPSG:3857").to_crs(epsg=4326)[0]
        return (best_poi_geo.y, best_poi_geo.x, poi_name)
    except Exception as e:
        print("Error loading scenic POI data:", e)
        return None




def get_google_route(start, end, mode="BICYCLE", waypoints=None):
    """
    Get a route from the Google Routes API.
    Returns a tuple: (route_coords, navigation_instructions)
    """
    base_url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "routes.polyline,routes.legs,routes.legs.steps"
    }
    payload = {
        "origin": {"location": {"latLng": {"latitude": start[0], "longitude": start[1]}}},
        "destination": {"location": {"latLng": {"latitude": end[0], "longitude": end[1]}}},
        "travelMode": mode
    }
    if waypoints:
        payload["intermediates"] = [
            {"location": {"latLng": {"latitude": wp[0], "longitude": wp[1]}}} for wp in waypoints
        ]
    
    response = requests.post(base_url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if not data.get("routes"):
            print("No routes found.")
            return ([], [])
        route = data["routes"][0]
        route_polyline = route["polyline"]["encodedPolyline"]
        route_coords = polyline.decode(route_polyline)
        instructions = []
        for leg in route.get("legs", []):
            for step in leg.get("steps", []):
                inst = step.get("navigationInstruction") or step.get("htmlInstructions") or ""
                if inst:
                    instructions.append(inst)
        return (route_coords, instructions)
    else:
        print("Error getting route:", response.json())
        return ([], [])



def get_safe_route_custom(start, end, crime_data, penalty=100, buffer_distance=50, margin=0.02):
    """
    Compute a safe route from start to end using OSMnx.
    Returns a list of (lat, lon) coordinates.
    """
    lat1, lon1 = start
    lat2, lon2 = end
    north = max(lat1, lat2) + margin
    south = min(lat1, lat2) - margin
    east = max(lon1, lon2) + margin
    west = min(lon1, lon2) - margin
    
    print("Downloading bike network from OSM...")
    G = ox.graph_from_bbox(north, south, east, west, network_type='bike')
    print("Projecting graph for accurate distance calculations...")
    G_proj = ox.project_graph(G)
    
    crime_proj = crime_data.to_crs(G_proj.graph['crs'])
    edges = ox.graph_to_gdfs(G_proj, nodes=False, edges=True)
    crime_counts = []
    for idx, row in edges.iterrows():
        edge_buffer = row['geometry'].buffer(buffer_distance)
        possible_idxs = list(crime_proj.sindex.intersection(edge_buffer.bounds))
        possible_crimes = crime_proj.iloc[possible_idxs]
        count = possible_crimes[possible_crimes.intersects(edge_buffer)].shape[0]
        crime_counts.append(count)
    edges['crime_count'] = crime_counts
    edges['safe_cost'] = edges['length'] + penalty * edges['crime_count']
    
    for u, v, key, data in G_proj.edges(keys=True, data=True):
        try:
            safe_cost = edges.loc[(u, v, key)]['safe_cost']
            data['safe_cost'] = safe_cost
        except KeyError:
            data['safe_cost'] = data.get('length', 1)
    
    start_point = ox.projection.project_geometry(Point(start[1], start[0]), to_crs=G_proj.graph['crs'])[0]
    end_point = ox.projection.project_geometry(Point(end[1], end[0]), to_crs=G_proj.graph['crs'])[0]
    start_node = ox.distance.nearest_nodes(G_proj, X=start_point.x, Y=start_point.y)
    end_node = ox.distance.nearest_nodes(G_proj, X=end_point.x, Y=end_point.y)
    
    try:
        route_nodes = nx.shortest_path(G_proj, start_node, end_node, weight='safe_cost')
    except nx.NetworkXNoPath:
        print("No safe path found.")
        return []
    
    safe_route_coords = []
    for node in route_nodes:
        point_proj = Point(G_proj.nodes[node]['x'], G_proj.nodes[node]['y'])
        point_latlon = ox.projection.project_geometry(point_proj, crs=G_proj.graph['crs'], to_crs="EPSG:4326")[0]
        safe_route_coords.append((point_latlon.y, point_latlon.x))
    
    return safe_route_coords

# -----------------------------
# Helper functions to generate turn-by-turn instructions
# -----------------------------
def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    Returns the bearing in degrees (0° to 360°).
    """
    lat1, lon1 = math.radians(pointA[0]), math.radians(pointA[1])
    lat2, lon2 = math.radians(pointB[0]), math.radians(pointB[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - (math.sin(lat1)*math.cos(lat2)*math.cos(dlon))
    initial_bearing = math.degrees(math.atan2(x, y))
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def haversine(pointA, pointB):
    """
    Calculate the great-circle distance between two points (in meters).
    """
    R = 6371000  # radius of Earth in meters
    lat1, lon1 = math.radians(pointA[0]), math.radians(pointA[1])
    lat2, lon2 = math.radians(pointB[0]), math.radians(pointB[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

def get_turn_by_turn_instructions(route_coords):
    """
    Generate simple turn-by-turn instructions for a route (list of (lat, lon)).
    """
    instructions = ["Start at your starting point."]
    threshold = 15  # degrees: if change in bearing is less than this, consider it 'straight'
    
    for i in range(1, len(route_coords) - 1):
        p_prev = route_coords[i - 1]
        p_curr = route_coords[i]
        p_next = route_coords[i + 1]
        
        bearing1 = calculate_initial_compass_bearing(p_prev, p_curr)
        bearing2 = calculate_initial_compass_bearing(p_curr, p_next)
        
        # Compute turning angle (normalize to -180...+180)
        turn_angle = (bearing2 - bearing1 + 360) % 360
        if turn_angle > 180:
            turn_angle -= 360
        
        # Compute distance from previous point to current point
        dist = haversine(p_prev, p_curr)
        
        instruction = f"After {dist:.0f} m, "
        if abs(turn_angle) < threshold:
            instruction += "continue straight."
        elif turn_angle > 0:
            instruction += f"turn right by {abs(turn_angle):.0f}°."
        else:
            instruction += f"turn left by {abs(turn_angle):.0f}°."
        instructions.append(instruction)
    instructions.append("You have arrived at your destination.")
    return instructions

# -----------------------------
# Functions to add navigation panel and generate maps
# -----------------------------
def add_navigation_panel(map_obj, instructions):
    """
    Add a static floating panel to the map displaying navigation instructions.
    """
    template = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 300px;
        height: 300px;
        z-index:9999;
        background-color: white;
        overflow: auto;
        padding: 10px;
        border:2px solid grey;">
      <h4>Navigation Instructions</h4>
      <ol>
      {% for step in this.instructions %}
         <li>{{ step|safe }}</li>
      {% endfor %}
      </ol>
    </div>
    {% endmacro %}
    """
    macro = MacroElement()
    macro._template = Template(template)
    macro.instructions = instructions
    map_obj.get_root().add_child(macro)

def generate_map(route_coords, crime_data, filename, color, scenic_poi=None, instructions=None):
    """
    Generate a Folium map showing the route.
    If navigation instructions are provided, a floating panel is added.
    """
    if not route_coords:
        print(f"No route coordinates provided for {filename}; skipping map generation.")
        return
    
    route_map = folium.Map(location=route_coords[0], zoom_start=14)
    folium.PolyLine(route_coords, color=color, weight=5, opacity=0.7).add_to(route_map)
    
    folium.Marker(
        location=route_coords[0],
        icon=folium.Icon(color='red', icon='play'),
        popup="Start"
    ).add_to(route_map)
    
    folium.Marker(
        location=route_coords[-1],
        icon=folium.Icon(color='red', icon='stop'),
        popup="End"
    ).add_to(route_map)
    
    if scenic_poi is None:
        # If no scenic poi is provided, add crime data markers instead
        for _, row in crime_data.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=3,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6
            ).add_to(route_map)
    else:
        # scenic_poi is expected as a tuple: (lat, lon, poi_name)
        poi_name = scenic_poi[2] if (isinstance(scenic_poi, tuple) and len(scenic_poi) == 3) else "Scenic POI"
        folium.Marker(
            location=[scenic_poi[0], scenic_poi[1]],
            popup=f'<b>🚩 {poi_name}</b>',
            tooltip=poi_name,
            icon=folium.Icon(color='orange', icon='flag')
        ).add_to(route_map)
    
    if instructions:
        add_navigation_panel(route_map, instructions)
    
    route_map.save(filename)
    print(f"Map saved as {filename}")


def main():
    start_address = input("Enter the start address: ")
    end_address = input("Enter the end address: ")
    
    start = geocode_address(start_address)
    end = geocode_address(end_address)
    
    if start is None or end is None:
        print("Error geocoding addresses. Please check your input.")
        return
    
    print(f"Start coordinates: {start}")
    print(f"End coordinates: {end}")
    
    print("Fetching crime data...")
    crime_data = fetch_crime_data()
    if crime_data is None:
        print("Error loading local crime data.")
        return
    
    scenic_waypoint = get_scenic_waypoint(start, end)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_baseline = executor.submit(get_google_route, start, end, "BICYCLE")
        future_scenic  = executor.submit(get_google_route, start, end, "BICYCLE",
                                          waypoints=[(scenic_waypoint[0], scenic_waypoint[1])] if scenic_waypoint else None)
        baseline_route, baseline_instructions = future_baseline.result()
        scenic_route, scenic_instructions   = future_scenic.result()
    
    safe_route = get_safe_route_custom(start, end, crime_data, penalty=100, buffer_distance=50, margin=0.02)
    # Generate turn-by-turn instructions for the safe route using the custom algorithm
    safe_instructions = get_turn_by_turn_instructions(safe_route) if safe_route else []
    
    if baseline_route:
        generate_map(baseline_route, crime_data, "baseline_route.html", "blue", instructions=baseline_instructions)
    else:
        print("Error getting baseline route.")
    
    if safe_route:
        generate_map(safe_route, crime_data, "safest_route.html", "red", instructions=safe_instructions)
    else:
        print("Error getting safe route.")
    
    if scenic_route:
        generate_map(scenic_route, crime_data, "scenic_route.html", "green", scenic_poi=scenic_waypoint, instructions=scenic_instructions)
    else:
        print("Error getting scenic route.")

if __name__ == "__main__":
    main()




