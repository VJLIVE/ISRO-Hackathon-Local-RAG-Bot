import pickle, time, requests, json
from shapely.geometry import shape, Point
from haversine import haversine
import spacy, folium

# === Load Metadata ===
with open("vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

nlp = spacy.load("en_core_web_sm")

# === Global Cache ===
CACHE_FILE = "geocache.pkl"
cache = {}

def load_cache():
    global cache
    try:
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
            print(f"🗂️ Loaded cache with {len(cache)} places.")
    except FileNotFoundError:
        print("📂 No cache found, starting fresh.")
        cache = {}

def save_cache():
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
        print(f"💾 Cache saved with {len(cache)} places.")

def save_metadata():
    with open("vector_store/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("💾 metadata.pkl updated")

# === Step 1: Extract locations ===
def extract_locations():
    for entry in metadata:
        text = entry['content']
        doc = nlp(text)
        locs = list({ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]})
        entry['locations'] = [{"name": loc} for loc in locs]
    print("✅ Extracted location names")

# === Step 2: Enrich with lat/lon + periodic save ===
def get_coordinates(place, retries=3):
    if place in cache:
        return cache[place]

    url = "https://nominatim.openstreetmap.org/search"
    params = {'q': place, 'format': 'json', 'limit': 1}
    headers = {'User-Agent': 'MOSDAC-Bot/1.0 (your_email@example.com)'}
    for attempt in range(retries):
        try:
            res = requests.get(url, params=params, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if data:
                    lat, lon = float(data[0]['lat']), float(data[0]['lon'])
                    cache[place] = (lat, lon)
                    return lat, lon
                else:
                    print(f"⚠️ No result for {place}")
                    return None, None
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed for {place}: {e}")
        time.sleep(2)
    print(f"❌ Could not resolve {place} after {retries} attempts")
    return None, None

def enrich_with_coordinates():
    count = 0
    for entry in metadata:
        for loc in entry['locations']:
            if "lat" not in loc or loc["lat"] is None:
                lat, lon = get_coordinates(loc['name'])
                loc['lat'], loc['lon'] = lat, lon
                print(f"📍 {loc['name']} → ({lat}, {lon})")
                time.sleep(1)
                count += 1

                # Save every 10 locations
                if count % 10 == 0:
                    save_metadata()
                    save_cache()
                    print(f"🔷 Progress saved after {count} locations.")
    print("✅ Enriched all locations with lat/lon")
    save_metadata()
    save_cache()

# === Step 3: Queries ===
def find_nearby_chunks(query_place, radius_km=100):
    lat, lon = get_coordinates(query_place)
    if lat is None: return []
    results = []
    for entry in metadata:
        for loc in entry['locations']:
            if loc.get("lat") and loc.get("lon"):
                dist = haversine((lat, lon), (loc['lat'], loc['lon']))
                if dist <= radius_km:
                    results.append((entry, loc, dist))
    results.sort(key=lambda x: x[2])
    return results

def load_polygon(geojson_file):
    with open(geojson_file) as f:
        return shape(json.load(f)['features'][0]['geometry'])

def find_within_polygon(polygon_shape):
    results = []
    for entry in metadata:
        for loc in entry['locations']:
            if loc.get("lat") and loc.get("lon"):
                pt = Point(loc["lon"], loc["lat"])
                if polygon_shape.contains(pt):
                    results.append((entry, loc))
    return results

def plot_on_map(results, output_html="map.html"):
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    for entry, loc, *rest in results:
        folium.Marker(
            [loc['lat'], loc['lon']],
            popup=f"{loc['name']} — {entry['source'][:50]}"
        ).add_to(m)
    m.save(output_html)
    print(f"🗺️ Map saved to {output_html}")

# === CLI ===
if __name__ == "__main__":
    load_cache()
    print("""
=== GEO-SPATIAL PIPELINE ===

1️⃣ Extract location names
2️⃣ Enrich with lat/lon
3️⃣ Save metadata
4️⃣ Query by distance
5️⃣ Query by polygon
6️⃣ Visualize last result
0️⃣ Exit
    """)
    last_results = []

    while True:
        choice = input("Choose an option: ").strip()
        if choice == "1":
            extract_locations()
        elif choice == "2":
            enrich_with_coordinates()
        elif choice == "3":
            save_metadata()
            save_cache()
        elif choice == "4":
            place = input("Enter place name: ")
            radius = float(input("Enter radius (km): "))
            results = find_nearby_chunks(place, radius)
            for entry, loc, dist in results:
                print(f"📄 {entry['source']} — 📍 {loc['name']} @ {dist:.1f} km")
            last_results = results
        elif choice == "5":
            geojson = input("Enter path to GeoJSON: ")
            poly = load_polygon(geojson)
            results = find_within_polygon(poly)
            for entry, loc in results:
                print(f"📄 {entry['source']} — 📍 {loc['name']}")
            last_results = [(e, l, 0) for e, l in results]
        elif choice == "6":
            if last_results:
                plot_on_map(last_results)
            else:
                print("⚠️ No results to plot.")
        elif choice == "0":
            save_metadata()
            save_cache()
            break
        else:
            print("❌ Invalid choice")
