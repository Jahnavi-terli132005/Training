cache = {}

def get_data(key):
    data = cache.get(key)
    if data is not None:
        return data
    
    data = fetch_from_db(key)
    cache[key] = data
    return data
