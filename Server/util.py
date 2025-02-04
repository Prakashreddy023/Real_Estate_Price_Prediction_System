import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def load_saved_artifacts():
    global __data_columns
    global __locations
    global __model

    print("Loading saved artifacts...")

    try:
        with open("./artifacts/columns.json", "r") as f:
            __data_columns = json.load(f)['data_columns']
            __locations = __data_columns[3:]  # All location columns start from index 3

        with open("./artifacts/banglore_house_price_prediction_model.pickle","rb") as f:
            __model = pickle.load(f)  # Corrected: using "rb" instead of "wb"
        print("Loading saved artifacts...done")

    except Exception as e:
        print(f"Error loading artifacts: {e}")

def get_estimated_price(location, sqft, bhk, bath):
    global __data_columns, __model

    if __data_columns is None or __model is None:
        raise Exception("Model or data columns are not loaded. Call load_saved_artifacts() first.")

    try:
        loc_index = __data_columns.index(location.lower()) if location.lower() in __data_columns else -1
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

def get_location_names():
    global __locations
    return __locations if __locations else []

def get_data_columns():
    global __data_columns
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))  # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location
