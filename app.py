# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load models and artifacts
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/iso_forest.pkl', 'rb') as f:
    iso_forest = pickle.load(f)

with open('models/dbscan_core_points.pkl', 'rb') as f:
    core_points = pickle.load(f)

with open('models/eps.pkl', 'rb') as f:
    eps = pickle.load(f)

feature_names = ['TransactionAmount', 'CustomerAge', 'TransactionDuration', 'LoginAttempts', 'AccountBalance']

# Reasonable bounds (can be adjusted)
BOUNDS = {
    'TransactionAmount': (0, 10000),
    'CustomerAge': (18, 100),
    'TransactionDuration': (18, 600),
    'LoginAttempts': (1, 20),
    'AccountBalance': (0, 50000)
}

def validate_input(data):
    for name in feature_names:
        val = data.get(name)
        if val is None:
            return False, f"Missing field: {name}"
        low, high = BOUNDS[name]
        if not (low <= val <= high):
            return False, f"{name} must be between {low} and {high}"
    return True, ""

def predict_new(transaction_dict):
    new_X = np.array([[transaction_dict[feat] for feat in feature_names]])
    new_X_scaled = scaler.transform(new_X)

    iso_pred = iso_forest.predict(new_X_scaled)
    is_outlier_iso = bool(iso_pred[0] == -1)

    if len(core_points) > 0:
        nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(core_points)
        dist, idx = nbrs.kneighbors(new_X_scaled)
        is_outlier_dbscan = bool(dist[0,0] > eps)
        dist_val = float(dist[0,0])
    else:
        is_outlier_dbscan = True
        dist_val = None

    return {
        'iso_forest_outlier': is_outlier_iso,
        'dbscan_outlier': is_outlier_dbscan,
        'distance_to_core': dist_val
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    required = set(feature_names)
    if not required.issubset(data.keys()):
        return jsonify({'error': 'Missing fields'}), 400
    try:
        for k in feature_names:
            data[k] = float(data[k])
    except ValueError:
        return jsonify({'error': 'Invalid numeric values'}), 400

    # Server-side validation
    valid, msg = validate_input(data)
    if not valid:
        return jsonify({'error': msg}), 400

    result = predict_new(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)