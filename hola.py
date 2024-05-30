from flask import Flask, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
import anthropic

app = Flask(__name__)
client = anthropic.Anthropic(
    api_key="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",
)

DATASET_PATH = 'datasetlimpiocovid19.xlsx'

def load_model():
    df = pd.read_excel(DATASET_PATH)
    X = df.drop('Flag_sospechoso', axis=1)
    y = df['Flag_sospechoso']
    model = DecisionTreeClassifier()
    model.fit(X, y)
    joblib.dump(model, 'covid_model.pkl')

if not os.path.exists('covid_model.pkl'):
    load_model()

model = joblib.load('covid_model.pkl')

def generate_recommendations(probability):
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
    except Exception as e:
        return jsonify({"error": "Por favor envíe un JSON válido"}), 400

    if not data:
        return jsonify({"error": "Por favor, rellene los campos correctamente. No se enviaron datos."}), 400

    expected_columns = ['tos', 'cefalea', 'congestion_nasal', 'dificultad_respiratoria', 'dolor_garganta', 'fiebre', 'diarrea', 'nauseas', 'anosmia_hiposmia', 'dolor_abdominal', 'dolor_articulaciones', 'dolor_muscular', 'dolor_pecho']
    for col in expected_columns:
        if col not in data:
            return jsonify({"error": f"Falta el campo: {col}"}), 400

    df = pd.DataFrame(data, index=[0])
    

    try:
        prediction_probability = model.predict_proba(df)[0][1]
        prediction_percentage = round(prediction_probability * 100, 1)
        recommendations = generate_recommendations(prediction_percentage)
        return jsonify({"probabilidad_covid": f"{prediction_percentage}%", "recomendaciones": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)