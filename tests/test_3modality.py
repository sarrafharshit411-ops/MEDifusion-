import requests
import json
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(project_root, "frontend", "sample_xray.png")

with open(img_path, "rb") as f:
    files = {
        'file': ('sample_xray.png', f, 'image/png'),
        'symptoms': (None, json.dumps({'fever': 1, 'cough': 1, 'rapid_breathing': 1, 'fatigue': 1})),
        'clinical_data': (None, json.dumps({
            'age': 55, 'temperature': 39.0, 'heart_rate': 104,
            'bp_systolic': 130, 'bp_diastolic': 85, 'respiratory_rate': 26,
            'oxygen_saturation': 91
        }))
    }
    res = requests.post('http://127.0.0.1:8000/api/predict/multimodal', files=files)
    print('3-Modality Status code:', res.status_code)
    data = res.json()
    print('Modalities used:', data['modalities_used'])
    print('Predicted class:', data['predicted_class'])
    print('Confidence:', data['confidence'])
    print('Gating weights:', data.get('gating_weights'))
