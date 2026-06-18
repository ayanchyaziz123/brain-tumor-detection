import os
import uuid
import torch
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from PIL import Image

from utils.predictor import TumorPredictor

# Limit CPU threads on Render free tier (1 vCPU)
torch.set_num_threads(1)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MODEL_PATH'] = os.path.join('model', 'brain_tumor_efficientnet.pth')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model at startup so the first prediction isn't slow
try:
    _predictor = TumorPredictor.get_instance(app.config['MODEL_PATH'])
    print('Model loaded at startup ✓')
except FileNotFoundError as e:
    _predictor = None
    print(f'WARNING: {e}')


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if _predictor is None:
        return jsonify({'error': 'Model not loaded. Please train and upload the model file.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a JPG, PNG, or similar image.'}), 400

    try:
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(filepath)
        result = _predictor.predict(image)
        result['image_url'] = url_for('static', filename=f'uploads/{filename}')
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    status = 'ok' if _predictor is not None else 'model_missing'
    return jsonify({'status': status})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
