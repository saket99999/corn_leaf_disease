from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the saved model
model = tf.keras.models.load_model('corn_leaf_disease_model.keras')

# Initialize Flask app
app = Flask(__name__)

# Define the class labels for the diseases
class_labels = ['Blight', 'Common rust', 'Grey leaf spot', 'Healthy']

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for predicting the disease
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Process the uploaded file
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((256, 256))  # Resize to model input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image

    # Predict using the loaded model
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    return jsonify({
        'prediction': predicted_class,
        'confidence': float(np.max(predictions))
    })

if __name__ == '__main__':
    app.run(debug=True)
