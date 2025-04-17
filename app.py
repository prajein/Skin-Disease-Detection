# app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os
import werkzeug

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Create upload directory if it doesn't exist
if not os.path.exists("./uploadimage"):
    os.makedirs("./uploadimage")

# Create templates and static directories if they don't exist
if not os.path.exists("./templates"):
    os.makedirs("./templates")
if not os.path.exists("./static"):
    os.makedirs("./static")

# Define image size to match the model
IMG_SIZE = 32

# Try to load class names from file
def load_class_names():
    try:
        if os.path.exists("class_names.txt"):
            with open("class_names.txt", "r") as f:
                return f.read().strip().split(",")
        else:
            # Default class names if file not found
            return ['actinic keratosis', 'basal cell carcinoma', 'pigmented benign keratosis', 
                   'dermatofibroma', 'melanoma', 'melanocytic nevi', 'vascular lesion']
    except Exception as e:
        print(f"Error loading class names: {e}")
        # Default class names in case of error
        return ['actinic keratosis', 'basal cell carcinoma', 'pigmented benign keratosis', 
               'dermatofibroma', 'melanoma', 'melanocytic nevi', 'vascular lesion']

# Load the class names
class_names = load_class_names()
print(f"Loaded class names: {class_names}")

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "sd_model.keras")
try:
    model = load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Running with dummy model for testing")
    # Create a dummy model for testing if the real model isn't available
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Input

    # Create dummy model that takes inputs of shape (IMG_SIZE, IMG_SIZE, 3)
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(class_names), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Dummy model created and compiled")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if (request.method == "POST"):    
        try:
            # Save the uploaded image to a temporary location
            imagefile = request.files['file']
            filename = werkzeug.utils.secure_filename(imagefile.filename)
            img_path = os.path.join("./uploadimage", filename)
            imagefile.save(img_path)
            
            # Check if the file exists and is readable
            if not os.path.exists(img_path):
                return jsonify({'error': f'Failed to save image to {img_path}'})
                
            # Preprocess the image
            img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb")
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array.astype('float32') / 255.0

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            
            # Get top 3 predictions for more comprehensive results
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            top_predictions = [
                {"class": class_names[i], "confidence": float(predictions[0][i] * 100)} 
                for i in top_indices
            ]
            
            # Get main prediction
            predicted_label = class_names[predicted_class]
            confidence = float(predictions[0][predicted_class]) * 100
            
            print("Predictions:", predictions)
            print("Predicted class index:", predicted_class)
            print("Predicted label:", predicted_label)
            print(f"Confidence: {confidence:.2f}%")

            # Return the file path to display in the frontend
            image_url = f"/uploads/{filename}"

            return jsonify({
                'predicted_class': predicted_label,
                'confidence': f"{confidence:.2f}%",
                'image_url': image_url,
                'top_predictions': top_predictions
            })
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': f'Prediction error: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('./uploadimage', filename)

if __name__ == '__main__':
    # Get port from environment variable or use default
    app.run(host="0.0.0.0", port=10001, debug=True)

