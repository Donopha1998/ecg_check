from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import joblib
import os
from flask_cors import CORS
app = Flask(__name__)
CORS(app) 
class CustomRMSprop(RMSprop):
    pass


model_image = load_model('my_model.h5', custom_objects={'CustomRMSprop': CustomRMSprop})
model_image = load_model('my_model.h5', custom_objects={'RMSprop': RMSprop})
model_data = joblib.load('model.pkl')

class_indices = {'Left Bundle Branch Block': 0, 'Normal': 1, 'Premature Atrial Contraction': 2, 'Premature Ventricular Contractions': 3, 'Right Bundle Branch Block': 4, 'Ventricular Fibrillation': 5}
index_to_class = {v: k for k, v in class_indices.items()}

@app.route('/predict_image', methods=['POST'])
def predict_image():
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    prediction = model_image.predict(img_array_expanded_dims)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = index_to_class[predicted_class_index]
    return predicted_class_name

@app.route('/predict_data', methods=['POST'])
def predict_data():

    data = request.get_json(force=True)
    

    predictions = model_data.predict([list(data.values())])

    return jsonify(predictions.tolist())


@app.route('/')
def hello_world():
    return 'Hello,!'


if __name__ == '__main__':
   app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 6000)))
