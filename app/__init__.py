import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image
from flask import jsonify
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
from torch import no_grad
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'app/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            predicted_dog_breed = predict_dog_breed(file_path)

            return render_template('index.html', filename=filename, predicted_dog_breed=predicted_dog_breed)

    return render_template('index.html')

def predict_dog_breed(image_path):
    model_name = "Pavarissy/ConvNextV2-large-DogBreed"
    preprocessor = AutoImageProcessor.from_pretrained(model_name)
    model = ConvNextV2ForImageClassification.from_pretrained(model_name)

    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return jsonify({"error": "Error opening image"}), 500

    inputs = preprocessor(image, return_tensors="pt")
    with no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    predicted_dog_breed = model.config.id2label[predicted_label]

    return predicted_dog_breed

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    parent_directory = os.path.abspath(os.path.join(app.root_path, os.pardir))
    return send_from_directory(os.path.join(parent_directory, app.config['UPLOAD_FOLDER']), filename)

if __name__ == '__main__':
    app.run(debug=True)
