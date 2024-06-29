import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import pickle
import matplotlib
matplotlib.use('Agg')  # 'Agg' backend to avoid GUI error
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = '12345'  # A secret key for flashing messages

# Path to the uploaded files
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static/uploads') #it returns the current working directory
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print("UPLOAD FOLDER CREATED:", UPLOAD_FOLDER)  # Debugging message
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your main model for kidney classification
model = tf.keras.models.load_model(r"model/VGG_Model1.h5")

# Load your additional model for CT_SCAN classification
ct_model = tf.keras.models.load_model(r"model/CT_Model.h5")

# Define the classes
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']
# Function to check if a filename has an allowed extension
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#rsplit is a method which is used to split a string into a list

# Function to classify whether the image is a CT_SCAN or NORMAL IMAGE
def classify_image(img):
    # Make prediction
    prediction = ct_model.predict(img)
    if prediction[0][0] > 0.5:  # Adjust this threshold according to your model's output
        return "CT_SCAN"
    else:
        return "NORMAL IMAGE"

# Function to generate accuracy and loss plots
def generate_plots(model_history, filename):
    # Plotting training & validation accuracy values
    plt.plot(model_history['accuracy'], label='Training Accuracy')
    plt.plot(model_history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    accuracy_plot = 'accuracy_plot_' + filename + '.png'
    plt.savefig(os.path.join('static', accuracy_plot))  # Save the plot as an image
    plt.close()

    # Plotting training & validation loss values
    plt.plot(model_history['loss'], label='Training Loss')
    plt.plot(model_history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    loss_plot = 'loss_plot_' + filename + '.png'
    plt.savefig(os.path.join('static', loss_plot))  # Save the plot as an image
    plt.close()

    return accuracy_plot, loss_plot


@app.route('/', methods=['GET','POST'])
def index():
    return render_template('Home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    filename = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)

            # Save the uploaded file
            try:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            except Exception as e:
                print(f"Error saving file: {e}")
                flash('Error saving file')
                return redirect(request.url)
            
            # Load the image for classification
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = image.load_img(img_path, target_size=(244, 244))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Classify the image
            image_class = classify_image(img_array)
            
            # If it's a CT_SCAN, proceed to prediction
            if image_class == "CT_SCAN":
                return redirect(url_for('predict', filename=filename))
            else:
                flash('Uploaded image is a NORMAL IMAGE. Please upload a CT SCAN.')
                return redirect(request.url)
    return render_template('upload.html')

# Prediction route
@app.route('/predict/<filename>')
def predict(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = image.load_img(img_path, target_size=(244, 244))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]

    # Delete the uploaded image after prediction
    os.remove(img_path)

    # Load the model history from the pickle file
    history_filename = r"model/VGG_Model3_history.pkl"
    with open(history_filename, 'rb') as file:
        model_history = pickle.load(file)

    # Generate accuracy and loss plots
    accuracy_plot, loss_plot = generate_plots(model_history, filename)

    # Pass filename, predicted class, and plots to result.html template
    return render_template('result.html', filename=filename, predicted_class=predicted_class, accuracy_plot=accuracy_plot, loss_plot=loss_plot)

if __name__ == '__main__':
    app.run(debug=True)