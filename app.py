from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/filter', methods=['POST'])
def filter():
    # Get the uploaded image
    image = request.files['image']
    original_image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Reduce the size of the original image
    max_dimension = 500
    height, width = original_image.shape[:2]
    if height > width:
        height = int(max_dimension * height / width)
        width = max_dimension
    else:
        width = int(max_dimension * width / height)
        height = max_dimension
    original_image = cv2.resize(original_image, (width, height))

    # Get the selected filter
    filter_name = request.form['filter']

    # Apply the selected filter
    if filter_name == 'grayscale':
        filtered_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    elif filter_name == 'sobel':
        filtered_image = cv2.Sobel(original_image, cv2.CV_8U, 1, 1, ksize=3)
    elif filter_name == 'box':
        filtered_image = cv2.boxFilter(original_image, -1, (3, 3))
    elif filter_name == 'laplacian':
        filtered_image = cv2.Laplacian(original_image, cv2.CV_8U)
    elif filter_name == 'median':
        filtered_image = cv2.medianBlur(original_image, 3)
    elif filter_name == 'gaussian':
        filtered_image = cv2.GaussianBlur(original_image, (3, 3), 0)
    elif filter_name == 'bilateral':
        filtered_image = cv2.bilateralFilter(original_image, 9, 75, 75)

    # Reduce the size of the filtered image
    max_dimension = 500
    height, width = filtered_image.shape[:2]
    if height > width:
        height = int(max_dimension * height / width)
        width = max_dimension
    else:
        width = int(max_dimension * width / height)
        height = max_dimension
    filtered_image = cv2.resize(filtered_image, (width, height))

    # Encode the original and filtered images as base64 strings
    ret, original_jpeg = cv2.imencode('.jpg', original_image)
    original_base64 = base64.b64encode(original_jpeg.tobytes()).decode('utf-8')

    ret, filtered_jpeg = cv2.imencode('.jpg', filtered_image)
    filtered_base64 = base64.b64encode(filtered_jpeg.tobytes()).decode('utf-8')

    # Render the HTML page with the original and filtered images side by side
    return render_template('result.html', original=original_base64, filtered=filtered_base64)

if __name__ == '__main__':
    app.run(debug=True)
