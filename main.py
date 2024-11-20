import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No file uploaded.", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file.", 400
    
    # Save the uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Load the image
    image = cv2.imread(filepath)
    if image is None:
        return "Error loading the image. Ensure it is a valid image file.", 400
    
    # Get selected transformations
    transformations = request.form.getlist('transformations')
    if not transformations:
        return "No transformations selected.", 400

    rows, cols, _ = image.shape
    processed_images = []

    # Apply transformations based on user input
    if 'translate' in transformations:
        try:
            tx = int(request.form['tx'])
            ty = int(request.form['ty'])
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
            processed_images.append(('Translated', translated_image))
        except ValueError:
            return "Invalid translation values.", 400

    if 'rotate' in transformations:
        try:
            angle = int(request.form['angle'])
            center = (cols // 2, rows // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
            processed_images.append(('Rotated', rotated_image))
        except ValueError:
            return "Invalid rotation angle.", 400

    if 'scale' in transformations:
        try:
            scale_x = float(request.form['scale_x'])
            scale_y = float(request.form['scale_y'])
            scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y)
            processed_images.append(('Scaled', scaled_image))
        except ValueError:
            return "Invalid scale values.", 400

    if 'shear' in transformations:
        try:
            shear_x = float(request.form['shear_x'])
            shear_y = float(request.form['shear_y'])
            shearing_matrix = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
            sheared_image = cv2.warpAffine(image, shearing_matrix, (cols, rows))
            processed_images.append(('Sheared', sheared_image))
        except ValueError:
            return "Invalid shearing values.", 400

    if 'flip' in transformations:
        flipped_image = cv2.flip(image, 1)
        processed_images.append(('Flipped', flipped_image))

    if 'crop' in transformations:
        try:
            crop_x1 = int(request.form['crop_x1'])
            crop_y1 = int(request.form['crop_y1'])
            crop_x2 = int(request.form['crop_x2'])
            crop_y2 = int(request.form['crop_y2'])
            cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
            processed_images.append(('Cropped', cropped_image))
        except ValueError:
            return "Invalid crop coordinates.", 400

    # For perspective, use a hardcoded transformation matrix (can be adjusted)
    if 'perspective' in transformations:
        pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
        pts2 = np.float32([[10, 100], [180, 50], [50, 250], [200, 220]])
        perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
        perspective_image = cv2.warpPerspective(image, perspective_matrix, (cols, rows))
        processed_images.append(('Perspective', perspective_image))

    # Save processed images and prepare them for the result page
    results = []
    for name, img in processed_images:
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{name}.jpg")
        cv2.imwrite(processed_path, img)
        results.append((name, f"/processed/{name}.jpg"))

    return render_template('result.html', results=results)

@app.route('/processed/<filename>')
def serve_processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
