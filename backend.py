from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from flask_cors import CORS
import face_recognition

app = Flask(__name__)
cors = CORS(app, resources={r"/upload_images": {"origins": "*"}})

@app.route('/upload_images', methods=['POST'])
def upload_images():
    try:
        image1_file = request.files['image1']
        image2_file = request.files['image2']

        # Read the images from the uploaded files.
        image1 = cv2.imdecode(np.frombuffer(image1_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(np.frombuffer(image2_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Detect faces in the images.
        image1_face_locations = face_recognition.face_locations(image1)
        image2_face_locations = face_recognition.face_locations(image2)

        # Calculate face encodings for the detected faces.
        image1_face_encodings = face_recognition.face_encodings(image1, image1_face_locations)
        image2_face_encodings = face_recognition.face_encodings(image2, image2_face_locations)

        # Calculate the match rate as a percentage.
        if image1_face_encodings and image2_face_encodings:
            match_distances = []
            for encoding1 in image1_face_encodings:
                for encoding2 in image2_face_encodings:
                    distance = face_recognition.face_distance([encoding1], encoding2)[0]
                    match_distances.append(distance)

            # Calculate similarity score as a percentage.
            match_rate = 100 - (np.mean(match_distances) * 100)

            # Draw rectangles around faces in the images.
            for face_location in image1_face_locations:
                top, right, bottom, left = face_location
                cv2.rectangle(image1, (left, top), (right, bottom), (0, 255, 0), 2)

            for face_location in image2_face_locations:
                top, right, bottom, left = face_location
                cv2.rectangle(image2, (left, top), (right, bottom), (0, 255, 0), 2)

            # Convert the images to base64.
            image1_base64 = base64.b64encode(cv2.imencode('.png', image1)[1]).decode('utf-8')
            image2_base64 = base64.b64encode(cv2.imencode('.png', image2)[1]).decode('utf-8')
            # Return the base64-encoded images, match rate, and the images with rectangles.
            return jsonify({
                'image1': image1_base64,
                'image2': image2_base64,
                'match_rate': match_rate
            })
        else:
            # No faces found in one or both images.
            return jsonify({'error': 'No faces detected in one or both images'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
