from flask import Flask, request, jsonify
import numpy as np
import json
import model
import cv2
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)
EMOTIONS = ["happy", "neutral", "sad", "surprise"]
cnn_model = model.Initialize_model()

def detect_emotion(image_array):
    # Here, you should integrate your neural network model to detect emotions
    # For now, this function returns a dummy response
    # cnn_model = model.Initialize_model()
    # use model.predict to get the emotions
    processed_image = image_array.reshape(1, 48, 48, 1)
    emotions = cnn_model.predict(processed_image)
    # get index of the max value
    index = np.argmax(emotions)
    return EMOTIONS[index]

def resize_image(image_array):
    # Here, you should integrate your neural network model to detect emotions
    # For now, this function returns a dummy response
    image_array = np.array(image_array).astype('float32')
    image_array /= 255.0
    image_array = cv2.resize(image_array, (48, 48))

    return image_array

def rgb_to_grayscale(rgb_array):
    grayscale_array = [[0 for _ in range(len(rgb_array[0]))] for _ in range(len(rgb_array))]

    for y in range(len(rgb_array)):
        for x in range(len(rgb_array[y])):
            r, g, b = rgb_array[y][x]
            grayscale = 0.299 * r + 0.587 * g + 0.114 * b
            grayscale_array[y][x] = int(grayscale)

    return grayscale_array


@app.route('/get-image', methods=['GET'])
def get_image():
    image_path = '7443.jpg'
    image_array = cv2.imread(image_path)
    # to grey scale
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    image_array = resize_image(image_array)
    return jsonify(image_array.tolist())

@app.route('/detect-emotion', methods=['POST'])
@cross_origin()
def emotion_detection():
    try:
        # Parse the incoming data as a JSON
        data = request.json

        image_array = rgb_to_grayscale(data['image'])
        image_array = np.array(image_array)
        image_array = resize_image(image_array)

        emotion = detect_emotion(image_array)

        # Return the response
        return jsonify(emotion)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=8080, debug=True)