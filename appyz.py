import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, request
from recognisablity import predictor_alphabet, predictor_number  # Import both predictor functions

app = Flask(__name__)

# OpenCV VideoCapture for live video feed
camera = cv2.VideoCapture(0)
prediction_result = ""  # To store the prediction result
mode = "alphabet"  # Default mode for recognition (can switch between 'alphabet' and 'number')

# Global HSV parameters
hsv_params = {
    "l_h": 0, "l_s": 0, "l_v": 0,
    "u_h": 179, "u_s": 255, "u_v": 255
}

@app.route("/update_hsv", methods=["POST"])
def update_hsv():
    """Update HSV values."""
    global hsv_params
    data = request.json
    hsv_params.update({key: int(value) for key, value in data.items()})
    return jsonify({"message": "HSV updated", "hsv_params": hsv_params})

@app.route("/switch_mode", methods=["POST"])
def switch_mode():
    """Switch between alphabet and number recognition."""
    global mode
    data = request.json
    if data.get("mode") in ["alphabet", "number"]:
        mode = data["mode"]
        return jsonify({"message": f"Switched to {mode} mode"})
    return jsonify({"error": "Invalid mode"}), 400

def gen_frames():
    """Generate frames for live video feed."""
    global prediction_result, hsv_params, mode
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), 2)
            imcrop = frame[102:298, 427:623]
            hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)

            # Apply the mask using dynamic HSV parameters
            lower_hsv = np.array([hsv_params["l_h"], hsv_params["l_s"], hsv_params["l_v"]])
            upper_hsv = np.array([hsv_params["u_h"], hsv_params["u_s"], hsv_params["u_v"]])
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

            # Show the mask in a separate window
            cv2.imshow("Mask Window", mask)

            img_name = "static/temp.png"
            save_img = cv2.resize(mask, (64, 64))
            cv2.imwrite(img_name, save_img)

            # Predict based on the selected mode
            if mode == "alphabet":
                prediction_result = predictor_alphabet()
            else:
                prediction_result = predictor_number()

            # Display the original video frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Close the mask window on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the windows
    camera.release()
    cv2.destroyAllWindows()

@app.route("/")
def index():
    """Home page with sign list and mode switching."""
    signs = [{"letter": chr(i), "image": f"/static/signs/{chr(i).lower()}.jpeg"} for i in range(ord('A'), ord('Z') + 1)]
    return render_template("index.html", signs=signs, mode=mode)

@app.route("/video_feed")
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/get_prediction")
def get_prediction():
    """Route to get the current prediction."""
    return jsonify({"prediction": prediction_result, "mode": mode})

if __name__ == "__main__":
    app.run(debug=True)
