import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load both models
alphabet_model = load_model('Trained_model.h5')
number_model = load_model('Trainedmodel.h5')

# Labels
alphabet_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
number_labels = [str(i) for i in range(0, 10)]

# Predictor function for alphabets
def predictor_alphabet():
    test_image = image.load_img('static/temp.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = alphabet_model.predict(test_image)
    predicted_label = alphabet_labels[np.argmax(result)]
    return predicted_label

# Predictor function for numbers
def predictor_number():
    test_image = image.load_img('static/temp.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = number_model.predict(test_image)
    predicted_label = number_labels[np.argmax(result)]
    return predicted_label

# Camera logic for real-time sign recognition
def start_camera(mode='alphabet'):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, lambda x: None)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, lambda x: None)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, lambda x: None)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, lambda x: None)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, lambda x: None)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, lambda x: None)

    img_text = ''
    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        # Get trackbar values
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), 2)
        lower_blue = np.array([l_h, l_s, l_v])
        upper_blue = np.array([u_h, u_s, u_v])
        imcrop = img[102:298, 427:623]
        hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
        cv2.imshow("test", frame)
        cv2.imshow("mask", mask)

        # Save and predict
        img_name = "static/temp.png"
        save_img = cv2.resize(mask, (64, 64))
        cv2.imwrite(img_name, save_img)

        # Predict based on selected mode
        if mode == 'alphabet':
            img_text = predictor_alphabet()
        else:
            img_text = predictor_number()

        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
            break

    cam.release()
    cv2.destroyAllWindows()
