import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model for numbers
number_classifier = load_model('Trainedmodel.h5')

# Predictor function for numbers
def predictor():
    """Predicts the number from the saved image."""
    test_image = image.load_img('static/temp.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    
    result = number_classifier.predict(test_image)
    print(result)

    # Map prediction to numbers (0-9)
    labels = [str(i) for i in range(10)]    #labels = [str(i) for i in range(10)]
    predicted_label = labels[np.argmax(result)]
    return predicted_label

# Camera logic encapsulated in a function (same as alphabets but only for numbers)
def start_camera():
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
        
        img_name = "1.png"
        save_img = cv2.resize(mask, (64, 64))
        cv2.imwrite(img_name, save_img)
        
        # Get the predicted number
        img_text = predictor()

        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
            break

    cam.release()
    cv2.destroyAllWindows()
