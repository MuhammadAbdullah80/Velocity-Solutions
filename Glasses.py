import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and Haarcascade
model = load_model("glasses_cnn_model.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
glasses_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]

        # Preprocess face for CNN prediction
        face_resized = cv2.resize(roi_color, (100, 100))
        face_array = np.expand_dims(face_resized / 255.0, axis=0)

        # Predict with CNN
        prediction = model.predict(face_array)[0][0]
        label = "Wearing Glasses" if prediction >= 0.5 else "No Glasses"

        if prediction >= 0.5:
            # Detect glasses with Haarcascade
            glasses = glasses_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)

            if len(glasses) > 0:
                # Combine all detected glasses boxes into one
                gx_min = min([gx for gx, _, _, _ in glasses])
                gy_min = min([gy for _, gy, _, _ in glasses])
                gx_max = max([gx + gw for gx, _, gw, _ in glasses])
                gy_max = max([gy + gh for _, gy, _, gh in glasses])

                # Dynamic padding
                pad_x = int(0.2 * (gx_max - gx_min))
                pad_y = int(0.4 * (gy_max - gy_min))

                x1 = max(0, gx_min - pad_x)
                y1 = max(0, gy_min - pad_y)
                x2 = min(roi_color.shape[1], gx_max + pad_x)
                y2 = min(roi_color.shape[0], gy_max + pad_y)

                # Draw red rectangle
                cv2.rectangle(roi_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw label above face
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if prediction >= 0.5 else (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Real-Time Glasses Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()
