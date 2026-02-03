import cv2
import os

# Load Haar Cascade safely
cascade_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("ERROR: Haar cascade file not loaded")
    exit()

# Start webcam
cap = cv2.VideoCapture(0)

saved = False  # save only once

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if not saved:
            face_crop = frame[y:y+h, x:x+w]

            output_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "output",
                "face.jpg"
            )

            cv2.imwrite(output_path, face_crop)
            print("Face image saved at:", output_path)
            saved = True

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
