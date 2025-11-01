import cv2
import mediapipe as mp
import serial
import time

arduino = serial.Serial('COM5', 9600)
print("Hey Vaibhav... connecting to Arduino")
time.sleep(2)
print("Connected!")


eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


cap = cv2.VideoCapture(0)
smoothed_pan = 90
smoothed_tilt = 90
alpha = 0.2 

while True:
    success, frame = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    eyes_open = 0  

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)

        if len(eyes) > 0:
            eyes_open = 1
            cv2.putText(frame, "Eyes Open", (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Eyes Closed", (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0]
        nose = landmarks.landmark[1]

        
        nx, ny = int(nose.x * w), int(nose.y * h)

        
        pan = int((nx / w) * 180)
        tilt = int((ny / h) * 180)
        tilt = 180 - tilt  

        
        smoothed_pan = int(alpha * pan + (1 - alpha) * smoothed_pan)
        smoothed_tilt = int(alpha * tilt + (1 - alpha) * smoothed_tilt)

        
        data = f"{smoothed_pan} {smoothed_tilt} {eyes_open}\n"
        arduino.write(data.encode())

        
        cv2.circle(frame, (nx, ny), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"Pan: {smoothed_pan} Tilt: {smoothed_tilt}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Head Controlled Servos + Eye LED", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
