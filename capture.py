import cv2
import os
save_path = f"images/train/arshiga"
os.makedirs(save_path, exist_ok=True)
count=0
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Press SPACE to Capture", frame)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        face_img=frame[y:y+h,x:x+w]

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == 32:
        img_path = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(img_path, cv2.resize(face_img, (128, 128)))
        print(f"Saved: {img_path}")
        count += 1

cap.release()
cv2.destroyAllWindows()