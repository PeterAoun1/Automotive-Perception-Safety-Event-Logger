import cv2

cascade = cv2.CascadeClassifier("cars.xml")
cap = cv2.VideoCapture("traffic_3.mp4")  # or 0 for webcam
fps = cap.get(cv2.CAP_PROP_FPS) or 30

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,60))
    for (x,y,w,h) in dets:
        # draw bbox and bottom-center point (ground contact)
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        bx, by = x + w//2, y + h  # bottom-center
        cv2.circle(frame, (bx,by), 3, (0,0,255), -1)
    cv2.putText(frame, f"detections: {len(dets)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
    cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()