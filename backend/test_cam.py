import cv2

cap = cv2.VideoCapture(0)  # если не откроется — попробуем 1
while True:
    ret, frame = cap.read()
    if not ret:
        print("Нет кадра")
        break

    cv2.imshow("Camera test", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()