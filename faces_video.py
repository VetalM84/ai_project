import cv2


# load the faces model
faces = cv2.CascadeClassifier('faces.xml')
# use webcam
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    result = faces.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in result:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the output
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
