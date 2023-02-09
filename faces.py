import cv2

# load the faces model
faces = cv2.CascadeClassifier('faces.xml')
# read the image
img = cv2.imread('media/people.jpg')
# resize the image
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
# convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# detect the faces
results = faces.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=2)

# draw the rectangle around each face
for (x, y, w, h) in results:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# display the image
cv2.imshow('Detected', img)
# wait for a key to be pressed to close the window
cv2.waitKey(0)
