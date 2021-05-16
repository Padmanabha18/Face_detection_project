import cv2 as cv

#Reading image
img = cv.imread('/Users/padmanab/Desktop/Pycharm/Face_detection_project/group 1.jpg.jpg')
#cv.imshow('Padmanab', img)

#Covertig to gray image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

#Haarcascading
haar_cascade = cv.CascadeClassifier('haar_face.xml')

#Comparing image with Haar cascade
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#Drawing rectangle over the detected face
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
cv.imshow('Detected faces', img)
print(f'Number of faces found:= {len(faces_rect)}')
cv.waitKey(0)