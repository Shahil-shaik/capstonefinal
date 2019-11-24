import cv2
import os,sys

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = sys.argv[1]

cam = cv2.VideoCapture(sys.argv[2])

count = 0

while(cam.isOpened()) :
    ret, img = cam.read()
    if ret == True:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1

            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('image', img)

        k = cv2.waitKey(10) & 0xff 
        if k == 27:
            break
    else:
        cam.release()
        cv2.destroyAllWindows() 