import cv2
import argparse
import numpy as np


"""parser = argparse.ArgumentParser()
parser.add_argument("path", help="path")
args = parser.parse_args()"""

img = cv2.imread("Images/download.jpeg")
const = 450
height, width, channels = img.shape
blurred_img = cv2.GaussianBlur(img, (21, 21), 30)
dsize = (const, round(height*(const/width)))
mask = np.zeros((height, width, channels), dtype=np.uint8)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces_detect = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_detect:
  """cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)"""
  mask = cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)

out = np.where(mask!=(255, 255, 255), img, blurred_img)
out2 = cv2.resize(out, dsize)
cv2.imshow('image', out2)
cv2.waitKey()

