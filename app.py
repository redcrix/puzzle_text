import cv2
import numpy as np
import csv

# reading the image
image = cv2.imread("puzzlev4.jpg")

edged = cv2.Canny(image, 10, 250)
#cv2.imshow("Edges", edged)
#cv2.waitKey(0)

# applying closing function
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
#cv2.imshow("Closed", closed)
#cv2.waitKey(0)

# finding_contours
(cnts, hier) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(cnts)
idx = 0
print("Number of Contours found = " + str(len(cnts)))

# Mapping ClassID to traffic sign names
signs = []
with open('images/icon.txt', 'r') as csvfile:
    signnames = csv.reader(csvfile, delimiter=',')
    next(signnames, None)
    for row in signnames:
        signs.append(row[1])
    csvfile.close()

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    print(x,y,w,h)
    if w>50 and h>50:
        idx+=1
        new_img = image[y:y+h,x:x+w]
        #print(new_img,idx)

        ae = cv2.imwrite('images/'+ str(idx) + '.png', new_img)
        img = cv2.imread('images/'+ str(idx) + '.png')

        #print(img.shape)
        threshold = 0.7
        res = cv2.matchTemplate(image, img, cv2.TM_CCOEFF_NORMED)

        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(res)
        #print(signs)




