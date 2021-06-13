import cv2
from recognition import *
from detect import *
import numpy as np

cap = cv2.VideoCapture('video')

while (cap.isOpened()):
    plateRet, plateFrame = cap.read()
    
    plate_oringal, edge = E2E.format()

    _,plate_cnts,_ = cv2.findContours(plateFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for plate_cnt in plate_cnts:
        if E2E.predict():
            rect = cv2.minAreaRect(plate_cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(plate_oringal, [box], 0, (0, 255, 0), 2)
            Xs, Ys = [i[0] for i in box], [i[1] for i in box]
            x1, y1 = min(Xs), min(Ys)
            x2, y2 = max(Xs), max(Ys)

            angle = rect[2]
            if angle < -45: angle += 90

            W, H = rect[1][0], rect[1][1]
            aspect_ratio = float(W)/H if W > H else float(H)/W

            center = ((x1+x2)/2, (y1+y2)/2)
            size = (x2-x1, y2-y1)
            M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
            tmp = cv2.getRectSubPix(edge, size, center)
            TmpW = H if H > W else W
            TmpH = H if H < W else W
            tmp = cv2.getRectSubPix(tmp, (int(TmpW),int(TmpH)), (size[0]/2, size[1]/2))
            __,tmp = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


            # white_pixels = 0
            # for x in range(tmp.shape[0]):
            #     for y in range(tmp.shape[1]):
            #         if tmp[x][y] == 255:
            #             white_pixels += 1

            





    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()