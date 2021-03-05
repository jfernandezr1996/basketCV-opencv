# Librerias
import cv2
import numpy as np


# Player Detection
def player_detection_HOG(frame, speed, confidence, tracking,
                         lower_barcelona = None, upper_barcelona = None, 
                         color_barcelona = None, lower_vs = None, upper_vs = None, 
                         color_vs = None, count = None, folder = None):
    """
        Detección de los jugadores en la pista de baloncesto usando HOG (Histograms of Oriented Gradients).
        
        @param: frame Imagen a procesar
        @param: speed Tipo de velocidad de la imagen ['fast', 'slow']. La configuración del HOG será distinta.
        @param: confidence True/False. Si True, detección de futbolistas y %. Si False, detección de jugadores + equipo.
        @param: tracking True/False. Si True, utilizaremos la identificación de jugador+equipo para hacer tracking futuro.
    """
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # HOG (Histograms of Oriented Gradients)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
    ## params HOG
    if speed == 'fast':
        rects, weights = hog.detectMultiScale(img_gray, padding=(4, 4), scale=1.02)
    elif speed == 'slow':
        rects, weights = hog.detectMultiScale(img_gray, winStride=(4, 4), padding=(4, 4), scale=1.02)
        
    if confidence:
        for i, (x, y, w, h) in enumerate(rects):
            if (h<=300) and (w<=140) and (y <= 570): # human dimensions
                start_point = (int(x+1/4*w), int(y+1/4*h))
                end_point = (int(x+3/4*w), int(y+3/4*h))
                if weights[i] < 0.13:
                    continue
                elif weights[i] < 0.3 and weights[i] > 0.13:
                    cv2.rectangle(frame, start_point, end_point, (0, 0, 255), 2)
                if weights[i] < 0.7 and weights[i] > 0.3:
                    cv2.rectangle(frame, start_point, end_point, (50, 122, 255), 2)
                if weights[i] > 0.7:
                    cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
            
            cv2.putText(frame, 'High confidence', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, 'Moderate confidence', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 122, 255), 2)
            cv2.putText(frame, 'Low confidence', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    else:
        
        for i, (x,y,w,h) in enumerate(rects):
            if (h<=300) and (w<=140) and (y > 170):
                # params
                start_point_rect = (int(x+1/4*w), int(y+1/4*h))
                end_point_rect = (int(x+3/4*w), int(y+3/4*h))
                point_circle = (int(x+1/2*w), int(y+1/2*h))
                radius = 5
                thickness = -1
                
                player_img = frame[int(y+1/4*h):int(y+3/4*h),int(x+1/4*w):int(x+3/4*w)]
                player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
                
                # team 1 - barcelona
                mask1 = cv2.inRange(player_hsv, lower_barcelona, upper_barcelona)
                res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                
                res1_hsv = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
                res1_hsv = cv2.cvtColor(res1_hsv, cv2.COLOR_BGR2GRAY)
                nzCountE1 = cv2.countNonZero(res1_hsv)
                
                # team 2 - away (vs)
                mask2 = cv2.inRange(player_hsv, lower_vs, upper_vs)
                res2 = cv2.bitwise_and(player_img, player_img, mask = mask2)
                
                res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
                res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
                nzCountE2 = cv2.countNonZero(res2)
                
                if nzCountE1 > nzCountE2:
                    if (nzCountE1 > 50) and (nzCountE1 < 1000):
                        if tracking:
                            cv2.circle(frame, point_circle, radius, color_barcelona, thickness)
                        else:
                            cv2.rectangle(frame, start_point_rect, end_point_rect, color_barcelona, 2)
                    else:
                        pass
                else:
                    if (nzCountE2 > 50) and (nzCountE2 < 1000):
                        if tracking:
                            cv2.circle(frame, point_circle, radius, (0,0,255), thickness)
                        else:
                            cv2.rectangle(frame, start_point_rect, end_point_rect, color_vs, 2)
                    else:
                        pass

    return frame