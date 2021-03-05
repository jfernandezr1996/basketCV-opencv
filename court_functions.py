# Librerias
import numpy as np
import cv2


# Detección del campo de baloncesto
def convert_BGR_HSV(color_bgr):
    """
        Transforma un color BGR en sus parámetros HSV.
        
        @param: color_bgr BGR color.
    """
    BGR_color = np.uint8([[color_bgr]])
    HSV_color = cv2.cvtColor(BGR_color, cv2.COLOR_BGR2HSV)
    hue = HSV_color[0][0][0]

    # rangos [low, upper] de color HSV
    lower_color = np.array([hue - 10,10,10])
    upper_color = np.array([hue + 10,255,255])
    return lower_color, upper_color
    

def load_and_court_detection(court, frame, resize, court_color_BGR, 
                             min_line_Hough = None, max_gap_Hough = None, 
                             folder = None, count = None):
    """
        Lectura y procesado de imagen
        Detección del campo de baloncesto usando Canny edge detection y la transformación HoughLines.
        
        @param: court True/False si se ejecuta el proceso de detección del campo
        @param: frame Imagen a procesar.
        @param: resize True/False si es necesario cambiar el tamaño de la imagen.
        @param: court_color_BGR BGR color pista de baloncesto.
        @param: min_line_Hough Longitud mínima de la línea/segmento a considerar.
        @param: max_gap_Hough Máxima distancia entre segmentos para tratarlos como una única línea.
    """
    if resize:
        frame = cv2.resize(frame, (960,600))
        
    # convertir la imagen a HSV
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
    lower_color, upper_color = convert_BGR_HSV(court_color_BGR)
                               
    # threshold and mask
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    res = cv2.bitwise_and(frame, frame, mask = mask)

    # conversión a tonos grises
    ##gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if court: # COURT DETECTION: Canny Edge Detection + Hough Transformation
        edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
        
        #lines = cv2.HoughLinesP(edges, 1, np.pi/180, 150, 100, min_line_Hough, max_gap_Hough)    
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 150, 100, min_line_Hough, max_gap_Hough) #funciona bien para original image
        
        # detección de lineas sobre imagen
        court = frame.copy()
        LINE_COLOR = (0, 255, 0) # verde
        if lines is None:
            pass
        else:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if y1 in range(250, 570):
                    cv2.line(court, (x1,y1), (x2,y2), LINE_COLOR, 5)
    else:
        court = []
                
    return court, frame