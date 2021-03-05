# Librerias
import cv2

# Dependencias
from court_functions import load_and_court_detection
from video_functions import make_video
from player_detection_functions import player_detection_HOG
from player_tracking_functions import players_tracking

# Funciones de cada algoritmia desarrollada
def run_court_detection(input_video):
    """
    Ejecución del algoritmo de detección del campo

    Parameters
    ----------
    input_video : string
        Ruta del video de entrada.

    Returns
    -------
    Detección del campo frame-by-frame y creación del video resultante
    """
    
    cap = cv2.VideoCapture(input_video)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            court, _ = load_and_court_detection(court = True, frame = frame, resize = True, 
                                                court_color_BGR = [153,204,255], 
                                                min_line_Hough = 10, max_gap_Hough = None, 
                                                count = count)
            name = "court/frame%d.jpg" % count
            cv2.imwrite(name, court)
            
            count += 1
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break
            
    make_video('court/', 'output/court_detection.avi', 30.0) # output video
    cap.release()
    cv2.destroyAllWindows()  
    
    
    
def confidence_player_detection(input_video):
    """
    Algoritmo de detección de personas en cada frame

    Parameters
    ----------
    input_video : string
        Ruta del video de entrada.

    Returns
    -------
    Detección indicando el grado de confianza de personas en el frame y 
    generación del video resultante de proceso.

    """
    cap = cv2.VideoCapture(input_video)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            _, frame = load_and_court_detection(court = False, frame = frame, resize = True, 
                                                court_color_BGR = [153,204,255])
            players_confidence = player_detection_HOG(
                frame = frame, speed = 'fast', confidence = True, tracking = False)
            
            name = "confidence/frame%d.jpg" % count
            cv2.imwrite(name, players_confidence)
            
            count += 1
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break
            
    make_video('confidence/', 'output/people_confidence_detection.avi', 30.0) # output video
    cap.release()
    cv2.destroyAllWindows() 
    
    
    
def player_detection(input_video, lower_team, upper_team, color_team, 
                     lower_barcelona = (100,150,20), 
                     upper_barcelona = (130,255,255),
                     color_barcelona = (255,0,0)):
    """
    Algoritmo de detección de jugadores de baloncesto en la cancha.

    Parameters
    ----------
    input_video : string
        Ruta del video de entrada.
    lower_team : tupple
        Array BGR (límite inferior) equipo adversario al FC Barcelona.
    upper_team : tupple
        Array BGR (limite superior) equipo adversario al FC Barcelona.
    color_team : tupple
        Array BGR equipo adversario al FC Barcelona.
    lower_barcelona : tupple
        Array BGR (límite inferior) FC Barcelona.
    upper_barcelona : tupple
        Array BGR (límite superior) FC Barcelona.
    color_barcelona : tupple
        Array BGR FC Barcelona

    Returns
    -------
    Detección y clasificación de jugadores de baloncesto.

    """
    cap = cv2.VideoCapture(input_video)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            _, frame = load_and_court_detection(
                court = False, frame = frame, resize = True, 
                court_color_BGR = [153,204,255])
            players_detection = player_detection_HOG(
                frame = frame, speed = 'fast', confidence = False, tracking = False, 
                lower_barcelona = lower_barcelona, 
                upper_barcelona = upper_barcelona, 
                color_barcelona = color_barcelona, lower_vs = lower_team, 
                upper_vs = upper_team, color_vs = color_team, count = count)
            
            name = "player_team/frame%d.jpg" % count
            cv2.imwrite(name, players_detection)
            
            count += 1
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break
            
    make_video(
        'player_team/', 'output/players_detection.avi', 30.0) # output video
    cap.release()
    cv2.destroyAllWindows()    
    
    
    
def player_tracking(input_video, input_pitch, color_team, 
                    lower_team, upper_team, color_team_polygon, 
                    color_barcelona_polygon,
                    color_barcelona = (255,0,0), 
                    lower_barcelona = (100,150,20), 
                    upper_barcelona = (130,255,255)):
    """
    Algoritmo de rastreo de jugadores y proyección coordenadas 2D

    Parameters
    ----------
    input_video : string
        Ruta del video de entrada.
    input_pitch : string
        Ruta imagen cancha de baloncesto donde proyectar 2D.
    lower_team : tupple
        Array BGR (límite inferior) equipo adversario al FC Barcelona.
    upper_team : tupple
        Array BGR (limite superior) equipo adversario al FC Barcelona.
    color_team : tupple
        Array BGR equipo adversario al FC Barcelona.
    lower_barcelona : tupple
        Array BGR (límite inferior) FC Barcelona.
    upper_barcelona : tupple
        Array BGR (límite superior) FC Barcelona.
    color_barcelona : tupple
        Array BGR FC Barcelona
    color_team_polygon : tupple
        Array BGR color polygon voronoi adversario FC Barcelona.

    Returns
    -------
    None.

    """
    cap = cv2.VideoCapture(input_video)
    count = 0
    coordinates_circles = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            _, frame = load_and_court_detection(
                court = False, frame = frame, resize = True, court_color_BGR = [153,204,255])
            players_detection = player_detection_HOG(
                frame = frame, speed = 'fast', confidence = False, tracking = True, 
                lower_barcelona = lower_barcelona, upper_barcelona = upper_barcelona, 
                color_barcelona = color_barcelona, lower_vs = lower_team, 
                upper_vs = upper_team)
            
            players_homography, all_coordinates, voronoi_output = players_tracking(
                players_detection, input_pitch, color_team, color_barcelona, 
                team_color_polygon=color_team_polygon, 
                barcelona_color_polygon=color_barcelona_polygon)
            coordinates_circles.extend(all_coordinates)
            
            name_voronoi = "voronoi/framevoronoi%d.jpg" % count
            cv2.imwrite(name_voronoi, voronoi_output)
            
            name_homography = "homography/framehomography%d.jpg" % count
            cv2.imwrite(name_homography, players_homography)
            
            count += 1
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break
    
    # video output      
    make_video('voronoi/', 'output/voronoi_diagram.avi', 5.0) # output video
    make_video('homography/', 'output/homography_2D.avi', 5.0) # output video

    cap.release()
    cv2.destroyAllWindows() 
    
    
    
