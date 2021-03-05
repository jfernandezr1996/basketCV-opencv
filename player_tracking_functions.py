# Librerias
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils


# Players Tracking: proyeccion 2D, heatmap and Voronoi Diagram
def draw_voronoi(img, subdiv, color_points, color_polygon):
    """
        Draw Voronoi Polygon
    """
    (facets, centers) = subdiv.getVoronoiFacetList([])
    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)
        ifacet = np.array(ifacet_arr, np.int)
        cv2.fillConvexPoly(img, ifacet, color_polygon[i], cv2.LINE_AA, 0)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0,0,0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 5, color_points[i], -1)


def players_tracking(
        im_original, ruta_img_court, team_color, barcelona_color, 
        team_color_polygon, barcelona_color_polygon):
    """
    Proyección de los jugadores sobre imagen de cancha de baloncesto.

    Parameters
    ----------
    im_original : cv2 image
        Frame resultante de la función player_detection_HOG()..
    ruta_img_court : string
        Directorio de la imagen de la cancha de baloncesto donde proyectar la homografía..
    team_color : tupple
        BGR color del equipo adversario al FC Barcelona.
    barcelona_color : tupple
        BGR color FC Barcelona.
    team_color_polygon : tupple
        If True voronoi, color of rival Barcelona polygon color.
    barcelona_color_polygon: tupple
        If True voronoi, color Barcleona polygon.

    Returns
    -------
    homography_pos : image
        Proyeccion de las coordenadas 2D en imagen.
    list_coordinates : list
        Coordenadas de los puntos detectados en cada frame + color
    img_voronoi : image
        Diagrama de Voronoi a partir de la proyeccion 2D.
    """
    
    plt.imshow(im_original)
    left_x, right_x = plt.xlim()
    left_y, right_y = plt.ylim()
    pts_src = np.array([[left_x-20,left_y-50], 
                        [900+20,500-50], 
                        [580+20,250-20], 
                        [left_x-20,300-20]])
    
    im_dst = cv2.imread(ruta_img_court)
    plt.imshow(im_dst)
    left_x, right_x = plt.xlim()
    left_y, right_y = plt.ylim()
    pts_dst = np.array([[left_x, left_y], 
                        [right_x, left_y], 
                        [right_x, right_y], 
                        [left_x, right_y]])
    
    # homography
    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(im_original, h, (im_dst.shape[1], im_dst.shape[0]))
    
    # rangos BGR blue (Barcelona)
    lower_blue = np.array(list(barcelona_color))                         
    upper_blue = np.array([255,155,155]) 
    color_blue = list(barcelona_color)
    # rangos GBR red (Alba Berlin) -- 
    ## es el color usado para los círculos en player_detection_HOG(tracking=True)
    lower_color = np.array([0,0,255])
    upper_color = np.array([155,155,255])
    color = team_color
    
    boundaries = [(lower_blue, upper_blue, color_blue),
                  (lower_color, upper_color, color)]
    
    # Radius of circle 
    radius = 5

    # Line thickness of 2 px 
    thickness = -1

    court_img = cv2.imread(ruta_img_court)
    
    homography_pos = court_img.copy()
    
    list_coordinates = []
    
    for r in boundaries:
        mask = cv2.inRange(im_out, r[0], r[1])
        result = cv2.bitwise_and(im_out, im_out, mask = mask)
        mask = cv2.inRange(result, r[0], r[1])
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if cnts is not None:
            for cnt in cnts:
                for pos in cnt[0]:
                    center_coordinates = (pos[0], pos[1])
                    if r[2] == color_blue:
                        list_coordinates.append([center_coordinates, 'barcelona'])
                    else:
                        list_coordinates.append([center_coordinates, 'other'])
                    cv2.circle(homography_pos, center_coordinates, radius, 
                               tuple(r[2]), thickness) 
                    
    voronoi_input = court_img.copy()
    size = voronoi_input.shape

    # Rectangle to be used with Subdiv2D
    rect = (0, 0, size[1], size[0])

    # Points
    points = [i[0] for i in list_coordinates]
    list_colors_points = [i[1] for i in list_coordinates]
    color_points = [barcelona_color if x == 'barcelona' else 
                    tuple(team_color) for x in list_colors_points]
    color_polygon = [barcelona_color_polygon if x == 'barcelona' else 
                     tuple(team_color_polygon) for x in list_colors_points]


    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)
    
    # Draw points
    for i in range(len(points)):
        cv2.circle(voronoi_input, points[i], radius, color_points[i], thickness)
    
    # Allocate space for Voronoi Diagram
    img_voronoi = np.zeros(voronoi_input.shape, dtype = voronoi_input.dtype)

    # Draw Voronoi diagram
    draw_voronoi(img_voronoi, subdiv, color_points, color_polygon)

    return homography_pos, list_coordinates, img_voronoi