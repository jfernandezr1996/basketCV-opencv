# Librerias
import re
import os
import cv2


# Funciones
def atoi(text):
    # A helper function to return digits inside text
    return int(text) if text.isdigit() else text


def natural_keys(text):
    # A helper function to generate keys for sorting frames AKA natural sorting
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def make_video(image_folder, video_name, speed):
    images = [img for img in os.listdir(image_folder)]
    images.sort(key=natural_keys)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    video = cv2.VideoWriter(video_name, fourcc, speed, (width, height))
    
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    
    for file in os.listdir(image_folder):
        os.remove(image_folder + file)
    os.rmdir(image_folder)