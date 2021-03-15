#Librerias
import streamlit as st
import io
import os
import moviepy.editor as moviepy
import time
import cv2
import json

# Dependencias
from functions import *

def delete_folder(folder):
    for file in os.listdir(folder):
        os.remove(folder + "/" + file)
    #os.rmdir(folder)

def create_folder(folder):
    if folder not in os.listdir():
        os.mkdir(folder)
    else:
        delete_folder(folder)
        
def convert_avi_to_mp4(video_name):
    clip = moviepy.VideoFileClip('output/'+video_name+'.avi')
    clip.write_videofile('output/'+video_name+'.mp4')
 
def preprocess_input(file):
    g = io.BytesIO(file.read())
    temp_location = "input/input_video.mp4"
    with open(temp_location, 'wb') as out:
        out.write(g.read())
    #out.close()
    return temp_location

def verify(file):
    if 'input_video.mp4' in os.listdir('input'):
        return 'input/input_video.mp4'
    else:
        loc = preprocess_input(file)
        return loc
    
        
def make_video_streamlit(video_name):
    convert_avi_to_mp4(video_name)
    video_detect = open('output/'+video_name+'.mp4', 'rb')
    video_bytes = video_detect.read()
    return video_bytes

def show_process(fun, message):
    st.sidebar.info(':hourglass_flowing_sand: ' + message)
    m = st.sidebar.empty()
    bar = st.sidebar.progress(0)
    for i in range(100):
        m.text(f'{i+1} %')
        time.sleep(0.1)
        bar.progress(i+1)
    return fun

def algorithm_run(fun, name, video_display, video_name):
    show_process(fun, 'Running ' + name + '...')
    st.sidebar.success(':white_check_mark: The ' + name + ' completed successfully.')
    if video_display:
        video_output = make_video_streamlit(video_name)
        st.video(video_output)

    

# Proceso - ejecución
def main():
    
    st.sidebar.title('basketCV :basketball:')
    
    menu = st.sidebar.selectbox('Menu', ['Home', 'Demo'])
    
    if menu == 'Home':
        
        st.title('Welcome to basketCV app :wave: !!')
        
        st.markdown("""___""")
                
        st.write(':memo: basketCV is an application for the detection, \
                 segmentation, classificacion and tracking of basketball \
                 players from a MP4 video applying Computer Vision techniques.')
                 

        st.image('img/process.PNG', width=700)
        
    else:
        
        with st.sidebar.beta_expander('Input'):
        
            image_file = st.file_uploader(
                "Video", type = ['mp4'])
            params_file = st.file_uploader(
                "Params", type = ['json'])

        if (image_file is None) or (params_file is None):
            st.title('Welcome to basketCV app :wave:')
            st.markdown("""___""")
            
            st.write(':file_folder: Videos of FC Barcelona :red_circle: :large_blue_circle: \
                 Euroleague matches have been used to test the tool.')
            
            st.write(":point_right: The player detection and tracking process \
                     is divided into the following stages:")
            st.write(":one: Input MP4 video")
            st.write(":two: Court Detection")
            st.write(":three: People Detection")
            st.write(":four: Player Team Detection")
            st.write(":five: Player Tracking: Homography, Voronoi")
                
            st.image('img/flujo.PNG', width=700)
            
        else:
            
            # create folders
            create_folder('input')
            create_folder('output')
            
            # preprocess JSON
            params_file.seek(0)
            json_file = json.load(params_file)
            lower_team = tuple(json_file['lower_team'])
            upper_team = tuple(json_file['upper_team'])
            color_team = tuple(json_file['color_team'])
            color_team_polygon = json_file['color_team_polygon']
            match = json_file['match']
            params_file = None
            
            with st.sidebar.beta_expander('Algorithm'):
            
                algorithm = st.selectbox(
                    'Algorithm', ['Select Algorithm',
                                  'Court Detection', 'People Detection', 
                                  'Player Team Detection', 'Player Tracking'])
            
            if algorithm == 'Select Algorithm':
                
                st.title('Video file: ' + match + " :dvd:")
                st.markdown("""___""")
                
                st.header('Display input video :arrow_forward:')
                str_file = verify(image_file)
                video_demo = open('input/input_video.mp4', 'rb')
                video_bytes_demo = video_demo.read()
                
                st.video(video_bytes_demo)
                
            else:
                st.title('Algorithm for Player Detection and Tracking :bar_chart:')
                st.markdown("""___""")

                if algorithm == 'Court Detection':
                    
                    st.header('Court Detection :mag_right:')
                    
                    # video process
                    str_file = verify(image_file)
                    create_folder('court')
                    
                    # run
                    algorithm_run(run_court_detection(str_file),
                                  'court detection algorithm',
                                  True, 'court_detection')
                    
                else:
                    
                    if algorithm == 'People Detection':
                        
                        st.header('People Detection :pushpin:')
                        
                        str_file = verify(image_file)
                        create_folder('confidence')
                        
                        # run
                        algorithm_run(confidence_player_detection(str_file),
                                      'confidence people detection algorithm',
                                      True, 'people_confidence_detection')
                        
                    else:
                        
                        if algorithm == 'Player Team Detection':
                            
                            st.header('Player Team Detection :dart:')
                            
                            str_file = verify(image_file)
                            create_folder('player_team')
                            
                                                    
                            # run
                            algorithm_run(player_detection(str_file, lower_team, 
                                                           upper_team, color_team),
                                          'player team detection algorithm',
                                          True, 'players_detection')
                            
                        else:
                            
                            st.header('Player Tracking :chart_with_upwards_trend:')
                            str_file = verify(image_file)
                            create_folder('homography')
                            create_folder('voronoi')

                            algorithm_run(player_tracking(
                                str_file, 'img/campo.png', list(color_team), 
                                lower_team, upper_team, color_team_polygon, 
                                (255,204,153)), 
                                'player tracking algorithm', False, '')
                            
                            video_homography = make_video_streamlit(
                                'homography_2D')
                            video_voronoi = make_video_streamlit(
                                'voronoi_diagram')
                            
                            col1, col2 = st.beta_columns(2)
                            col1.subheader("2D Projection Homography :bulb:")
                            col1.video(video_homography)
                            col2.subheader('Voronoi Diagram :art:')
                            col2.video(video_voronoi)
                

                            
                 
if __name__ == "__main__":
    main()
