import face_recognition
import cv2
import numpy as np
import os
import glob
import tkinter as tk
import pandas as pd
import pandastable
from pandastable import Table, TableModel
from flask import Flask, render_template, Response
import pandas as pd

data = pd.read_excel("g.xlsx")

app=Flask(__name__)
last_person = ""

def gen_frames():
    global last_person
    # opening camera and face recognition
    faces_encodings = []
    faces_names = []
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'faces', '')
    list_of_files = [f for f in glob.glob(path+'*.jpg')]
    number_files = len(list_of_files)
    names = list_of_files.copy()
    last_person = ""

    for i in range(number_files):
        globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
        globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
        faces_encodings.append(globals()['image_encoding_{}'.format(i)])
        # Create array of known names
        names[i] = names[i].replace(path, "")
        names[i] = names[i].replace(".jpg", "")
        faces_names.append(names[i])

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces (faces_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance( faces_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = faces_names[best_match_index]
                    last_person = name
                    # data = last_person
                face_names.append(name)
        process_this_frame = not process_this_frame
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
        # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (160, 160, 255), 2)
        # Input text label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (160, 160, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (83, 42, 38), 2)
            # cv2.putText(frame, "press 's' to show student's information or 'q' to exit", (5, 30), font, 0.65, (83,42,38), 2)
            
        # Display the resulting image
        # cv2.imshow('Face Recognition', frame)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        

        # b = cv2.waitKey(1)
        #Hit 's' to show information in table
        # if b & 0xFF == ord('s'):
        # print(data.loc[data['name'] == last_person])
        # information = data.loc[data['name'] == last_person]
        # col = data.columns
        # information = information.transpose()
        # information.insert(0, "0", col, allow_duplicates=True)
        # information.columns = ['1', '2']
        # root = tk.Tk()
        # root.title(last_person)
        # frame = tk.Frame(root)
        # frame.pack(fill='both', expand=True)
        # pt = Table(frame, showtoolbar=True, showstatusbar=True)
    
        # pt.show()
        # pt.model.df = information
        # pt.autoResizeColumns()
        # pt.columncolors['1'] = '#986D8E' #color a specific column
        # pt.columncolors['2'] = '#FFA0A0' #color a specific column
        # pt.redraw()
        # root.mainloop()

        # Hit 'q' on the keyboard to quit!
        # if b & 0xFF == ord('q'):
            # break





@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/background_process_test')
def background_process_test():
    # print ("Hello")
    information = data.loc[data['name'] == last_person]
    # information = information.to_html()
    information = information.to_json(orient='records')
    # print(type(data.loc[data['name'] == last_person]))
    # print("type ", information)
    return information


if __name__=='__main__':
    app.run(debug=True)