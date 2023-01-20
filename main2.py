import pandas as pd
import streamlit as st
import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime
import os
import time
import base64

st.set_page_config(page_title="Attendance System", page_icon=":guardsman:", layout="wide")

# Create a title and a subheader
st.title("Welcome to Attendance System")
st.subheader("Powered by Face Recognition")

# Load known images and create encodings
known_face_encodings = []
known_faces_names = []
for file in os.listdir("faces"):
    if file.endswith(".jpg"):
        name = file.split(".")[0]
        image = face_recognition.load_image_file("faces/" + file)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_faces_names.append(name)

# initialize list to store students and timestamp
students = known_faces_names.copy()
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# adjust tolerance level
tolerance = 0.7

# Create a button to start the attendance process
if st.button("Start Attendance"):
    # Open a video capture
    videocap = cv2.VideoCapture(0)
    with open(current_date+'.csv', 'w+', newline='') as f:
        lnwriter = csv.writer(f)
        i = 60
        while True:
            _, frame = videocap.read()
            if frame is None:
                pass
                break
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            frame_height, frame_width = frame.shape[:2]
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance)
                name = "Unknown"
                if True in matches:
                    best_match_index = np.argmax(matches)
                    name = known_faces_names[best_match_index]
                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])
                    f.flush()

                    frame = cv2.resize(frame, (int(frame_width / 2), int(frame_height / 2)))
                    # Display the image
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame, use_column_width=True)
                    time.sleep(20)
            i -= 1

            if i <=0 or st.button("Stop Attendance"):
                    break


            videocap.release()
            st.success("Attendance process complete.")
            st.write(name + " has been marked as present")
            st.info("Check the " + current_date + ".csv file for attendance records.")

            # if st.button("Download Attendance"):
            #     st.set_page_config(page_title=current_date + " Attendance", layout="wide")
            #     st.write("Download the attendance file: ")
            #     csv = open(current_date + '.csv', 'r').read()
            #     b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            #     href = f'<a href="data:file/csv;base64,{b64}" download="{current_date}.csv">Download CSV File</a>'
            #     st.markdown(href, unsafe_allow_html=True)
            # attendance_data = pd.read_csv(current_date + '.csv')
            # st.dataframe(attendance_data)  # display the data in a table
            # st.info("Check the " + current_date + ".csv file for attendance records.")
            # st.write(frame)


