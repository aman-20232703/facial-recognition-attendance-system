# PIP install cmake
# pip install face_recognition
# pip install opencv python
# pip install numpy

import face_recognition
import cv2
import csv
import numpy as np
from datetime import datetime

video_capture=cv2.VideoCapture(0)

aman_image=face_recognition.load_image_file("faces/aman.jpg")
aman_encoding=face_recognition.face_encodings(aman_image)[0]


vicky_image=face_recognition.load_image_file("faces/vicky.jpg")
vicky_encoding=face_recognition.face_encodings(vicky_image)[0]

known_face_encoding= [aman_encoding, vicky_encoding]
known_face_names=["aman","vicky"]

students=known_face_names.copy()

face_locations=[]
face_encodings=[]

now=datetime.now()
current_date=now.strftime("%y-%m-%d")

f=open(f"{current_date}.csv", "w+", newline="")
lnwriter=csv.writer(f)

while True:
    _, frame=video_capture.read()
    small_frame= cv2.resize(frame,(0,0), fx=0.25, fy=0.25)
    rgb_amall_frame=cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    face_locations=face_recognition.face_locations(rgb_amall_frame)
    face_encodings=face_recognition.face_encoding(rgb_amall_frame,face_locations)
    
    for face_encoding in face_encodings:
        matches=face_recognition.compare_faces(known_face_encoding, face_encoding)
        face_distance=face_recognition.face_distance(known_face_encoding,face_encoding)
        
        best_match_index=np.argmin(face_distance)
        
        if (matches[best_match_index]):
            name=known_face_encoding[best_match_index]
            
        if name in known_face_names:
            font=cv2.FONT_HERSHEY_COMPLEX
            bottomLeftCorner=(10,100)
            fontScale=1.5
            fontColor=(255,0,0)
            thickness=3
            lineType=2
            bottomLeftCorner=(10,100)
            cv2.putText(frame,name+"present",bottomLeftCorner,fontScale,fontColor,thickness,lineType)
        
        if name in students:
            students.remove(name)
            current_time=now.strftime("%H-%M-%S")
            lnwriter.writerow([ name, current_time])
        
    cv2.imshow("attendance",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
video_capture.release()
cv2.destroyAllWindows()
f.close()





#  Required installations (run in terminal before executing this script)
# pip install cmake
# pip install face_recognition
# pip install opencv-python
# pip install numpy

import face_recognition  # For facial recognition
import cv2               # For capturing webcam video
import csv               # For writing attendance into CSV
import numpy as np       # For numerical operations
from datetime import datetime  # For timestamping attendance

# Start capturing video from the default webcam
video_capture = cv2.VideoCapture(0)

# Load reference images of known students and generate face encodings
aman_image = face_recognition.load_image_file("faces/aman.jpg")
aman_encoding = face_recognition.face_encodings(aman_image)[0]

vicky_image = face_recognition.load_image_file("faces/vicky.jpg")
vicky_encoding = face_recognition.face_encodings(vicky_image)[0]

# List of known encodings and corresponding names
known_face_encoding = [aman_encoding, vicky_encoding]
known_face_names = ["aman", "vicky"]

# Create a copy for tracking students who haven't yet been marked present
students = known_face_names.copy()

# Initialize frame data
face_locations = []
face_encodings = []

# Create filename with current date
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Create and open a CSV file for recording attendance
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

# Start face recognition and attendance marking loop
while True:
    _, frame = video_capture.read()

    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    #  Convert image from BGR (OpenCV default) to RGB (face_recognition expects this)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect all face locations and their encodings in current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Loop through each detected face
    for face_encoding in face_encodings:
        # Compare detected face with known faces
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
        
        best_match_index = np.argmin(face_distance)  # Find closest match

        name = ""
        if matches[best_match_index]:
            name = known_face_names[best_match_index]  # 🧾 Get matching name

            # Display name and "present" status on frame
            cv2.putText(frame, f"{name} - Present", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 3)

            # If student is still in the list, mark attendance
            if name in students:
                students.remove(name)  # Remove from list so they aren't marked twice
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])  # Write to CSV

    # Show the live camera feed
    cv2.imshow("🎓 Attendance System", frame)

    # Quit the app when 'q' is pressed
