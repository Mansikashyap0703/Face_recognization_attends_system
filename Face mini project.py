import cv2
import numpy as np
import face_recognition
import os  
from datetime import datetime  

path = 'C:/Users/Mansi/OneDrive/Desktop/project/image'
images = []
personNames = []
myList = os.listdir(path)  
print(myList)
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])
print(personNames)

def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
#print(faceEncodings(images))
def attendance(name):
    with open('C:/Users/Mansi/OneDrive/Desktop/project/Attend.csv', 'r+') as f:
    # r+ ka matlab read and append dono kr sakte hain.
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')


encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25) 
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (79, 255, 160), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (160, 255, 40), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (29, 255, 255), 2)
            attendance(name)
        else:
            print("Name is not matched")

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13:  #ascii code of enter key
        break
'''img1 = face_recognition.load_image_file('C:/Users/Mansi/OneDrive/Desktop/project/image/kamal sir.jfif')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1Test = face_recognition.load_image_file('C:/Users/Mansi/OneDrive/Desktop/project/image/mi.jfif')
img1Test = cv2.cvtColor(img1Test, cv2.COLOR_BGR2RGB)

face = face_recognition.face_locations(img1)[0]
print(face)
encodeFace = face_recognition.face_encodings(img1)[0]
print(encodeFace)
cv2.rectangle(img1, (face[3], face[0]), (face[1], face[2]), (255, 0, 255), 2)

faceTest = face_recognition.face_locations(img1Test)[0]
encodeTestFace = face_recognition.face_encodings(img1Test)[0]
cv2.rectangle(img1Test, (faceTest[3], faceTest[0]), (faceTest[1], faceTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeFace], encodeTestFace)
faceDis = face_recognition.face_distance([encodeFace], encodeTestFace)
print(results, faceDis)
cv2.putText(img1Test, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('1 image mansi', img1)
cv2.imshow('2 image kamal sir', img1Test)'''
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

