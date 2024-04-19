import cv2
import imutils
# translation, rotation, resize, detect edges --> functions
from imutils import face_utils
#to get landmark of the left and the right eye
import dlib
from scipy.spatial import distance
from pygame import mixer

mixer.init()
mixer.music.load("music.wav")

#e.a.r = sum of vertical / 2*horizontal distance, drops suddenly when eyes close
#calculate for both eyes and compare to threshold
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1],eye[5])
    B = distance.euclidean(eye[2],eye[4])
    C = distance.euclidean(eye[0],eye[3])
    ear = (A + B)/(2.0 * C)
    return ear

thresh = 0.25
#variable for different systems depending on camera
flag = 0
#frame count
frame_check = 10
#for creating delay in warning
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
#landmark detect for each eye
detect = dlib.get_frontal_face_detector()
#doesn't accept any parameter
predict=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#detects landmarks from face --> snapchat filters use landmarks as a point of reference
#6 points in eyes --> p0 to p3 horizontal and p1 p5 and p2 p4 short points
cap=cv2.VideoCapture (0)
#this will returns the frames detected from camera. 0 means using primary camera of the system
while True:
    ret,frame=cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #converts face to gray scale
    subjects = detect(gray,0)
    for subject in subjects:
        shape = predict(gray,subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEar = eye_aspect_ratio(leftEye)
        rightEar = eye_aspect_ratio(rightEye)
        ear = (leftEar+rightEar)/ (2.0)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame,[leftEyeHull],-1, (0,255,0), 1)
        cv2.drawContours(frame,[rightEyeHull],-1, (0,255,0), 1)
        if ear < thresh :
            flag += 1
            print(flag)
            if flag>= frame_check:
                cv2.putText(frame, "ALERT",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0), 2)
                cv2.putText(frame, "ALERT",(10,325), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0), 2)
                mixer.music.play()
        else:
            flag=0
    cv2.imshow("Frame",frame)
    #used to display image on a window
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()
