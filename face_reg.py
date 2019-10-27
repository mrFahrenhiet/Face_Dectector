import cv2
import numpy as np
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
data_path = './data/'

file_name = input("Enter Name:")
while True:
 	ret,frame = cap.read()
 	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 	if ret==False:
 		continue
 	
 	faces = face_cascade.detectMultiScale(frame,1.3,5)
 	faces = sorted(faces,key=lambda f: f[2]*f[3])
 	for face in faces[-1:]:
 		x,y,w,h = face
 		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
 		offset = 10
 		face_sec = frame[y-offset:y+h+offset,x-offset:x+w+offset]
 		face_sec = cv2.resize(face_sec,(100,100))
	 	skip+=1
	 	if skip%10==0:
	 		face_data.append(face_sec)
	 		print(len(face_data))
 		cv2.imshow("Video Frame",frame)
 		cv2.imshow("stored image",face_sec)

 	
 	key = cv2.waitKey(1) & 0xFF
 	if key == ord('q'):
 		break
#covert to numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#saving
np.save(data_path+file_name+".npy",face_data)
print("Saved")

cap.release()
cv2.destroyAllWindows()
