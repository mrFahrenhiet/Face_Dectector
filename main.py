import cv2
import numpy as np
import os
#KNN Algo
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,query_pt,k=5):
    vals = []
    m = X.shape[0]
    for i in range(m):
        d = dist(query_pt,X[i])
        vals.append((d,Y[i]))
    
    vals = sorted(vals)
    vals  = vals[:k]
    vals = np.array(vals)
    print(vals)
    new_vals = np.unique(vals[:,1],return_counts=True)
    print(new_vals)
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    return pred

#Video Cam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
labels = []
class_id = 0
name = {}
data_path = './data/'

#data perps
for fx in os.listdir(data_path):
	if fx.endswith('.npy'):
		name[class_id] = fx[:-4]
		print("Loaded file " + fx)
		data_item = np.load(data_path+fx)
		face_data.append(data_item)
		# Y data
		tar = class_id*np.ones((data_item.shape[0],1))
		class_id+=1
		labels.append(tar)
face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0)
print(face_dataset.shape)
print(face_labels.shape)


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
        pred = knn(face_dataset,face_labels,face_sec.flatten())
        pred_name = name[int(pred)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,150),2,cv2.LINE_AA)
        cv2.imshow("Video Frame",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
