import cv2
import os
import numpy as np
import warnings
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", category=FutureWarning)
#Function that returns the co-ordinates of faces if detected
def Face_Recognition(img):
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    classifier = cv2.CascadeClassifier('Face Mask Recognition/haarcascade_frontalface_alt.xml')
    try:
        faces = classifier.detectMultiScale(gray_image)
        return faces #Face found
    except:
        pass

#Create the data with_mask and without_mask
cat = ["WithMask","WithoutMask"] 
directory = r"Face Mask Recognition" #Parent directory of folders
data = [] 
for category in cat:
    inside = []
    path = os.path.join(directory,category)
    print(path)
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        image = cv2.imread(img_path)
        faces = Face_Recognition(image) 
        for (x,y,w,h) in faces:
            image = image[y:y+h+1][x:x+w+1][:]
        try:
            image = cv2.resize(image,(50,50))
        except:
            continue
        if len(inside) == 250:
            break
        inside.append(image)
    data.append(inside)
data = np.array(data)
mask = np.array(data[0])
nomask = np.array(data[1])
mask = mask.reshape((250,50*50*3))
nomask = nomask.reshape((250,50*50*3))
X = np.concatenate((mask,nomask))
labels = np.zeros(X.shape[0])
labels[250:] = 1
mask_nomask = {"Mask":0,"No Mask":1}
print(X.shape,labels.shape)
#Creating a Support Vector Model
svm = SVC(kernel='linear', C = 1.0)
x_train,x_test,y_train,y_test = train_test_split(X,labels,test_size = 0.3)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
svm.fit(x_train,y_train)
y_pred = svm.predict(x_test)
print(accuracy_score(y_pred,y_test))

img = cv2.imread("Images\download (1).jpg")
faces = Face_Recognition(img)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0))
cv2.imshow("Face",img)
cv2.waitKey()
cv2.destroyAllWindows()
img = img[y:y+h+1,x:x+w+1,:]
img = cv2.resize(img,(50,50))
print(img.shape)
img = img.reshape((1,-1))
print(img.shape)
y_pred = svm.predict(img)
print(y_pred)