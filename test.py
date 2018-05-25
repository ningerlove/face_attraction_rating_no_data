from sklearn.externals import joblib
import os,dlib
from skimage import io
import numpy as np

modle_path_file = "H:/PYTHONcode/faceattractineness/modle_file"
test_path_file =  "H:/PYTHONcode/faceattractineness/ownpictures"
predictor_path = './shape_predictor_68_face_landmarks.dat'
# 2.人脸识别模型
face_rec_model_path =  './dlib_face_recognition_resnet_model_v1.dat'



modle_file = os.listdir(modle_path_file)
test_file = os.listdir(test_path_file)

n_test = len(test_file)

test_data = np.zeros((n_test,128))

detector = dlib.get_frontal_face_detector()
# 2.加载人脸关键点检测器
sp = dlib.shape_predictor(predictor_path)
# 3. 加载人脸识别模型
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

for modle in modle_file:
    reg = joblib.load(os.path.join(modle_path_file,modle))
    for picture in range(len(test_file)):
        img = io.imread(os.path.join(test_path_file,test_file[picture]))
        dets = detector(img, 1)            
        dist = {}
        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            d_test = np.array(face_descriptor) 
        test_data[picture,:] = d_test


score = reg.predict(test_data )
print("picture的得分：{}".format(score))