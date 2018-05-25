from sklearn.externals import joblib
import os,dlib
from skimage import io
import numpy as np

modle_path_file = "H:/PYTHONcode/faceattractineness/modle_file"
test_path_file =  "H:/PYTHONcode/faceattractineness/ownpictures"
predictor_path = './shape_predictor_68_face_landmarks.dat'
# 2.人脸识别模型
face_rec_model_path =  './dlib_face_recognition_resnet_model_v1.dat'






def caculate_score(input_path):
    
    modle_file = os.listdir(modle_path_file)
    
    test_data = np.zeros((1,128))
    detector = dlib.get_frontal_face_detector()
    # 2.加载人脸关键点检测器
    sp = dlib.shape_predictor(predictor_path)
    # 3. 加载人脸识别模型
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    score = []
    final_score = 0
    for modle in modle_file:
        modle = joblib.load(os.path.join(modle_path_file,modle))
        img = io.imread(input_path)
        dets = detector(img, 1)            
        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            d_test = np.array(face_descriptor) 
        test_data[0,:] = d_test


        score.append(modle.predict(test_data ))
        final_score = np.array(score).mean()
        
       # print("picture的得分：{}".format(final_score))
        return final_score
    
input = "H:/PYTHONcode/faceattractineness/ownpictures/1.jpg"

sc = caculate_score(input)
print("picture的得分：{}".format(sc))