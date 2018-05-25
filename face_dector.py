import os,dlib,glob,numpy
from skimage import io
import cv2,re
from PIL import Image
# 4.需识别的人脸
faces_folder_path = "H:/PYTHONcode/faceattractineness/Data_Collection_resize"
rating_xlx_path ="H:/PYTHONcode/faceattractineness/Rating_Collection/Attractiveness_label.xlsx" 
faces_vector_path = "H:/PYTHONcode/faceattractineness/faces_vector"
predictor_path = './shape_predictor_68_face_landmarks.dat'
# 2.人脸识别模型
face_rec_model_path =  './dlib_face_recognition_resnet_model_v1.dat'


#n_img = 500
img_file = os.listdir(faces_folder_path)
n_img = len(img_file)
#print(n_img)
#print(img_file)
# 1.加载正脸检测器
detector = dlib.get_frontal_face_detector()
# 2.加载人脸关键点检测器
sp = dlib.shape_predictor(predictor_path)
# 3. 加载人脸识别模型
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
win = dlib.image_window()
# 候选人脸描述子list

for f in img_file:
    i = int(f.split("-")[2].split(".")[0])
    vector_txt = f.split(".")[0]+'.txt'
    #print(j)
    img=io.imread(os.path.join(faces_folder_path,f))
    b, g, r = cv2.split(img)    
    img2 = cv2.merge([r, g, b])
    # 1.人脸检测
    dets = detector(img2, 1)
    print(dets)
    print(("Number of faces : {}".format(i)))
    
    for k, d in enumerate(dets):  
        # 2.关键点检测
        shape = sp(img2, d)
        # 画出人脸区域和和关键点
        win.clear_overlay()
        win.add_overlay(d)
        win.add_overlay(shape)
    
        # 3.描述子提取，128D向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        # 转换为numpy array
        v = numpy.array(face_descriptor)  
        numpy.savetxt(os.path.join(faces_vector_path,vector_txt), v)


#下面的代码将所有图片转为向量，由于机器原因，部分图片通过上面方式无法转为向量
# =============================================================================
# import os,dlib,glob,numpy
# from skimage import io
# import cv2,re
# from PIL import Image
# 
# picture_path = "H:/PYTHONcode/faceattractineness/Data_Collection"
# faces_folder_path = "H:/PYTHONcode/faceattractineness/Data_Collection_resize"
# rating_xlx_path ="H:/PYTHONcode/faceattractineness/Rating_Collection/Attractiveness_label.xlsx" 
# faces_vector_path = "H:/PYTHONcode/faceattractineness/faces_vector"
# predictor_path = './shape_predictor_68_face_landmarks.dat'
# # 2.人脸识别模型
# face_rec_model_path =  './dlib_face_recognition_resnet_model_v1.dat'
# 
# 
# # 1.加载正脸检测器
# detector = dlib.get_frontal_face_detector()
# # 2.加载人脸关键点检测器
# sp = dlib.shape_predictor(predictor_path)
# # 3. 加载人脸识别模型
# facerec = dlib.face_recognition_model_v1(face_rec_model_path)
# win = dlib.image_window()
# 
# img_vecor__file = os.listdir(faces_vector_path)
# n_img = len(img_vecor__file)
# print(n_img)
# 
# n = []
# for f in img_vecor__file:
#     i = int(f.split("-")[2].split(".")[0])
#     n.append(i)
# 
# m = []    
# for j in range(1,501):
#     if j not in n:
#         print(j)
#         m.append(j)
# 
# print(len(m))
# 
# for i in m:
#     f = "SCUT-FBP-"+str(i)+".jpg"
#     print(f)
#     i = int(f.split("-")[2].split(".")[0])
#     vector_txt = f.split(".")[0]+'.txt'
#     #print(j)
#     #img=Image.open(os.path.join(faces_folder_path,f))
#     #img.show()
#     
#     img=io.imread(os.path.join(picture_path,f))
#     b, g, r = cv2.split(img)    
#     img2 = cv2.merge([r, g, b])
#     # 1.人脸检测
#     dets = detector(img2, 1)
#     print(("Number of faces : {}".format(i)))
#     
#     for k, d in enumerate(dets):  
#         # 2.关键点检测
#         shape = sp(img2, d)
#         # 画出人脸区域和和关键点
#         win.clear_overlay()
#         win.add_overlay(d)
#         win.add_overlay(shape)
#     
#         # 3.描述子提取，128D向量
#         face_descriptor = facerec.compute_face_descriptor(img, shape)
#         # 转换为numpy array
#         v = numpy.array(face_descriptor)  
#         numpy.savetxt(os.path.join(faces_vector_path,vector_txt), v)
# 
# =============================================================================
