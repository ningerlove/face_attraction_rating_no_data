import os,dlib,glob,numpy
from skimage import io
import cv2,re

predictor_path = './shape_predictor_68_face_landmarks.dat'
# 2.人脸识别模型
face_rec_model_path =  './dlib_face_recognition_resnet_model_v1.dat'
# 3.候选人脸文件夹
faces_folder_path = './candidate-faces/'
# 4.需识别的人脸
img_path = 'test3.jpg'

# 1.加载正脸检测器
detector = dlib.get_frontal_face_detector()
# 2.加载人脸关键点检测器
sp = dlib.shape_predictor(predictor_path)
# 3. 加载人脸识别模型
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
win = dlib.image_window()
# 候选人脸描述子list
descriptors = {}
# 对文件夹下的每一个人脸进行:
# 1.人脸检测
# 2.关键点检测
# 3.描述子提取
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print(("Processing file: {}".format(f)))
    pattern = re.compile(r"\\\w+.jpg")
    name = pattern.findall(f)[0][1:-4]    
    img = io.imread(f)
    b, g, r = cv2.split(img)    
    img2 = cv2.merge([r, g, b])
    win.clear_overlay()
    win.set_image(img)

    # 1.人脸检测
    dets = detector(img, 1)
    print(("Number of faces detected: {}".format(len(dets))))

    for k, d in enumerate(dets):  
        # 2.关键点检测
        shape = sp(img, d)
        # 画出人脸区域和和关键点
        win.clear_overlay()
        win.add_overlay(d)
        win.add_overlay(shape)

        # 3.描述子提取，128D向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        # 转换为numpy array
        v = numpy.array(face_descriptor)  
        descriptors[name] = v

# 对需识别人脸进行同样处理
# 提取描述子，不再注释
img = io.imread(img_path)
dets = detector(img, 1)

dist = {}
for k, d in enumerate(dets):
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    d_test = numpy.array(face_descriptor) 

    # 计算欧式距离
    for i in descriptors.keys():
        dist_ = numpy.linalg.norm(descriptors[i]-d_test)
        dist[i] =  dist_

# 候选人名单
candidate = {'liudehua':'刘德华','liuye':'刘烨','wuyanzu':"吴彦祖",'chengguanxi':"程冠希",'yuwenle':"余文乐",'wenzhang':"文章"}
cd_sorted = sorted(iter(dist.items()), key=lambda c:c[1])
print("候选人可能性排序："+str(cd_sorted))
if cd_sorted[0][1]> 0.5:
    print("没有找到候选人")
else:
    print("候选人是："+candidate[cd_sorted[0][0]])
dlib.hit_enter_to_continue()
