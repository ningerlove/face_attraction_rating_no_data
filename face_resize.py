import os       
import xlrd
from PIL import Image


picture_path = "H:/PYTHONcode/faceattractineness/Data_Collection"
resize_picture_path = "H:/PYTHONcode/faceattractineness/Data_Collection_resize"
rating_xlx_path ="H:/PYTHONcode/faceattractineness/Rating_Collection/Attractiveness_label.xlsx" 

rating_data = xlrd.open_workbook(rating_xlx_path).sheet_by_index(0)   
rating = rating_data.col_values(1)
#print(rating)

#图片名字：SCUT-FBP-446.jpg
img_file = os.listdir(picture_path)
n_img = len(img_file)
#print(img)

#图片形状(498, 654)

#img_priginal_sise = img.size
#print(img_priginal_sise)

for i in range(n_img):
    img=Image.open(os.path.join(picture_path,img_file[i]))
    out = img.resize((128, 128))
    out.save(os.path.join(resize_picture_path,img_file[i]))


