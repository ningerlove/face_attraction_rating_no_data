import os,xlrd
from PIL import Image
import numpy as np
from sklearn.linear_model import Ridge,LinearRegression,Lasso,  ElasticNet
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

picture_path = "H:/PYTHONcode/faceattractineness/Data_Collection"
faces_folder_path = "H:/PYTHONcode/faceattractineness/Data_Collection_resize"
rating_xlx_path ="H:/PYTHONcode/faceattractineness/Rating_Collection/Attractiveness_label.xlsx" 
faces_vector_path = "H:/PYTHONcode/faceattractineness/faces_vector"
predictor_path = './shape_predictor_68_face_landmarks.dat'
modle_path_file = "H:/PYTHONcode/faceattractineness/modle_file"

img_vecor_file = os.listdir(faces_vector_path)
n_img = len(img_vecor_file)
#print(n_img)
all_x_data = np.zeros((n_img,128))

modles = [Lasso,Ridge,LinearRegression, ElasticNet]

for f in img_vecor_file:
    i = int(f.split("-")[2].split(".")[0])
    picture_vector = np.loadtxt(os.path.join(faces_vector_path,f))
    all_x_data[i-1,:] = picture_vector

rating_data = xlrd.open_workbook(rating_xlx_path).sheet_by_index(0)   
rating = rating_data.col_values(1)
all_y_data = np.array(rating[1:])



X_train, X_test, y_train, y_test = train_test_split(all_x_data, all_y_data, test_size=0.4, random_state=0)
print(X_train[:10,:])
#reg.coef_
mean_error_all = {}
r2_all = {}
for modle in modles:
    reg_modle = modle()
    reg_modle.fit(X_train,y_train)
    y_predict =reg_modle.predict(X_test)
    mean_error = mean_squared_error( y_predict,y_test)
    r2 = r2_score(y_predict,y_test)
   # print("{}的平均误差：{}".format(modle,mean_error))
    #print("{}的r2：{}".format(modle,r2))
    mean_error_all[str(modle)] = mean_error
    r2_all[str(modle)] = r2

best_modle = sorted(iter(mean_error_all.items()), key=lambda c:c[1])
for i in range(len(modles)):

    print("第{}名{}的平均误差：{}".format(i+1,best_modle[i][0],best_modle[i][1]))
    print("第{}名{}的r2：{}".format(i+1,best_modle[i][0],r2_all[best_modle[i][0]]))



modle = Ridge()
n_alphas = 200
param_grid = dict(alpha = np.logspace(-4, -1, n_alphas))
grid = GridSearchCV(modle, param_grid, cv=10)

grid.fit(X_train,y_train)
print(grid.best_params_)

# =============================================================================
# mean_error_test = []
# x = np.logspace(-10, 10, n_alphas)
# for alpha in np.logspace(-2, -1.5, n_alphas):
#     
#     modle = Ridge(alpha = alpha)
#     modle.fit(X_train,y_train)
#    
#     print("{}模型的平均误差：{}".format(alpha,mean_squared_error(modle.predict(X_test),y_test)))
#     mean_error_test.append(mean_squared_error( modle.predict(X_test),y_test))
# 
# 
# plt.plot(x,mean_error_test)
# =============================================================================




# =============================================================================
# final_modle = grid.best_estimator_
# final_modle_y_predict =final_modle.predict(X_test)
# print("最好的模型的平均误差：{}".format(mean_squared_error(final_modle_y_predict,y_test)))
# print("最好的模型的r2：{}".format( r2_score(final_modle_y_predict,y_test)))
# =============================================================================
# =============================================================================
# red_modle =str(reg_LinearRegression)+".m"
# joblib.dump(reg_LinearRegression, os.path.join(modle_path_file,red_modle))
# =============================================================================


