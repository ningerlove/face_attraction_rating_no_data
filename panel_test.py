from tkinter import *
from numpy import *
from PIL import Image,ImageTk
import PIL
import tkinter
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#matplotlib.use('TkAgg')
from caculate_score import *
canvas_width = 300
canvas_height = 400



def getInputs():
    try:
        picture = str(Picture.get())
    except:
        picture = "H:/PYTHONcode/faceattractineness/ownpictures/1.jpg"
        print("输入图片地址")
        tolNentry.delete(0, END)
        tolNentry.insert(0, "H:/PYTHONcode/faceattractineness/ownpictures/1.jpg")

    return picture

def drawNewPicture():
    
    picture = getInputs()
    im=PIL.Image.open(picture)
    out = im.resize((128*3, 128*3))
    img=ImageTk.PhotoImage(out)
    label.config(image = img)
    label.inage = img
    
    
    Score_fen_ = caculate_score(picture)
    Score_fen.set(Score_fen_)





def main(root):
    root.title("颜值打分工具")
    #Label(root, text="颜值打分工具").grid(row=0,column=1)
   
    Label(root, text="输入图片").grid(row=1, column=0,sticky=E)    
    global Picture
    Picture = Entry(root)
    Picture.grid(row=1, column=1,sticky=W)
    Picture.insert(0, 'H:/PYTHONcode/faceattractineness/ownpictures/1.jpg')
    
    Button(root, text="确定", command=drawNewPicture).grid(row=1, column=2,sticky=E)
    
    global label
    im=PIL.Image.open("H:/PYTHONcode/faceattractineness/ownpictures/1.jpg")
    out = im.resize((128*2, 128*2))
    img=ImageTk.PhotoImage(out)
    label=tkinter.Label(root,image = img)
    label.grid(row=2, column=0,columnspan = 3,sticky=W,padx=40,pady=10)
    label.image = img


    Label(root, text="打分").grid(row=3, column=0,sticky=E)
    global Score,Score_fen
    Score_fen = StringVar() 
    Score = Entry(root,textvariable=Score_fen)
    Score.grid(row=3, column=1,sticky=W)

    quit = Button(root, text="QUIT", fg="red",
                              command=root.destroy)

    quit.grid(row=3, column=2,sticky=E)
if __name__ == "__main__":

    # 创建一个事件
    root = tkinter.Toplevel()
    # test_widget_text(root)
    main(root)

    # 启动事件循环
    root.mainloop()