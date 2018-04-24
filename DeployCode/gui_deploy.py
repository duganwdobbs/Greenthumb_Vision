from collections import deque
import cv2
from PIL import Image, ImageTk
import time
import tkinter as tk
import numpy as np
from SplashScreen import SplashScreen
import webbrowser as wb
from Juan_Image_Trial import JuanImageTrial

from   deploy_v2     import Deploy_Network

plants   = [ 'Apple','Cherry','Corn','Grape','Peach','Strawberry','Bell_Pepper','Potato','Tomato']

def visit_site():
  wb.open('https://greenthumbvision.com')

def quit_(root):
  root.destroy()

def run_network(net,cam):
  (readsuccessful, img) = cam.read()
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # img = cv2.resize(img,(960,540))
  imgH,imgW,imgC = img.shape
  lef_w = (imgW - 256) //2
  rig_w = lef_w + 256
  top_h = (imgH - 256) //2
  bot_h = top_h + 256
  tl = (lef_w-3,top_h-3)
  br = (rig_w+3,bot_h+3)
  red= (255,0,0)
  cv2.rectangle(img,tl,br,red,3)
  img = img[top_h:bot_h,lef_w:rig_w,:]
  net.run(img)

def update_image(image_label, cam):
  (readsuccessful, img) = cam.read()
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # img = cv2.resize(img,(960,540))
  imgH,imgW,imgC = img.shape
  lef_w = (imgW - 256) //2
  rig_w = lef_w + 256
  top_h = (imgH - 256) //2
  bot_h = top_h + 256
  tl = (lef_w-3,top_h-3)
  br = (rig_w+3,bot_h+3)
  red= (255,0,0)
  cv2.rectangle(img,tl,br,red,3)
  ret_img = img[top_h:bot_h,lef_w:rig_w,:]
  img     = img[:,::-1,:]
  a = Image.fromarray(img)
  b = ImageTk.PhotoImage(image=a)
  image_label.configure(image=b)
  image_label._image_cache = b  # avoid garbage collection
  master.update()
  return ret_img


def update_fps(fps_label,net,img):
  try:
    p_log,d_log = net.run(img)
    p_log = np.squeeze(p_log)
    d_log = np.squeeze(d_log)

    plant_cl  = np.argmax(np.copy(p_log))
    plant_per = p_log[plant_cl]
    plant_str = net.p_log_to_desc(plant_cl)

    disea_cl  = np.argmax(np.copy(d_log))
    disea_per = d_log[disea_cl]
    disea_str = net.d_log_to_desc(plant_cl,disea_cl)

    fps_label.configure(text='Plant: %s %.2f %% \nDisease: %s %.2f %%'%(plant_str,plant_per,disea_str,disea_per))
  except:
    print("Error?")


def update_all(root, image_label, cam, fps_label,net):
  img = update_image(image_label, cam)
  update_fps(fps_label,net,img)
  root.after(10, func=lambda: update_all(root, image_label, cam, fps_label, net))


if __name__ == '__main__':
  groot = tk.Tk()

  # groot.overrideredirect(True)
  splash = JuanImageTrial(groot)
  groot.title("Loading...")
  groot.update_idletasks()

  net = Deploy_Network()
  groot.update_idletasks()

  # USB
  # gst_str = ("v4l2src device=/dev/video{} ! "
  #              "video/x-raw, width=(int){}, height=(int){}, format=(string)RGB ! "
  #              "videoconvert ! appsink").format(0, 1920, 1080)\
  # cam =  cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
  # TX2
  # gst_str = ("nvcamerasrc ! "
  #              "video/x-raw(memory:NVMM), width=(int)2592, height=(int)1458, format=(string)I420, framerate=(fraction)30/1 ! "
  #              "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
  #              "videoconvert ! appsink").format(960, 720)
  # cam = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
  cam = cv2.VideoCapture(0)
  run_network(net,cam)

  # root.destroy()
  groot.withdraw()

  master = tk.Toplevel()
  master.title("Greenthumb-Vision")

  loading_label = tk.Label(master=master)# label for the loading frame
  loading_label.grid(row=0,column=0,columnspan=4,rowspan=3)

  loading_label.grid_forget()
  image_label = tk.Label(master=master)# label for the video frame
  image_label.grid(row=0,column=0,columnspan=4)

  fps_label = tk.Label(master=master)# label for fps
  fps_label.grid(row=1,column=1,columnspan=2)

  # quit button
  quit_button = tk.Button(master=master, text='Quit',command=lambda: quit_(groot))
  quit_button.grid(row=1, column=0)

  help_button = tk.Button(master=master, text='Click Here For Help!',command =lambda: visit_site())
  help_button.grid(row=1, column=3)

  master.protocol("WM_DELETE_WINDOW", lambda: quit_(groot))

  # setup the update callback
  master.after(0, func=lambda: update_all(master, image_label, cam, fps_label, net))
  master.mainloop()
