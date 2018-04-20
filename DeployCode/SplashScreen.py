from tkinter import *
import tkinter as Tk
import cv2
import numpy as np
from PIL import ImageTk
from PIL import Image

class SplashScreen(Frame):
    def __init__(self, master=None, width=0.2, height=0.2, useFactor=True):
        Frame.__init__(self, master)
        self.pack(side=TOP, fill=BOTH, expand=YES)
        self.root = master

        # get screen width and height
        ws = self.master.winfo_screenwidth()
        hs = self.master.winfo_screenheight()
        w = (useFactor and ws*width) or width
        h = (useFactor and ws*height) or height
        # calculate position x, y
        x = (ws/2) - (w/2) 
        y = (hs/2) - (h/2)
        self.master.geometry('%dx%d+%d+%d' % (w, h, x, y))
        
        self.master.overrideredirect(True)
        self.lift()
        self.config(bg="#65ff32")
        m = Label(master, text="I'm Sorry This Is Taking So Long :'( \n\n\nPls Don't Close.")
        m.pack(side=TOP, expand=YES)
        m.config(bg="#65ff32", justify=CENTER, font=("calibri", 29))

        #img = Image.open("logo.png")
        #img = np.asarray(img)
        #print(img.shape)
        #img = ImageTk.PhotoImage(img)
        #loading_label = Label(self, image = img)# label for the loading frame
        #loading_label.image = img
        #loading_label.pack()

        master.update_idletasks()

    def close(self):
        self.root.destroy()


# splash = SplashScreen(Tk.Tk())
