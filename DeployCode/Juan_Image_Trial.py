from tkinter import *
from PIL import Image, ImageTk
import cv2
import tkinter.ttk as ttk
import numpy as np

class JuanImageTrial:
    def __init__(self, parent):

        self.parent = parent
        #parent.overrideredirect(True)

        #self.progressbar = ttk.Progressbar(orient=HORIZONTAL, length=10000, mode='determinate')
        #self.progressbar.pack(side="bottom")
        #self.progressbar.start()

        self.setImage()
        root.juanSplash()
        self.juanWindow()
        
    def setImage(self):
        #Kept for later reference to variable
        self.chaboi = Image.open('logo.png')

    def juanSplash(root):
        # Trying to bypass garbage collection
        root.imgSplash = imgSplash = ImageTk.PhotoImage(file=r'logo.png')
        
    def juanWindow(self):
        size1, size2 = self.chaboi.size

        adjustedsize1 = (self.parent.winfo_screenwidth()-size1)//2
        adjustedsize2 = (self.parent.winfo_screenheight()-size2)//2

        self.parent.geometry("%ix%i+%i+%i" %(size1, size2,
                                             adjustedsize1,adjustedsize2))

        self.label = Label(self.parent, image=root.imgSplash)
        self.label.image = root.imgSplash
        self.label.pack()

if __name__ == '__main__':
    root = Tk()

    root.overrideredirect(True)
    app = JuanImageTrial(root)
    #progressbar.start()

    root.after(6010, root.destroy)

    root.mainloop()
