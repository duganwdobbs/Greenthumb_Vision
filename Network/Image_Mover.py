# Image Mover

# This is a simple script to move files into an image directory.
import os
import time
import filetools as ft

directory = 'F:/Greenthumb_Vision/network_log/tensorlogs/26_Feb_2018_21_00_TEST/Images/'

files     = ft.find_files(directory,ext = '.png')
x = 0
num = len(files)
for f in files:
  x += 1
  # os.rename(directory + f, directory + 'Images/' + f)
  os.remove(directory + f)
  # time.sleep(.005)
  if x%1000 == 0:
    print("%d / %d, %.2f%%"%(x,num,(x / num)))
