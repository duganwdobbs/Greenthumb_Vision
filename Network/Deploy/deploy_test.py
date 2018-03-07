# Deploy Test
from   PIL        import Image
from   deploy     import Deploy_Network

import tensorflow as     tf
import numpy      as     np

import filetools
import sys
import os

# Runs a single image from given path
def run_from_image_path(path):
  # try:
    image = Image.open(path)
    image = np.asarray(image)
    p_logs,d_logs = net.run(image)
    net.result_verbose(p_logs,d_logs)
  # except:
  #   print('Invalid Path!')

def run_from_direcotry(path):
  fs = filetools.find_files(path,ext = '.png')
  files = [path + '/' + f for f in fs]
  for d in range(1,len(files)):
    v     = files[d]
    print(v)
    run_from_image_path(v)

# Receives a list of directories and runs all images inside those directories.
def predef_images(paths):
  paths = paths[1:]
  for path in paths:
    run_from_directory(path)

# An interactive mode where you can give an endless number of images to the
# network through command line.
def interactive_dir():
  quit = False
  while not quit:
    path = input("Enter the path to the folder containing images to run, or \'quit\' to exit.")
    if path == 'quit':
      break
    run_from_direcotry(path)

# An interactive mode where you can give an endless number of images to the
# network through command line.
def interactive_img():
  quit = False
  while not quit:
    path = input("Enter the path to the image to run, or \'quit\' to exit.")
    if path == 'quit':
      break
    run_from_image_path(path)

def main(_):
  np.set_printoptions(precision=2)
  try:
    bad_data = False
    path = sys.argv[0]
    bad_data = len(sys.argv) is 1
  except:
    bad_data = True
  if bad_data:
    print("Invalid argument, please provide example path(s)")
  else:
    global net
    net = Deploy_Network()
    paths = sys.argv[1:]
    # print(paths[0])
    if paths[0] == 'v':
      predef_images(paths)
    if paths[0] == 'i':
      interactive_img()
    if paths[0] == 'iv':
      interactive_dir()

if __name__ == '__main__':
  tf.app.run()
