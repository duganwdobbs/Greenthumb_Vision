# Deploy Test
from   PIL        import Image
from   deploy     import Deploy_Network

import tensorflow as     tf
import numpy      as     np

import filetools
import sys
import os

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
    net = Deploy_Network()
    ps  = net.get_plants()
    ds  = net.get_diseases()
    paths = sys.argv[1:]
    if paths[0] is 'v':
      fs = filetools.find_files(paths[1],ext = '.png')
      paths = [paths[1] + '/' + f for f in fs]
      print(paths)

    for d in range(1,len(paths)):
      v     = paths[d]
      image = Image.open(v)
      image = np.asarray(image)
      p_log,d_log = net.run(image)
      print(p_log)
      input(d_log)
      # p_log = p_log * 100
      # d_log = d_log * 100
      # print("Image %d, %s, %s"%(d+1,ps[p_log],ds[p_log][d_log]))




if __name__ == '__main__':
  tf.app.run()
