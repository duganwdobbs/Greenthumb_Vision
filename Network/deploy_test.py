# Deploy Test
from   PIL        import Image
from   deploy     import Deploy_Network

import tensorflow as     tf
import numpy      as     np

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
    for d in range(1,len(sys.argv)):
      v     = sys.argv[d]
      image = Image.open(v)
      image = np.asarray(image)
      p_log,d_log = net.run(image)
      # p_log = p_log * 100
      # d_log = d_log * 100
      for x in range(len(ps)):
        print("\n%s : %f, "%(ps[x],p_log[x]))
        for y in range(len(ds[x])):
          print("%s : %f, "%(ds[x][y],d_log[x][y]),end=" ")




if __name__ == '__main__':
  tf.app.run()
