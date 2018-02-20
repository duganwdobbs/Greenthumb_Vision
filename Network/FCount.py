#MatVision
import h5py
# import pylab
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.image as mp_img
import sys
import os
image_directory = 'E:/BinaLab-Semantic-Segmentation/data/'
file_list = [f for f in os.listdir(image_directory) if f.endswith('.mat')]



def calc_freq(freq,labelarr):
  # input(np.histogram(labelarr,bins = [0,1,2,3,4,5,6,7])[0])
  # print(frequency)
  freq = freq + np.histogram(labelarr,bins = [0,1,2,3,4,5,6,7])[0]
  return freq

displayf = False
displayi = False

#Gets the five class frequency list.
def getFiveClassArr():
  freq = getFreqArr()
  freq = np.array([freq[1],freq[2],freq[3],freq[4]+freq[0]+freq[5],freq[6]])
  copyfreq = np.copy(freq)
  copyfreq.sort()
  median = copyfreq[2]
  avg = np.average(freq)

  freq = np.array(median/freq)
  freq = [max(f,.5) for f in freq]

  # freq = np.array(freq/avg)
  # freq = 1/np.tanh(freq)

  return np.reshape(freq,(5,1))

def getFreqArr():
  frequency = np.array([0,0,0,0,0,0,0])

  for f in file_list:
    label_filename = image_directory + f
    labelbak = h5py.File(label_filename)

    try:
      label=labelbak
      label = label.get('f')
      labelarr = np.array(label)
      labelarr = labelarr - 1
      labelarr = np.transpose(labelarr)
      frequency = calc_freq(frequency,labelarr)
      if displayf:
        displaymat(label_filename,labelarr)

    except TypeError:
      try:
        label=labelbak
        label = label.get('ind')
        labelarr = np.array(label)
        labelarr = labelarr - 1
        labelarr = np.transpose(labelarr)
        frequency = calc_freq(frequency,labelarr)
        if displayi:
          displaymat(label_filename,labelarr)
      except TypeError:
        print(f)

  classes = np.array([ "Car", "Tree", "Water", "Building", "Ground", "Boat", "Road"])
  median = np.sort(frequency)[int(frequency.size/2)]
  average= np.average(frequency)
  mul    = median/frequency
  av     = average/frequency
  perc   = frequency/np.sum(frequency)

  return(frequency)


if __name__ == "__main__":
  print(getFiveClassArr())
