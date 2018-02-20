#Filetoos!
import os

# This function recursivly makes directories.
def directoryFixer(directory):
  directory = "".join(str(x) for x in directory)
  try:
    os.stat(directory)
  except:
    try:
      os.mkdir(directory)
    except:
      subDir = directory.split('/')
      while (subDir[-1] == ''):
          subdir = subdir[:-1]
      newDir = ""
      for x in range(len(subDir)-2):
        newDir += (subDir[x])
        newDir += ('/')
      print ("Attempting to pass... " + str(newDir))
      directoryFixer(newDir)
      os.mkdir(directory)

# This function finds all files of given extention in given path.
def find_files(path,ext = '.png'):
  return [path + '/' + f for f in os.listdir(path) if f.endswith(ext)]

# This function uses find_files to find ALL FILES recursivly in given path root
def parse_dir(base_directory,ext = '.png'):
  returnlist = []
  for x in os.walk(base_directory):
      x = x[0].replace('\\','/')
      # print("Walking: "+x)
      appendlist = find_files(x,ext)
      if appendlist:
        returnlist.append(appendlist)
  ret_list = []
  for r in returnlist:
    for s in r:
      ret_list.append(s)
  return ret_list
