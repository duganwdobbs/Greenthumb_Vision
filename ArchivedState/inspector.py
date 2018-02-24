# INSECTOR
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def inspect(file):
  print_tensors_in_checkpoint_file(file_name=file, tensor_name='', all_tensors=False)
  input("PRESS ENTER TO CONTINUE...")
