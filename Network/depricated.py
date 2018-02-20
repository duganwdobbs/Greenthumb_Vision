# Expects Patch Size [1,H,W,1] or number of patches to extract.
def ImageToPatch(image,patch_size = None,patches = None):
  rate = [1,1,1,1]
  if patch_size is None:
    patH = image.shape[1] / patches
    patW = image.shape[2] / patches
    patch_size = [1,patH,patW,1]
  stride = patch_size # This can be changed to allow overlap, but must the
                      # inverse function must be changed.
  patches = tf.extract_image_patches([image],patch_size,patch_size,rate,'VALID')
  #patches = tf.space_to_batch
  return patches

# Eexpects Image Size [H,W,C] or number of patches.
def PatchToImage(patch,image_size = None,patches = None):
  rate = [1,1,1,1]
  if image_size is None:
    imgH = patch.shape[1] * patches
    imgW = patch.shape[2] * patches
    chan = patch.shape[3] / patches
    image_size = [imgH,imgW,chan]
    recon_size = [1, imgH, imgW, chan]
  # Changing format from [C,H,W,B] -> [B,H,W,C]
  # B -> Batch
  # C -> Channels (Features)
  form = (patch.shape[3],patch.shape[1],patch.shape[2],patch.shape[0])
  patch = tf.reshape(patch,form)
  recon = tf.reshape(patch,recon_size)
  recon = tf.space_to_depth(recon,)
