import numpy as np

cmat = np.asarray(
[[  1133021,    602123],
 [   228521,  15988335]]

    )
classes = cmat.shape[1]
tp = np.zeros(classes)
fp = np.zeros(classes)
fn = np.zeros(classes)
for x in range(classes):
  for y in range(classes):
    if(x == y):
      tp[x] = tp[x] + cmat[x][y]
    else:
      fn[x] = fn[x] + cmat[x][y]
      fp[y] = fp[y] + cmat[x][y]

tn = np.sum(cmat) - (tp + fp)

accuracy    = 100 * (tp + tn) / (tp + tn + fp + fn)
precision   = 100 * tp / (tp + fp)
recall      = 100 * tp / (tp + fn)
IOU         = 100 * tp / (tp + fp + fn)
specificity = 100 * tn / (tn + fp)
fonescore   = 2 * 1 / ( 1 / precision + 1 / recall)

np.set_printoptions(precision=3)

print('PER CLASS METRICS:')

print("Accuracy:    ",end="")
print(accuracy)
print("Avg:         %2f"%np.mean(accuracy))
print("Precision:   ",end="")
print(precision)
print("Avg:         %2f"%np.mean(precision))
print("Recall:      ",end="")
print(recall)
print("Avg:         %2f"%np.mean(recall))
print("IOU:         ",end="")
print(IOU)
print("Avg:         %2f"%np.mean(IOU))
print("Specificity: ",end="")
print(specificity)
print("Avg:         %2f"%np.mean(specificity))
print("F1:          ",end="")
print(fonescore)
print("Avg:         %2f"%np.mean(fonescore))

print('\n\nFULL DATA METRICS:')

tp = np.sum(tp)
tn = np.sum(tn)
fp = np.sum(fp)
fn = np.sum(fn)


accuracy    = 100 * (tp + tn) / (tp + tn + fp + fn)
precision   = 100 * tp / (tp + fp)
recall      = 100 * tp / (tp + fn)
IOU         = 100 * tp / (tp + fp + fn)
specificity = 100 * tn / (tn + fp)
fonescore   = 2 * 1 / ( 1 / precision + 1 / recall)

print("Accuracy:    %2f"%(accuracy))
print("Precision:   %2f"%(precision))
print("IOU:         %2f"%(IOU))
print("Specificity: %2f"%(specificity))
print("F1:          %2f"%(fonescore))
