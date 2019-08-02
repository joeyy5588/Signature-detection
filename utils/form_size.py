import os
import cv2

a = {}
prefix = '../data/handwriting/'
l = os.listdir(prefix)
for i in l:
	fn = prefix + i
	img = cv2.imread(fn)
	key = img.shape[0]
	key = round(key, 2)
	if key not in a:
		a[key] = 1
	else:
		a[key] += 1
print(a)