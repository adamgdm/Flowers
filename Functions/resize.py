import cv2

define resize(img, target_size):
	if target_size is None:
		target_size = (512, 512)
	
	img = cv2.imread(img)
	if img is None:
		return None
	
	img = cv2.resize(img, target_size)
	if img is None:
		return None
	
	return img
