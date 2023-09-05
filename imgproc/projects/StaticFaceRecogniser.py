import cv2
import os
import matplotlib.pyplot as plt

imagesDir = "D:\\ai-res\\faces\\set1"

imageName = "1-6"

READ_COLOR = 0
READ_GRAYSCALE = 1
imageExts = [".png", ".jpg", ".jpeg"]
shortImg = (600, 500)

def imagePath(imagesDir, imageName):
	path = ""
	for ext in imageExts:
		path = os.path.join(imagesDir, imageName + ext)
		if os.path.isfile(path):
			break
	return path
	
def resize_img(img):
	img_resized = cv2.resize(img, shortImg)
	return img_resized

def show(image):
	cv2.imshow("image", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


imagePath = imagePath(imagesDir, imageName)
print("Full path of image {} ".format(imagePath))

img = cv2.imread(imagePath)
show(img)
print("Image path = {}, shape = {}".format(imagePath, img.shape))

image_resize = resize_img(img)

show(image_resize)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Gray scale image shape {}".format(gray_image.shape))

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

print("face {}".format(face))
for (x, y, w, h) in face:
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
	img_rgb = img#cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	#plt.figure(figsize=(20,10))
	#plt.imshow(img_rgb)
	#plt.axis('off')
	show(img_rgb)

