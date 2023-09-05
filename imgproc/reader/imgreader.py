#import tkinter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from skimage import io

#matplotlib.use('TkAgg')



imagePathPic1 = "D:\\ai-res\\images\\pic1.png";
imagePathGeekForGeek = "D:\\ai-res\\images\\geeks14.png";
imagePath = imagePathGeekForGeek

READ_COLOR = 1
READ_GRAYSCALE = 0
 

def show(str, image):
	cv2.imshow(str, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def read_img_with_mat():
	img=cv2.imread( imagePath)

	#Displaying image using plt.imshow() method

	
	plt.imshow(img)
 
	#hold the window
	plt.waitforbuttonpress()
	plt.close('all')

def read_img_with_cv2(imagePath):

	arr = np.array([0,9,8])
	arr2 = np.array( [[9,8,1], [78, 88, 45], [ 89,67,32], [12, 34, 56]])
	print("array 1D {} ",format(arr))

	print("array 2D {} ",format(arr2))
	# To read image from disk, we use
	# cv2.imread function, in below method,


	# img = cv2.imread("https://media.geeksforgeeks.org/wp-content/uploads/20220604150230/pic23-200x141.png", cv2.IMREAD_COLOR)
	#img = io.imread("https://media.geeksforgeeks.org/wp-content/uploads/20220604150230/pic23-200x141.png", cv2.IMREAD_COLOR)
	img = cv2.imread(imagePath , READ_COLOR)

	x= 100
	y = 10
	w = 30
	h = 90

	cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 3)

	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#cv2.imshow("image", img)
	#cv2.rectangle(img, (110,110), (140,140), (0,255,0), 3)

	#cv2.imshow("image", img)
	#cv2.rectangle(img, (10,150), (100,250), (0,0,255), 3)

	print("Image = {} , type = {}".format(img.shape, type(img)))
	show("Image", img)
	face = gray_image[y:y+h, x:x+w]
	show("face", face)



 
def rgb(imagePath):
	image = cv2.imread(imagePath)
	B, G, R = cv2.split(image)
	#Corresponding channels are separated
    
	cv2.imshow("original", image)
	cv2.waitKey(0)
  
	cv2.imshow("blue", B)
	cv2.waitKey(0)
	  
	cv2.imshow("Green", G)
	cv2.waitKey(0)
	  
	cv2.imshow("red", R)
	cv2.waitKey(0)
	  
	cv2.destroyAllWindows()

def main():
	read_img_with_cv2(imagePath)
	#rgb(imagePath)
	#read_img_with_mat()

def getT():
	(a, b, c) = ("123", "abc", "ert")
	return (a, b, c)

if __name__== "__main__":
	images = [[2, 3, 4],[5,6,7]]
	labels = [34, 56]
	for i in [images, labels]:
		print("in {}".format(i) )
		print("numpy {}".format(np.array(i)) )

	t = getT()
	(a, b, c) = t
	print("a--- {}".format(a ) )
	main()



