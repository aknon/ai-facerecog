from PIL import Image



imagePathTemp = "D:\\ai-res\\images\\sample1.jpg";

imagePath = imagePathTemp

READ_COLOR = 0
READ_GRAYSCALE = 1

def removeChannel(imagePath):

	# Creating a image object, of the sample image
	img = Image.open(imagePath)

	# A 12-value tuple which is a transform matrix for dropping
	# green channel (in this case)
	matrix = ( 1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0)

	# Transforming the image to RGB using the aforementioned matrix
	img = img.convert("RGB", matrix)

	# Displaying the image
	img.show()


if __name__=="__main__":
	removeChannel(imagePath)