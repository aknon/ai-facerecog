from PIL import Image
from pytesseract import pytesseract

imagePathTemp = "D:\\ai-res\\images\\geeks1.jfif";

tesseractPath = "D\\software\\tesseract\\tesseract.exe"

imagePath = imagePathTemp

READ_COLOR = 0
READ_GRAYSCALE = 1

def extractTextFromImage(imagePath, tesseractPath):


	# Defining paths to tesseract.exe
	# and the image we would be using
	path_to_tesseract = tesseractPath
	image_path = imagePath

	# Opening the image & storing it in an image object
	img = Image.open(image_path)

	# Providing the tesseract
	# executable location to pytesseract library
	pytesseract.tesseract_cmd = path_to_tesseract

	# Passing the image object to
	# image_to_string() function
	# This function will
	# extract the text from the image
	text = pytesseract.image_to_string(img)

	# Displaying the extracted text
	print(text[:-1])

if __name__=="__main__":
	extractTextFromImage(imagePath, tesseractPath)
