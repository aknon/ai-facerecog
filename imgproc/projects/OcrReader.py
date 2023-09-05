# Import required packages
import cv2
#import pytesseract

imagePathGeekForGeek = "D:\\ai-res\\images\\ocr1.jpg";
imagePath = imagePathGeekForGeek

READ_COLOR = 0
READ_GRAYSCALE = 1

# Mention the installed location of Tesseract-OCR in your system
tesseractPath = "D\\software\\tesseract\\tesseract.exe"


def readTextFromImage(imagePath, tesseractPath):

	#pytesseract.pytesseract.tesseract_cmd = tesseractPath

	# Read image from which text needs to be extracted
	img = cv2.imread(imagePath)


	# Preprocessing the image starts

	# Convert the image to gray scale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Performing OTSU threshold
	ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

	# Specify structure shape and kernel size.
	# Kernel size increases or decreases the area
	# of the rectangle to be detected.
	# A smaller value like (10, 10) will detect
	# each word instead of a sentence.
	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

	# Applying dilation on the threshold image
	
	## Nitesh:: 8-11-2023 - Adding 'dilation' and commenting our dilation will have different effects
	#dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

	# Finding contours
	contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL,
													cv2.CHAIN_APPROX_NONE)

	# Creating a copy of image
	im2 = img.copy()

	# A text file is created and flushed
	file = open("recognized.txt", "w+")
	file.write("")
	file.close()

	# Looping through the identified contours
	# Then rectangular part is cropped and passed on
	# to pytesseract for extracting text from it
	# Extracted text is then written into the text file
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		
		# Drawing a rectangle on copied image
		rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
		# Cropping the text block for giving input to OCR
		#cropped = im2[y:y + h, x:x + w]
		
		# Open the file in append mode
		#file = open("recognized.txt", "a")
		
		# Apply OCR on the cropped image
		#text = pytesseract.image_to_string(cropped)
		
		# Appending the text into file
		#file.write(text)
		#file.write("\n")
		
		# Close the file
		#file.close
		#cropImg(im2, x, y, w, h, file)

		cv2.imshow("image", im2)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def cropImg(image, x, y, h, w, file):

	cropped = image[y:y+h, x:x+w]

	file = open("recognised.txt", "a")
	text = ""
	#text = pytesseract.image_to_string(cropped)

	file.write(text)
	file.write("\n")
	file.close()

if __name__==("__main__"):
	readTextFromImage(imagePath, tesseractPath)