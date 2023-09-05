import cv2


imagePathGeekForGeek = "D:\\ai-res\\images\\black-dot1.jpg";
imagePath = imagePathGeekForGeek

READ_COLOR = 0
READ_GRAYSCALE = 1

def findContours(imagePath):

	path = imagePath
	gray = cv2.imread(path, 0)

	print("Gray {} . Elements = {}".format(gray.shape, gray[0:10, 2]))

	# threshold
	th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

	# findcontours
	cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
	print("\nNumber of contours = {} ".format(len(cnts)) )

	# filter by area
	s1 = 3
	s2 = 20
	xcnts = []
	for cnt in cnts:
		if s1<cv2.contourArea(cnt) <s2:
			xcnts.append(cnt)
	print("\nDots number: {}".format(len(xcnts)))

if __name__ == "__main__":
	findContours(imagePath)

