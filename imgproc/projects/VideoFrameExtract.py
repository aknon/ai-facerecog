# Program To Read video
# and Extract Frames

import cv2


videoPathTemp = "D:\\ai-res\\videos\\video1.mp4";


videoPath = videoPathTemp

READ_COLOR = 0
READ_GRAYSCALE = 1

# Function to extract frames
def frameCapture(path):

	# Path to video file
	vidObj = cv2.VideoCapture(path)

	# Used as counter variable
	count = 0

	# checks whether frames were extracted
	success = 1

	while success:

		# vidObj object calls read
		# function extract frames
		success, image = vidObj.read()

		print("Success = {}".format(success))
		print("Image frame {}".format(str(image)))

		# Saves the frames with frame-count
		cv2.imwrite("frame%d.jpg" % count, image)

		count += 1


# Driver Code
if __name__ == '__main__':

	# Calling the function
	frameCapture(videoPath)
