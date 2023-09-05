import cv2, sys, numpy, os

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'D:\\ai-res\\faces\\set2\\master-data'
#datasets = 'D:\\ai-res\\faces\\set3\\tmp'

READ_COLOR = 1
READ_GRAYSCALE = 0

imagesDir = "D:\\ai-res\\faces\\set1"
imageName = "Sidney_Kimmel"

modelFile = "face_matcher.yml"
imageExts = [".png", ".jpg", ".jpeg"]
shortImg = (600, 500)
FORCE_CREATE_NEW_MODEL_ON_EVERY_RUN = True

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
	show("image", image)

def show(str, image):
	cv2.imshow(str, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def grayScale(imagePath):
	#imagePath = imagePath(imagesDir, imageName)
	print("Full path of image {} ".format(imagePath))

	img = cv2.imread(imagePath)
	#show(img)
	#print("Image path = {}, shape = {}".format(imagePath, img.shape))

	#image_resize = resize_img(img)

	#show(image_resize)

	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return (img, gray_image)

def trainFaceRecogniserModel( model, images, labels):
	#model = cv2.face.LBPHFaceRecognizer_create()
	print("Training model starts....")
	
	model.train(images, labels)

	print("Training model completes")

	#model.save(modelFile)

	#print("Saving model to {} ".format(modelFile))
	return model	

def predictFace(model, names, imagePath):
	(width, height) = (130, 100)
	
	face_cascade = cv2.CascadeClassifier(haar_file)
	
	(im, gray) = grayScale(imagePath)
	
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	
	for (x, y, w, h) in faces:
		cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
		face = gray[y:y + h, x:x + w]

		show("face", face)
		
		face_resize = cv2.resize(face, (width, height))
		# Try to recognize the face
		prediction = model.predict(face_resize)
		cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

		print("---------------\n------Prediction {}".format(prediction ))
		if prediction[1]<500:

			cv2.putText(im, '% s - %.0f' %
	(names[prediction[0]], prediction[1]), (x-10, y-10),
	cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
		else:
			cv2.putText(im, 'not recognized',
	(x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

	short_img = resize_img(im)
	show("face", short_img)

def prepareTrainingData(dataset):
	# Create a list of images and a list of corresponding names
	(images, labels, names, id) = ([], [], {}, 0)
	for (subdirs, dirs, files) in os.walk(datasets):
		for subdir in dirs:
			names[id] = subdir
			subjectpath = os.path.join(datasets, subdir)
			#print("subdir {}, names{}, subjectpath {}".format(subdir, names, subjectpath))
			
			for filename in os.listdir(subjectpath):
				#print("filename {}".format(filename))
				path = subjectpath + '/' + filename
				label = id
				img = cv2.imread(path, READ_GRAYSCALE)

				#print("Each img size {}".format(img.shape) )
				images.append(img)
				labels.append(int(label))
				#print("path {}, label=id {}".format(path, label))
			id += 1
	(width, height) = (130, 100)

	(images, labels) = [numpy.array(lis) for lis in [images, labels]]

	return (images, labels, names)

def createNewModelAndTrain(images, labels):
	print("Creatint new LBPH face recogniser mode. Will also train.")
	model = createNewModel()
	trainedModel = trainFaceRecogniserModel( model, images, labels)
	return trainedModel

def createLBPHFaceRecogModel():
	print("Initialising LBPH Face recogniser model")
	model = cv2.face.LBPHFaceRecognizer_create()	
	return model

def loadOrTrainNewModel(images, labels, modelFile, FORCE_CREATE_NEW_MODEL_ON_EVERY_RUN):
	model = None

	if FORCE_CREATE_NEW_MODEL_ON_EVERY_RUN:
		print("FORCE_CREATE_NEW_MODEL_ON_EVERY_RUN = {}. Will create new model".format(FORCE_CREATE_NEW_MODEL_ON_EVERY_RUN))
		model = createNewModelAndTrain(images, labels)
	elif os.path.isfile(modelFile):
		print("Model found on file {}.".format(modelFile))
		model = loadAvailableModel(modelFile)
	else:
		print("Invalid Use case. Model cannot be created/(or not avaialble) for training. Will exit")
	return model

def createNewModel():
	model = createLBPHFaceRecogModel()
	return model

def loadAvailableModel(modelFile):
	print("Loading available model found on file {}".format(modelFile))
	model = createNewModel()
	model.read(modelFile)
	return model

def saveModel(model, modelFile):
	print("Saving model to {} ".format(modelFile))
	model.save(modelFile)

def log(images, labels, names):
	#print("Images --- {}".format(images))
	#print("Labels --- {}".format(labels))
	print("Names -- {}".format(names))

if __name__ == "__main__":
	imagePath = imagePath(imagesDir, imageName)
	(images, labels, names) = prepareTrainingData(datasets)

	log(images, labels, names)

	model = loadOrTrainNewModel( images, labels, modelFile, FORCE_CREATE_NEW_MODEL_ON_EVERY_RUN)
	
	saveModel(model, modelFile)
	print("Saved model to {} ".format(modelFile))

	predictFace(model, names, imagePath)
