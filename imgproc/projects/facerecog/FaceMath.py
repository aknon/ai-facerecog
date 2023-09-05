from imutils import paths
import face_recognition
import pickle
import cv2
import os

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

 
#get paths of each file in folder named Images
#Images here contains my data(folders of various persons)
imagesDir = ""
imagePaths = list(paths.list_images(datasets))

print("Image paths {}".format(imagePaths))
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    
    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Use Face_recognition to locate faces
    boxes = face_recognition.face_locations(rgb,model='hog')
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)
    # loop over the encodings
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
#save emcodings along with their names in dictionary data
data = {"encodings": knownEncodings, "names": knownNames}
#use pickle to save data into a file for later use
f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()