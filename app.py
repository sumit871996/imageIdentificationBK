import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Create a FastAPI application
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Define the paths to cascade classifier XML files
fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
frontalface_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
upperbody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')

# Define a route for image classification for path
@app.get('/identify')
async def identify_image(image_path: str):
    # Read the image from the provided path
    cv_image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Detect humans using different cascade classifiers
    fullbody_humans = fullbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    frontalface_humans = frontalface_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    upperbody_humans = upperbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Print the number of humans detected by each classifier
    print("Full Body Humans: ", len(fullbody_humans))
    print("Frontal Face Humans: ", len(frontalface_humans))
    print("Upper Body Humans: ", len(upperbody_humans))


    # Check if humans are detected using each cascade classifier
    if len(fullbody_humans) > 0 and len(frontalface_humans) > 0 and len(upperbody_humans) > 0:
        class_label = 'Human'
        confidence = 1.0
    # elif len(frontalface_humans) > 0:
    #     class_label = 'Human (Frontal Face)'
    #     confidence = 1.0
    # elif len(upperbody_humans) > 0:
    #     class_label = 'Human (Upper Body)'
    #     confidence = 1.0
    else:
        class_label = 'Non-Human'
        confidence = 0.0

    # Prepare the response
    response = {
        'class_label': class_label,
        'confidence': float(confidence)
    }

    return response

@app.post('/image')
async def identify_image(image: bytes = File(...)):

    # Convert the image bytes to a NumPy array
    nparr = np.frombuffer(image, np.uint8)

    # Read the image from the NumPy array
    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Detect humans using different cascade classifiers
    fullbody_humans = fullbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    frontalface_humans = frontalface_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    upperbody_humans = upperbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Print the number of humans detected by each classifier
    print("Full Body Humans: ", len(fullbody_humans))
    print("Frontal Face Humans: ", len(frontalface_humans))
    print("Upper Body Humans: ", len(upperbody_humans))

    # Check if humans are detected using each cascade classifier
    if len(fullbody_humans) > 0 and len(frontalface_humans) > 0 and len(upperbody_humans) > 0:
        class_label = 'Human'
        confidence = 1.0
    else:
        class_label = 'Non-Human'
        confidence = 0.0

    # Prepare the response
    response = {
        'class_label': class_label,
        'confidence': float(confidence)
    }

    return response