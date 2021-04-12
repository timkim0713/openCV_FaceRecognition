import cv2
import numpy as numpy
import face_recognition

imgTim = face_recognition.load_image_file("images/tim1.jpg")
imgTim = cv2.cvtColor(imgTim, cv2.COLOR_BGR2RGB)


# imgTest = face_recognition.load_image_file("images/timTest.jpg")  # result true
imgTest = face_recognition.load_image_file("images/bill.jpg")  # result false
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


# detect face

FaceLocation = face_recognition.face_locations(imgTim)[
    0
]  # only first image as we have only 1
encodeTim = face_recognition.face_encodings(imgTim)[0]
cv2.rectangle(
    imgTim,
    (FaceLocation[3], FaceLocation[0]),
    (FaceLocation[1], FaceLocation[2]),
    (255, 0, 255),
    2,
)

FaceLocationTest = face_recognition.face_locations(imgTest)[
    0
]  # only first image as we have only 1
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(
    imgTest,
    (FaceLocationTest[3], FaceLocationTest[0]),
    (FaceLocationTest[1], FaceLocationTest[2]),
    (255, 0, 255),
    2,
)


# find the distance btwn faces

results = face_recognition.compare_faces([encodeTim], encodeTest)
faceDistance = face_recognition.face_distance([encodeTim], encodeTest)
print(results, faceDistance)
cv2.putText(
    imgTest,
    f"{results}{round(faceDistance[0],2)}",
    (50, 50),
    cv2.FONT_HERSHEY_COMPLEX,
    1,
    (0, 255, 255),
    2,
)


cv2.imshow("tim", imgTim)
cv2.imshow("tim test", imgTest)
cv2.waitKey(5000)
