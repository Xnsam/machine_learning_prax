import face_recognition
import cv2
import numpy as np

kiddo_image = face_recognition.load_image_file('/content/img/1609085547361.jpg')
kiddo_face_encoding = face_recognition.face_encodings(kiddo_image)

known_face_encodings = [kiddo_face_encoding]
known_face_names = ["Moxa Doshi"]

face_locations = []
face_encodings = []
face_names = []


frame = cv2.imread('/content/img/1609085554719.jpg')
# frame = cv2.imread('/content/test/1565367372895.jpg') 

small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
rgb_small_frame = small_frame[:, :, ::-1]

face_locations = face_recognition.face_locations(rgb_small_frame)
face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

face_names = []

for face_encoding in face_encodings:
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
  name = "Unknown"

  # face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  # best_match_index = np.argmin(face_distances)
  # print(matches)
  val = any(matches[0])
  # print(val)
  if val:
    name = known_face_names[0]
  
  face_names.append(name)

for (top, right, bottom, left), name in zip(face_locations, face_names):
  top *= 4
  right *= 4
  bottom *= 4
  left *= 4

  cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)

  cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
  font = cv2.FONT_HERSHEY_DUPLEX
  cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 2)

cv2.imwrite('test.png', frame)