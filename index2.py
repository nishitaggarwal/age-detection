# Import Libraries
import cv2
import numpy as np

# File paths
GENDER_PROTO = 'weights/deploy_gender.prototxt'
GENDER_MODEL = 'weights/gender_net.caffemodel'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
FACE_PROTO = "weights/deploy.prototxt.txt"
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
AGE_PROTO = 'weights/deploy_age.prototxt'
AGE_MODEL = 'weights/age_net.caffemodel'
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(20-25)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

# Frame size
frame_width = 1280
frame_height = 720

# Load models
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)


def get_faces(frame, confidence_threshold=0.5):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104, 177.0, 123.0))
    face_net.setInput(blob)
    output = np.squeeze(face_net.forward())
    faces = []
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x, start_y, end_x, end_y = start_x - 10, start_y - 10, end_x + 10, end_y + 10
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = max(0, end_x)
            end_y = max(0, end_y)
            faces.append((start_x, start_y, end_x, end_y))
    return faces


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    return gender_net.forward()


def get_age_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    age_net.setInput(blob)
    return age_net.forward()


def predict_age_and_gender():
    """Predict the gender and age of faces in a video stream."""
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frame = img.copy()
        if frame.shape[1] > frame_width:
            frame = image_resize(frame, width=frame_width)
        faces = get_faces(frame)
        for (start_x, start_y, end_x, end_y) in faces:
            face_img = frame[start_y:end_y, start_x:end_x]
            if face_img.shape[0] == 0 or face_img.shape[1] == 0:
                continue  # skip empty detections
            age_preds = get_age_predictions(face_img)
            gender_preds = get_gender_predictions(face_img)
            gender_idx = gender_preds[0].argmax()
            gender = GENDER_LIST[gender_idx]
            gender_confidence_score = gender_preds[0][gender_idx]
            age_idx = age_preds[0].argmax()
            age = AGE_INTERVALS[age_idx]
            age_confidence_score = age_preds[0][age_idx]
            label = f"{gender}-{gender_confidence_score*100:.1f}%, {age}-{age_confidence_score*100:.1f}%"
            print(label)
            yPos = start_y - 15
            if yPos < 15:
                yPos = start_y + 15
            box_color = (255,0,0) if gender == "Male" else (147, 20, 255)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
            cv2.putText(frame, label, (start_x, yPos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, box_color, 2)
        cv2.imshow("Gender Estimator", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_age_and_gender()
