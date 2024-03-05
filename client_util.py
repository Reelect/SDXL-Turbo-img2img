import cv2
import numpy as np
from keras.models import load_model
from utils.datasets import get_labels
from utils.preprocessor import preprocess_input

USE_WEBCAM = True

video_capture = cv2.VideoCapture(0)

frame_window = 10
cam = None

# 감정
emotion_classifier = load_model('./models/emotion_model.hdf5')
emotion_labels = get_labels('fer2013')
emotion_target_size = emotion_classifier.input_shape[1:3]

# 얼굴 탐지 모델 가중치
cascade_filename = './models/haarcascade_frontalface_alt.xml'
# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


# 성별
gender_net = cv2.dnn.readNetFromCaffe(
    './models/deploy_gender.prototxt',
    './models/gender_net.caffemodel')

gender_list = ['boy', 'girl']


def videoDetector(img):
    # resize
    img1 = cv2.resize(img, dsize=None, fx=1.0, fy=1.0)

    # 그레이 스케일 변환
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cascade 얼굴 탐지 알고리즘
    results = cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    if len(results) > 1:
        return f"{len(results)} persons, "
    elif len(results) == 1:
        x, y, w, h = results[0]
        face = img[int(y):int(y + h), int(x):int(x + h)].copy()  # for age, gender
        gray_face = gray2[y:y + h, x:x + w]  # for emotion
        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            return None

        # emotion
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]

        # gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_preds.argmax()


        # 분석 결과
        info = f"{gender_list[gender]}, {emotion_text}, "

        # 콘솔 출력
        return info
    else:
        return ""
