import tensorflow as tf
import cv2
import numpy as np
import random

model = tf.keras.models.load_model(
    'image final\keras_model.h5', compile=False)  # 載入模型
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
cp = ['paper', 'scissor', 'rock']


def text(text):
    global img
    global img1
    img1 = img.copy()
    cv2.putText(img1, text, (0, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('', img1)


def clip(cap):
    ret, frame = cap.read()
    img = cv2.resize(frame, (398, 224))
    img = img[0:224, 80:304]
    image_array = np.asarray(img)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    a, b, c, bg = prediction[0]
    return a, b, c, bg, img


def judge_game(p1, p2):
    if p1 == 'rock':
        if p2 == 'paper':
            text("lose")
        elif p2 == 'scissor':
            text("win")
        elif p2 == 'rock':
            text("deuce")
    elif p1 == 'paper':
        if p2 == 'paper':
            text("win")
        elif p2 == 'scissor':
            text("lose")
        elif p2 == 'rock':
            text("deuce")
    elif p1 == 'scissor':
        if p2 == 'paper':
            text("win")
        elif p2 == 'scissor':
            text("deuce")
        elif p2 == 'rock':
            text("lose")


cap = cv2.VideoCapture(0)
while True:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    if cv2.waitKey(1) == ord('p'):
        index = random.randint(0, 2)
        text('your turn!')
        cv2.waitKey(1000)
        text('3')
        cv2.waitKey(1000)
        text('2')
        cv2.waitKey(1000)
        text('1')
        cv2.waitKey(1000)
        a, b, c, bg, img = clip(cap)
        if a > 0.8:
            text('scissor')  # 使用 text() 函式，顯示文字
            player = 'scissor'
        if b > 0.8:
            text('rock')
            player = 'rock'
        if c > 0.8:
            text('paper')
            player = 'paper'
        judge_game(player, cp[index])
        tmp = "enemy is " + cp[index]
        cv2.putText(img1, tmp, (0, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('', img1)
        cv2.waitKey(3000)
        continue

    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, (398, 224))
    img = img[0:224, 80:304]
    image_array = np.asarray(img)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    a, b, c, bg = prediction[0]
    if a > 0.8:
        text('scissor')  # 使用 text() 函式，顯示文字
    elif b > 0.8:
        text('rock')
    elif c > 0.8:
        text('paper')

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
