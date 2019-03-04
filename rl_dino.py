import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import webbrowser
import time
import pyautogui as pag
import cv2
from collections import deque
import tensorflow as tf
import tensorflow.keras as keras
import sys

GAME_LEFT_OFFSET = 20
GAME_TOP_OFFSET = 150
GAME_RIGHT_OFFSET = 650
GAME_BOTTOM_OFFSET = 40

NN_PENALTY = 3.0
NN_LESS_PENALTY = 0.8
NN_REWARD = 1.0
NN_PENALTY_FRAMES_COUNT= 6
NN_LESS_PENALTY_FRAMES_COUNT= 4

NN_HIDDEN_SIZE = 15

NN_INPUT_WIDTH_RAW = (GAME_RIGHT_OFFSET + GAME_LEFT_OFFSET)//2
NN_INPUT_WIDTH = NN_INPUT_WIDTH_RAW//2 + 1
NN_INPUT_HEIGHT = (GAME_TOP_OFFSET + GAME_BOTTOM_OFFSET)//2

OUTPUT_LABELS = ['', 'space', 'down']

_nn_model = None

def open_game(cmd=None):
    browser = None
    blist = ["chrome", "firefox", "google-chrome", "chromium", "chromium-browser", "mozilla"]
    if cmd is not None:
        blist = blist + [cmd]
    for b in blist:
        try:
            browser = webbrowser.get(b)
            break
        except:
            pass
    if browser is None:
        print("Unable to find either chrome or firefox browser!!!!\n" +
            "If one of them is already installed but still I am unable to detect, " + 
            "please pass correct command to open them as 'string' parameter to the script " + 
            "with '%s' to let me know where to insert url in the command to open them")
        raise Exception("Unable to find chrome or firefox")

    browser.open_new("file:///" + os.path.join(os.getcwd(), "t-rex-runner-gh-pages", "index.html"))


def get_model():
    global _nn_model
    if _nn_model is None:
        _nn_model = keras.Sequential([
            keras.layers.Flatten(input_shape=(NN_INPUT_HEIGHT, NN_INPUT_WIDTH)),
            keras.layers.Dense(NN_HIDDEN_SIZE, activation=tf.nn.tanh),
            keras.layers.Dense(3, activation=tf.nn.softmax)
        ])
        _nn_model.compile(optimizer=tf.train.AdamOptimizer(), 
                         loss=keras.losses.mean_squared_error)
    return _nn_model
        

def is_game_over(screenshot, dino_go_template):
    result = cv2.matchTemplate(screenshot, dino_go_template, cv2.TM_SQDIFF_NORMED)
    if np.min(result) < 0.01:
        return True
    return False

def get_prediction(inp_img):
    model = get_model()
    pred = model.predict(np.array([inp_img]))
    # cv2.imshow('input', inp_img)
    # cv2.waitKey(1)
    # print(pred[0], OUTPUT_LABELS[np.argmax(pred[0])])
    return OUTPUT_LABELS[np.argmax(pred[0])]

def get_model_input(img):
    inp_img = cv2.cvtColor(cv2.resize(img[:, :NN_INPUT_WIDTH_RAW], fx=0.5, fy=0.5, dsize=(0,0)), cv2.COLOR_BGR2GRAY)/255.0
    _ , inp_img = cv2.threshold(inp_img, 0.5, 1.0, cv2.THRESH_BINARY_INV)
    return inp_img

def train_model(X, Y, W):
    model = get_model()
    model.fit(X, Y, sample_weight=W, shuffle=True)

def get_training_data(model_hist):
    X = []
    Y = []
    W = []
    for i in range(min(len(model_hist), NN_PENALTY_FRAMES_COUNT)): # penalty for last frames
        inp, out = model_hist.pop()
        X.append(inp)
        Y.append([0.1 if l==out else 0.45 for l in OUTPUT_LABELS])
        W.append(NN_PENALTY)
    #     cv2.imshow("%d" % i, inp)
    # cv2.waitKey(0)
    for i in range(min(len(model_hist), NN_LESS_PENALTY_FRAMES_COUNT)): # less penalty for next set of frames in the last
        inp, out = model_hist.pop()
        X.append(inp)
        Y.append([0.1 if l==out else 0.45 for l in OUTPUT_LABELS])
        W.append(NN_LESS_PENALTY)
    while len(model_hist)>0: # reward for first frames
        inp, out = model_hist.pop()
        X.append(inp)
        Y.append([0.9 if l==out else 0.05 for l in OUTPUT_LABELS])
        W.append(NN_REWARD)
    return X, Y, W

def get_game_screenshot(dino_pos):
    return np.array(pag.screenshot().crop((
                dino_pos[0] - GAME_LEFT_OFFSET, 
                dino_pos[1] - GAME_TOP_OFFSET, 
                dino_pos[0] + GAME_RIGHT_OFFSET, 
                dino_pos[1] + GAME_BOTTOM_OFFSET
            )))

def train(dino_pos):
    dino_go_temp = cv2.imread("dino_game_over.png")
    gcount = 1
    model_hist = deque(maxlen=3000)
    X, Y, W = [], [], []
    while True:
        down_pressed = False
        time.sleep(2 if gcount%5==0 else 0)
        while True:
            pag.press('space')
            time.sleep(0.1)
            im = get_game_screenshot(dino_pos)
            if not is_game_over(im, dino_go_temp):
                break
        time.sleep(2.5)
        while True:
            im = get_game_screenshot(dino_pos)
            if is_game_over(im, dino_go_temp):
                print("Finished game %d" % gcount)
                break
            im_inp = get_model_input(im)
            pred = get_prediction(im_inp)
            if pred == 'down':
               if not down_pressed:
                   pag.keyDown('down')
                   down_pressed = True
            elif down_pressed:
                pag.keyUp('down')
                down_pressed = False
            if pred == 'space':
                pag.press(pred, interval=0.01)
            model_hist.append((im_inp, pred))
        x, y, w = get_training_data(model_hist)
        X += x
        Y += y
        W += w
        if gcount%2 == 0:
            train_model(np.array(X), np.array(Y), np.array(W))
            X, Y, W = [], [], []
        model_hist.clear()
        gcount += 1
        if down_pressed:
            pag.keyUp('down')

if __name__ == "__main__":
    print("Starting the program. Please DO NOT TOUCH the computer because the program will be controlling keyboard!!!")
    print("If you want to stop the program, there will be 2 sec pause every 5 games, use that opportunity :)")
    time.sleep(2)

    try:
        open_game(sys.argv[1] if len(sys.argv)>1 else "")
        time.sleep(5)
    except:
        print("Unable to open game. Exiting...")
        exit(1)
    
    time.sleep(2)

    pos = pag.locateCenterOnScreen('dino.png')
    if pos is None:
        print("Unable to find game on screen! Please check if browser window is opening correctly...")
        exit(2)
    
    pag.click(pos[0], pos[1])
    train(pos)
