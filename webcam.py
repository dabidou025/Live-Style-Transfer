import cv2

import torch
import numpy as np

from PIL import Image
import time

from models.stmodel import STModel
from predictor import WebcamPredictor

import argparse
import os

def webcam(args):

    img_size = args.img_size
    load_model_path = args.load_model_path
    styles_path = args.styles_path
    style_id = args.style_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_styles = len(os.listdir(styles_path))
    st_model = STModel(n_styles)
    if True:
        st_model.load_state_dict(torch.load(load_model_path, map_location=device))
    st_model = st_model.to(device)

    predictor = WebcamPredictor(st_model, device)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        _, frame = cap.read()

        frame = cv2.resize(frame, (img_size, img_size), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        frame = np.swapaxes(frame, 0, 2)
        
        gen = predictor.eval_image(frame, style_id)
        gen = np.swapaxes(gen, 0, 2)

        gen = cv2.resize(gen, (400, 300))

        cv2.imshow('Input', gen)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
       


def main():
    parser = argparse.ArgumentParser(description="Predict parse")

    parser.add_argument("--img-size", type=int, required=True)

    parser.add_argument("--load-model-path", type=str, required=True)

    parser.add_argument("--styles-path", type=str, required=True)

    parser.add_argument("--style-id", type=int, required=True)
    
    args = parser.parse_args()

    webcam(args)

if __name__ == '__main__':
    main()

