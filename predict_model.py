import torch
from PIL import Image
import time

from models.stmodel import STModel
from predictor import Predictor

import argparse
import os

def predict(args):

    img_size = args.img_size
    load_model_path = args.load_model_path
    input_path = args.input_path
    styles_path = args.styles_path
    save_generated_path = args.save_generated_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_styles = len(os.listdir(styles_path))
    st_model = STModel(n_styles)
    if True:
        st_model.load_state_dict(torch.load(load_model_path, map_location=device))
    st_model = st_model.to(device)

    predictor = Predictor(st_model, device, img_size)

    img = Image.open(input_path).convert('RGB')
    
    for s in range(n_styles):
        t = time.time()
        gen = predictor.eval_image(img, s)
        print('Style', s, 'took', time.time() - t)
        gen.save(save_generated_path + '/gen_' + str(s+1) + '.jpg')


def main():
    parser = argparse.ArgumentParser(description="Predict parse")

    parser.add_argument("--img-size", type=int, required=True)

    parser.add_argument("--load-model-path", type=str, required=True)

    parser.add_argument("--input-path", type=str, required=True)

    parser.add_argument("--styles-path", type=str, required=True)

    parser.add_argument("--save-generated-path", type=str, required=True)
    
    args = parser.parse_args()

    predict(args)

if __name__ == '__main__':
    main()