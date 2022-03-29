# import dependencies
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from models.stmodel import STModel
from predictor import Predictor
import argparse
from glob import glob
import os
from ipywidgets import Box, Image
import gradio as gr

def predict_gradio(image):
    img_size = 512
    load_model_path = "./models/st_model_512_80k_12.pth"
    styles_path = "./styles/" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_styles = len(glob(os.path.join(styles_path, '*.jpg')))
    st_model = STModel(n_styles)
    if True:
        st_model.load_state_dict(torch.load(load_model_path, map_location=device))
    st_model = st_model.to(device)

    predictor = Predictor(st_model, device, img_size)

    list_gen=[]
    for s in range(n_styles):
        gen = predictor.eval_image(image, s)
        list_gen.append(gen)
    return list_gen

def gradio_pls():
    description="""
Upload a photo and click on submit to see the 12 styles applied to your photo. \n 
Keep in mind that for compatibility reasons your photo is cropped before the neural net applied the different styles.
<center>
<table><tr>
<td><img src="https://raw.githubusercontent.com/dabidou025/Live-Style-Transfer/main/styles/a_muse_picasso.jpg" width=100px></td>
<td><img src="https://raw.githubusercontent.com/dabidou025/Live-Style-Transfer/main/styles/britto.jpg" width=100px></td>

<td><img src="https://raw.githubusercontent.com/dabidou025/Live-Style-Transfer/main/styles/cat.jpg" width=100px></td>
<td><img src="https://raw.githubusercontent.com/dabidou025/Live-Style-Transfer/main/styles/cubist.jpg" width=100px></td>

<td><img src="https://raw.githubusercontent.com/dabidou025/Live-Style-Transfer/main/styles/fractal.jpg" width=100px></td>
<td><img src="https://raw.githubusercontent.com/dabidou025/Live-Style-Transfer/main/styles/horse.jpg" width=100px></td>

<td><img src="https://raw.githubusercontent.com/dabidou025/Live-Style-Transfer/main/styles/monet.jpg" width=100px></td>
<td><img src="https://raw.githubusercontent.com/dabidou025/Live-Style-Transfer/main/styles/sketch.jpg" width=100px></td>

<td><img src="https://raw.githubusercontent.com/dabidou025/Live-Style-Transfer/main/styles/starry_night.jpg" width=100px></td>
<td><img src="https://raw.githubusercontent.com/dabidou025/Live-Style-Transfer/main/styles/texture.jpg" width=100px></td>

<td><img src="https://raw.githubusercontent.com/dabidou025/Live-Style-Transfer/main/styles/tsunami.jpg" width=100px></td>
<td><img src="https://raw.githubusercontent.com/dabidou025/Live-Style-Transfer/main/styles/vibrant.jpg" width=100px></td>

</tr>
</table>
</center>
"""
    iface = gr.Interface(
        predict_gradio,   
        [
            gr.inputs.Image(type="pil", label="Image"),
        ],
        [
            gr.outputs.Carousel("image", label="Style"),
        ],
        layout="unaligned",
        title="Photo Style Transfer",
        description=description,
        theme="grass",
        allow_flagging='never'
    )

    return iface.launch(inline=True, height=800, width=800)