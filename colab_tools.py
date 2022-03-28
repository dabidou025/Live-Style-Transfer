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
from predictor import WebcamPredictor
import argparse
from glob import glob
import os
from ipywidgets import Box, Image
#matplotlib inline

# JavaScript to properly create our live video stream using our webcam as input
def video_stream():
  js = Javascript('''
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    var imgElement;
    var labelElement;
    
    var pendingResolve = null;
    var shutdown = false;
    var refresh_inst = false;
    
    function removeDom() {
       stream.getVideoTracks()[0].stop();
       video.remove();
       div.remove();
       video = null;
       div = null;
       stream = null;
       imgElement = null;
       captureCanvas = null;
       labelElement = null;
    }

    function animate() {
      setTimeout(function() {
        if (!shutdown) {
        requestAnimationFrame(animate);}
          if (pendingResolve) {
              var result = "";
              captureCanvas.getContext('2d').drawImage(video, 0, 0, 320, 240);
              result = captureCanvas.toDataURL('image/jpeg', 0.5)
              var lp = pendingResolve;
              pendingResolve = null;
              lp(result);
            }
        }, 1000 / 30);
      }
    
    async function createDom() {
      if (shutdown){
        removeDom();
        shutdown = true;
        return '';
      }

      if (div !== null) {
        return stream;
      }

      div = document.createElement('div');
      div.style.border = '2px solid black';
      div.style.padding = '3px';
      div.style.width = '100%';
      div.style.maxWidth = '300px';
      document.body.appendChild(div);
      
      const modelOut = document.createElement('div');
      modelOut.innerHTML = "<span>Status:</span>";
      labelElement = document.createElement('span');
      labelElement.innerText = 'No data';
      labelElement.style.fontWeight = 'bold';
      modelOut.appendChild(labelElement);
      div.appendChild(modelOut);
           
      video = document.createElement('video');
      video.style.visibility = 'hidden';
      video.width = div.clientWidth - 6;
      video.setAttribute('playsinline', '');
      video.onclick = () => { shutdown = true; };
      stream = await navigator.mediaDevices.getUserMedia(
          {video: { facingMode: "user", frameRate: { exact: 30 }}});
      div.appendChild(video);

      const refresh = document.createElement('button');
      refresh.innerHTML = 
          'Refresh';
      div.appendChild(refresh);
      refresh.onclick = () => { refresh_inst = true; };

      const instruction = document.createElement('button');
      instruction.innerHTML = 
          'Stop';
      div.appendChild(instruction);
      instruction.onclick = () => { shutdown = true; };

      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute';
      imgElement.style.zIndex = 1;
      div.appendChild(imgElement);
      
      video.srcObject = stream;
      await video.play();

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 320; //video.videoWidth;
      captureCanvas.height = 240; //video.videoHeight;
      window.requestAnimationFrame(animate);
      
      return stream;
    }
    async function stream_frame(label, imgData) {
      if (shutdown) {
        removeDom();
        shutdown = true;
        return '';
      }

      if (refresh_inst) {
        removeDom();
        refresh_inst = false;
        return '';
      }
      stream = await createDom();
      
      if (label != "") {
        labelElement.innerHTML = label;
      }
            
      if (imgData != "") {
        var videoRect = video.getClientRects()[0];
        imgElement.style.top = videoRect.top + "px";
        imgElement.style.left = videoRect.left + "px";
        imgElement.style.width = videoRect.width + "px";
        imgElement.style.height = videoRect.height + "px";
        imgElement.src = imgData;
      }
      
      var result = await new Promise(function(resolve, reject) {
        pendingResolve = resolve;
      });
      shutdown = false;
      
      return {'img': result};
    }
    ''')

  display(js)

# function to convert the JavaScript object into an OpenCV image
def js_to_image(js_reply):
  """
  Params:
          js_reply: JavaScript object containing image from webcam
  Returns:
          img: OpenCV BGR image
  """
  # decode base64 image
  image_bytes = b64decode(js_reply.split(',')[1])
  # convert bytes to numpy array
  jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
  # decode numpy array into OpenCV BGR image
  img = cv2.imdecode(jpg_as_np, flags=1)

  return img

# function to convert OpenCV mask image into base64 byte string to be overlayed on video stream
def mask_to_bytes(mask_array):
  """
  Params:
          mask_array: Numpy array (pixels) containing rectangle to overlay on video stream.
  Returns:
        bytes: Base64 image byte string
  """
  # convert array into PIL image
  mask_PIL = PIL.Image.fromarray(mask_array, 'RGB')
  iobuf = io.BytesIO()
  # format bbox into png for return
  mask_PIL.save(iobuf, format='jpeg')
  # format return string
  mask_bytes = 'data:image/jpeg;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

  return mask_bytes

def live_style_transfert(style_id):
    # start streaming video from webcam
    video_stream()
    # label for video
    label_html = 'Capturing...'

    #initialize the Net
    img_size = 512
    load_model_path = "./models/st_model_512_80k_12.pth"
    styles_path = "./styles/" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_styles = len(glob(os.path.join(styles_path, '*.jpg')))
    st_model = STModel(n_styles)
    if True:
        st_model.load_state_dict(torch.load(load_model_path, map_location=device))
    st_model = st_model.to(device)
    predictor = WebcamPredictor(st_model, device) 
    mask = ""

    while True:
        js_reply = eval_js('stream_frame("{}", "{}")'.format(label_html, mask))
        if not js_reply:
            break
        # convert JS response to OpenCV Image
        frame = js_to_image(js_reply["img"])
        # call our style_predictor on video frame
        frame = cv2.resize(frame, (img_size, img_size), fx=0.5, fy=0.5)
        frame = np.swapaxes(frame, 0, 2)
        gen = predictor.eval_image(frame, style_id)
        gen = np.swapaxes(gen, 0, 2)
        gen = cv2.resize(gen, (256, 256))
        #  draw the mask image
        mask_array = np.zeros([256,256, 3], dtype=np.uint8)
        mask_array[:,:,:] = (gen[:,:,:]).astype(float) 
        # convert overlay into bytes
        mask_bytes = mask_to_bytes(mask_array)
        # update bbox so next frame gets new overlay
        mask = mask_bytes

def live_app():
  from ipywidgets import IntSlider, ToggleButton
  style = IntSlider(min=1, max=12)
  print("To launch the app, make a choice on the SlideBar (it could take 10 seconds to launch)")
  print("To change the style, move the SlideBar and click on Refresh!")
  print("To properly stop the demo, click on Stop!")
  display(style)


  def on_value_change(change):
      live_style_transfert(change['new'])

  style.observe(on_value_change, names='value')

def display_style():
    styles_path = "./styles/" 
    list_file = glob(os.path.join(styles_path, '*.jpg'))
    list_widgets=[]
    for file in list_file:
      file = open(file, "rb")
      image = file.read()
      image = Image(value=image,width=100,height=100,)
      image.layout.object_fit ='contain'
      list_widgets.append(image)
    box = Box(children=list_widgets)
    return print('Try to match these styles with what you see on your webcam', display(box))