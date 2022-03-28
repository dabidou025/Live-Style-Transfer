# Live-Style-Transfer

Authors: *Matthieu Denis, Guillaume Bril, Alexandre Gommez, Hubert de Lesquen*

The purpose of this project is to offer a transfer of style in real time with the webcam.

To run the code in collab: (slow due to Google limitations)


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dabidou025/Live-Style-Transfer/blob/main/Live_Style_Transfer.ipynb)

To run the style transfer on your computer's webcam :
```markdown
python webcam.py \
--img-size 512 \
--load-model-path models/st_model_512_80k_12.pth \
--styles-path styles
```

To run the style transfer on the input folder's image :
```markdown
python .\predict_model.py \
--load-model-path models/st_model_512_80k_12.pth \
--save-generated-path generated \
--styles-path style_pictures \
--img-size 512 \
--input-path inputs/yourfile.jpg
```

To train the model :
```markdown
python train_main.py \
--dataset-path yourdatasetpath \
--styles-path style_pictures \
--save-model-path models \
--dataset-size 50000 \
--img-size 512 \
--n-epochs 1 \
--batch-size 4 \
--lr 0.001 \
--style-factor 80000
```


Sources: 

https://github.com/ryanwongsa/Real-time-multi-style-transfer

https://github.com/vindruid/yolov3-in-colab.git
