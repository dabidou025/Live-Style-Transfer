import torch
from PIL import Image
import time

from models.stmodel import STModel
from predictor import Predictor

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_styles = 16
    st_model = STModel(n_styles)
    if True:
        st_model.load_state_dict(torch.load('D:/code/models/notmine.pth', map_location=torch.device('cpu')))
    st_model = st_model.to(device)

    img_size = 1024
    predictor = Predictor(st_model, device, img_size)

    file_name = input('file name : ')
    img = Image.open('D:/code/inputs/' + file_name + '.jpg').convert('RGB')
    
    for s in range(n_styles):
        t = time.time()
        gen = predictor.eval_image(img, s)
        print('Style', s, 'took',time.time() - t)
        gen.save('D:/code/generated/gen_' + str(s) + '.jpg')

if __name__ == '__main__':
    main()