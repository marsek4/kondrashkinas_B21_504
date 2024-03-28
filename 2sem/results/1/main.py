import numpy as np
from PIL import Image

def main():

    def image_to_numpy_array(image_path):
        #открытие изображения
        image = Image.open(image_path)
        #преобразование изображения в np массив
        image_array = np.array(image)
        return image_array

    def semitone(img):
        return (0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 2]).astype(np.uint8)


    def tosemitone(imgname):
        img = image_to_numpy_array(imgname)
        return Image.fromarray(semitone(img), 'L')
    
    image_1 = tosemitone('input/im1.png')
    image_1.save("output/im_1.png")

if __name__ == "__main__": 
    main()