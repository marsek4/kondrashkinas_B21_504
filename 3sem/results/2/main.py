import numpy as np
from PIL import Image, ImageChops

def main():

    #метод консервативного сглаживания
    def conservative_smoothing(image):
        result = np.copy(image)
        rows, cols = image.shape

        for i in range(1, rows-1):
            for j in range(1, cols-1):
                window = image[i-1:i+2, j-1:j+2]
                median_value = np.median(window)
                result[i, j] = median_value

        return result
    
    #загрузка изображения
    input_image = Image.open('im1.png')
    filt_img = input_image.convert('L')
    filt_img.show()
    image_array = np.array(filt_img)

    #применение метода консервативного сглаживания
    smoothed_array = conservative_smoothing(image_array)

    #создание изображения из массива
    smoothed_img = Image.fromarray(smoothed_array.astype(np.uint8))
    smoothed_img.show()

    # Получаем разность между исходным и отфильтрованным изображениями
    diff_img = ImageChops.difference(filt_img, smoothed_img)
    diff_img.show()

if __name__ == "__main__": 
    main()