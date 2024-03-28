import numpy as np
from PIL import Image

def main():

    #алгоритм Отсу
    def otsu_thresholding(image_array):
        #получение ширины и высоты изображения
        height, width = image_array.shape

        #вычисление гистограммы яркости
        histogram = [0] * 256
        for y in range(height):
            for x in range(width):
                pixel_value = image_array[y, x]
                histogram[pixel_value] += 1

        #нормализация гистограммы
        total_pixels = height * width
        normalized_histogram = [count / total_pixels for count in histogram]

        #вычисление суммы гистограммы
        total = 0
        for i in range(256):
            total += i * normalized_histogram[i]

        #вычисление глобальной дисперсии
        sum_back = 0
        weight_back = 0
        maximum_variance = 0
        threshold = 0

        for i in range(256):
            weight_back += normalized_histogram[i]
            weight_fore = 1 - weight_back
            if weight_back == 0:
                continue
            if weight_fore == 0:
                break
            sum_back += i * normalized_histogram[i]
            mean_back = sum_back / weight_back
            mean_fore = (total - sum_back) / weight_fore
            between_variance = weight_back * weight_fore * (mean_back - mean_fore) ** 2
            if between_variance > maximum_variance:
                maximum_variance = between_variance
                threshold = i

        #применение порога к изображению
        binary_image = (image_array > threshold) * 255
        return binary_image
    
    #чтение изображения
    image = Image.open("input/im1.png'").convert("L")  #предполагаем, что изображение в градациях серого
    image_array = np.array(image)

    #применение алгоритма Отсу
    binary_image = otsu_thresholding(image_array)

    #преобразование np массива обратно в изображение PIL и сохранение результата
    binary_image_pil = Image.fromarray(binary_image.astype(np.uint8))
    binary_image_pil.save("output/im_1.png")

if __name__ == "__main__": 
    main()