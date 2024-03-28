from PIL import Image

def main():

    # Функция для передискретизации изображения в K раз за один проход по алгоритму ближайшего соседа
    def resample_one_pass(image, K):
        width, height = image.size
        new_width = int(width * K)
        new_height = int(height * K)
        resampled_image = image.resize((new_width, new_height), Image.NEAREST)
        return resampled_image

    # Загрузка исходного изображения
    input_image = Image.open('input/linux.png')

    # Демонстрация исходного изображения
    input_image.show()

    # Выполнение операций передискретизации
    M = 2
    N = 3
    K = M / N

    resampled_one_pass_image = resample_one_pass(input_image, K)
    print("Resampled Image Size (One Pass):", resampled_one_pass_image.size)

    # Демонстрация результатов
    resampled_one_pass_image.show()


if __name__ == "__main__": 
    main()