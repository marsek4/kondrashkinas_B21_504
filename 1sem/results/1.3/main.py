from PIL import Image

def main():

    # Функция для растяжения изображения в M раз по алгоритму ближайшего соседа
    def stretch_image(image, M):
        width, height = image.size
        new_width = width * M
        new_height = height * M
        stretched_image = image.resize((new_width, new_height), Image.NEAREST)
        return stretched_image

    # Функция для сжатия изображения в N раз по алгоритму ближайшего соседа
    def compress_image(image, N):
        width, height = image.size
        new_width = width // N
        new_height = height // N
        compressed_image = image.resize((new_width, new_height), Image.NEAREST)
        return compressed_image

    # Функция для передискретизации изображения в K=M/N раз путем растяжения и последующего сжатия (в два прохода)
    def resample_image(image, M, N):
        stretched_image = stretch_image(image, M)
        resampled_image = compress_image(stretched_image, N)
        return resampled_image

    # Загрузка исходного изображения
    input_image = Image.open('input/linux.png')

    # Демонстрация исходного изображения
    input_image.show()

    # Выполнение операций передискретизации
    M = 2
    N = 3
    K = M / N

    resampled_image = resample_image(input_image, M, N)
    print("Resampled Image Size (Two Passes):", resampled_image.size)

    # Демонстрация результатов
    resampled_image.show()


if __name__ == "__main__": 
    main()