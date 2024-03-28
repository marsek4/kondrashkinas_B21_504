from PIL import Image

def main():

    # Функция для сжатия изображения в N раз по алгоритму ближайшего соседа
    def compress_image(image, N):
        width, height = image.size
        new_width = width // N
        new_height = height // N
        compressed_image = image.resize((new_width, new_height), Image.NEAREST)
        return compressed_image

    # Загрузка исходного изображения
    input_image = Image.open('input/linux.png')

    # Демонстрация исходного изображения
    input_image.show()

    # Выполнение операций передискретизации
    M = 2
    N = 3
    K = M / N

    compressed_image = compress_image(input_image, N)
    print("Compressed Image Size:", compressed_image.size)

    # Демонстрация результатов
    compressed_image.show()


if __name__ == "__main__": 
    main()