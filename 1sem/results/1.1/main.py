from PIL import Image

def main():

    # Функция для растяжения изображения в M раз по алгоритму ближайшего соседа
    def stretch_image(image, M):
        width, height = image.size
        new_width = width * M
        new_height = height * M
        stretched_image = image.resize((new_width, new_height), Image.NEAREST)
        return stretched_image

    # Загрузка исходного изображения
    input_image = Image.open('input/linux.png')

    # Демонстрация исходного изображения
    input_image.show()

    # Выполнение операций передискретизации
    M = 2
    N = 3
    K = M / N

    stretched_image = stretch_image(input_image, M)
    print("Stretched Image Size:", stretched_image.size)

    # Демонстрация результатов
    stretched_image.show()


if __name__ == "__main__": 
    main()