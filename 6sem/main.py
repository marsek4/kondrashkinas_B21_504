import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageOps

phrase = "Ⱑ ⱂⱁⰿⱀⱓ ⱍⱆⰴⱀⱁⰵ ⰿⰳⱀⱁⰲⰵⱀⱐⰵ"
white = 255

font_ = ImageFont.truetype("input/Quivira.ttf", 52)
thresold_ = 75


def create_phrase_profiles(img: np.array):
    os.makedirs("output/phrase_profile", exist_ok=True)
    img_b = np.zeros(img.shape, dtype=int)
    img_b[img != white] = 1  # предполагается, что значение белого пикселя равно 255

    plt.bar(
        x=np.arange(start=1, stop=img_b.shape[1] + 1).astype(int),
        height=np.sum(img_b, axis=0),
        width=0.9
    )
    plt.savefig(f'output/phrase_profile/x.png')
    plt.clf()

    plt.barh(
        y=np.arange(start=1, stop=img_b.shape[0] + 1).astype(int),
        width=np.sum(img_b, axis=1),
        height=0.9
    )
    plt.savefig(f'output/phrase_profile/y.png')
    plt.clf()


def _simple_binarization(img, threshold=thresold_):
    new_image = np.zeros(shape=img.shape)
    new_image[img > threshold] = white
    return new_image.astype(np.uint8)


def generate_phrase_image():
    space_len = 5
    phrase_width = sum(font_.getsize(char)[0] for char in phrase) + space_len * (len(phrase) - 1)

    height = max(font_.getsize(char)[1] for char in phrase)

    img = Image.new("L", (phrase_width, height), color="white")
    draw = ImageDraw.Draw(img)

    current_x = 0
    for letter in phrase:
        width, letter_height = font_.getsize(letter)
        draw.text((current_x, height - letter_height), letter, "black", font=font_)
        current_x += width + space_len

    img = Image.fromarray(_simple_binarization(np.array(img)))
    img.save("output/original_phrase.bmp")

    np_img = np.array(img)
    create_phrase_profiles(np_img)
    ImageOps.invert(img).save("output/inverted_phrase.bmp")
    return np_img


def segment_letters(img):

    profile = np.sum(img == 0, axis=0)

    in_letter = False
    letter_bounds = []

    for i in range(len(profile)):
        if profile[i] > 0:
            if not in_letter:
                in_letter = True
                start = i
        else:
            if in_letter:
                in_letter = False
                end = i
                letter_bounds.append((start - 1, end))

    if in_letter:
        letter_bounds.append((start, len(profile)))

    return letter_bounds


def draw_bounding_boxes(img, bounds):

    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)

    for start, end in bounds:
        left, right = start, end
        top, bottom = 0, img.shape[0]
        draw.rectangle([left, top, right, bottom], outline="red")

    image.save("output/segmented_phrase.bmp")


if __name__ == "__main__":

    img = generate_phrase_image()
    bounds = segment_letters(img)
    draw_bounding_boxes(img, bounds)