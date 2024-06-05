import csv
import os
from math import ceil

import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
from matplotlib import pyplot as plt

glagolitsa_letters_unicode = [
    '2C00', '2C01', '2C02', '2C03', '2C04', '2C05', '2C06', '2C07',
    '2C08', '2C09', '2C0A', '2C0B', '2C0C', '2C0D', '2C0E', '2C0F',
    '2C10', '2C11', '2C12', '2C13', '2C14', '2C15', '2C16', '2C17',
    '2C18', '2C19', '2C1A', '2C1B', '2C1C', '2C1D', '2C1E', '2C1F',
    '2C20', '2C21', '2C22', '2C23', '2C24', '2C25', '2C26', '2C27',
    '2C28', '2C29', '2C2A', '2C2B', '2C2C', '2C2D', '2C2E', '2C2F',
    '2C30', '2C31', '2C32', '2C33', '2C34', '2C35', '2C36', '2C37',
    '2C38', '2C39', '2C3A', '2C3B', '2C3C', '2C3D', '2C3E', '2C3F',
    '2C40', '2C41', '2C42', '2C43', '2C44', '2C45', '2C46', '2C47',
    '2C48', '2C49', '2C4A', '2C4B', '2C4C', '2C4D', '2C4E', '2C4F',
    '2C50', '2C51', '2C52', '2C53', '2C54', '2C55', '2C56', '2C57',
    '2C58', '2C59', '2C5A', '2C5B', '2C5C', '2C5D', '2C5E', '2C5F'
]

glagolitsa_letters = [chr(int(letter, 16)) for letter in glagolitsa_letters_unicode]

unit_size = 54
thresold_ = 75
ttf_file = "input/Quivira.ttf"

WHITE = 255


def _simple_binarization(img, threshold=thresold_):
    semitoned = (0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 2]).astype(np.uint8)
    new_image = np.zeros(shape=semitoned.shape)
    new_image[semitoned > threshold] = WHITE
    return new_image.astype(np.uint8)


def generate_letters(sin_letters):

    font = ImageFont.truetype(ttf_file, unit_size)
    os.makedirs("output/letters", exist_ok=True)
    os.makedirs("output/inverse", exist_ok=True)

    for i in range(len(sin_letters)):
        letter = sin_letters[i]

        width, height = font.getsize(letter)
        #bbox = font.getbbox(letter)
        #width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        img = Image.new(mode="RGB", size=(ceil(width), ceil(height)), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), letter, "black", font=font)

        img = Image.fromarray(_simple_binarization(np.array(img), thresold_), 'L')
        img.save(f"output/letters/{i + 1}.png")

        ImageOps.invert(img).save(f"output/inverse/{i + 1}.png")

        # plt.imshow(img, cmap='gray')
        # plt.show()



def calculate_features(img):
    img_b = np.zeros(img.shape, dtype=int)
    img_b[img != WHITE] = 1  # предполагается, что значение белого пикселя равно 255

    # рассчет веса квадрантов и относительные веса

    (h, w) = img_b.shape
    h_half, w_half = h // 2, w // 2
    quadrants = {
        'top_left': img_b[:h_half, :w_half],
        'top_right': img_b[:h_half, w_half:],
        'bottom_left': img_b[h_half:, :w_half],
        'bottom_right': img_b[h_half:, w_half:]
    }
    weights = {k: np.sum(v) for k, v in quadrants.items()}
    rel_weights = {k: v / (h_half * w_half) for k, v in weights.items()}

    # рассчет центра масс
    total_pixels = np.sum(img_b) 
    y_indices, x_indices = np.indices(img_b.shape)
    y_center_of_mass = np.sum(y_indices * img_b) / total_pixels
    x_center_of_mass = np.sum(x_indices * img_b) / total_pixels
    center_of_mass = (x_center_of_mass, y_center_of_mass)

    # рассчет нормированного центра масс
    normalized_center_of_mass = (x_center_of_mass / (w - 1), y_center_of_mass / (h - 1))

    # рассчет инерции
    inertia_x = np.sum((y_indices - y_center_of_mass) ** 2 * img_b) / total_pixels
    normalized_inertia_x = inertia_x / h ** 2
    inertia_y = np.sum((x_indices - x_center_of_mass) ** 2 * img_b) / total_pixels
    normalized_inertia_y = inertia_y / w ** 2

    return {
        'weight': total_pixels,
        'weights': weights,
        'rel_weights': rel_weights,
        'center_of_mass': center_of_mass,
        'normalized_center_of_mass': normalized_center_of_mass,
        'inertia': (inertia_x, inertia_y),
        'normalized_inertia': (normalized_inertia_x, normalized_inertia_y)
    }


def create_features(sin_letters):
    with open('output/data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['weight', 'weights', 'rel_weights', 'center_of_mass', 'normalized_center_of_mass',
                      'inertia', 'normalized_inertia']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(sin_letters)):
            img_src = np.array(Image.open(f'output/letters/{i + 1}.png').convert('L'))
            features = calculate_features(img_src)
            writer.writerow(features)


def create_profiles(sin_letters):
    os.makedirs("output/profiles/x", exist_ok=True)
    os.makedirs("output/profiles/y", exist_ok=True)

    for i in range(len(sin_letters)):
        img = np.array(Image.open(f'output/letters/{i + 1}.png').convert('L'))
        img_b = np.zeros(img.shape, dtype=int)
        img_b[img != WHITE] = 1  # предполагается, что значение белого пикселя равно 255

        plt.bar(
            x=np.arange(start=1, stop=img_b.shape[1] + 1).astype(int),
            height=np.sum(img_b, axis=0),
            width=0.9
        )
        plt.ylim(0, unit_size)
        plt.xlim(0, 55)
        plt.savefig(f'output/profiles/x/{i + 1}.png')
        plt.clf()

        plt.barh(
            y=np.arange(start=1, stop=img_b.shape[0] + 1).astype(int),
            width=np.sum(img_b, axis=1),
            height=0.9
        )
        plt.ylim(unit_size, 0)
        plt.xlim(0, 55)
        plt.savefig(f'output/profiles/y/{i + 1}.png')
        plt.clf()


generate_letters(glagolitsa_letters)
create_features(glagolitsa_letters)
create_profiles(glagolitsa_letters)