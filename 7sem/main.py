import csv
import math
import numpy as np
from PIL import Image

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

WHITE = 255
PHRASE = "Ⱑ ⱂⱁⰿⱀⱓ ⱍⱆⰴⱀⱁⰵ ⰿⰳⱀⱁⰲⰵⱀⱐⰵ".replace(" ", "")


def calculate_features(img: np.array):
    img_b = np.zeros(img.shape, dtype=int)
    img_b[img != WHITE] = 1  # предполагается, что значение белого пикселя равно 255

    # расчет веса
    weight = np.sum(img_b)

    # расчет центра масс
    y_indices, x_indices = np.indices(img_b.shape)
    y_center_of_mass = np.sum(y_indices * img_b) / weight
    x_center_of_mass = np.sum(x_indices * img_b) / weight

    # расчет инерции
    inertia_x = np.sum((y_indices - y_center_of_mass) ** 2 * img_b) / weight
    inertia_y = np.sum((x_indices - x_center_of_mass) ** 2 * img_b) / weight

    return weight, x_center_of_mass, y_center_of_mass, inertia_x, inertia_y


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


def get_alphabet_info() -> dict[chr, tuple]:
    def parse_tuple(string):
        return tuple(map(float, string.strip('()').split(',')))

    tuples_list = dict()
    with open('input/data.csv', 'r') as file:
        reader = csv.DictReader(file)
        i = 0
        for row in reader:
            weight = int(row['weight'])
            center_of_mass = parse_tuple(row['center_of_mass'])
            inertia = parse_tuple(row['inertia'])
            tuples_list[glagolitsa_letters[i]] = weight, *center_of_mass, *inertia
            i += 1
    return tuples_list


def create_hypothesis(alphabet_info: dict[chr, tuple], target_features):
    def euclidean_distance(feature1, feature2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(feature1, feature2)))

    distances = dict()
    for letter, features in alphabet_info.items():
        distance = euclidean_distance(target_features, features)
        distances[letter] = distance

    max_distance = max(distances.values())

    similarities = [(letter, round(1 - distance / max_distance, 2)) for letter, distance in distances.items()]

    return sorted(similarities, key=lambda x: x[1])


def get_phrase_from_hypothesis(img: np.array, bounds) -> str:
    alphabet_info = get_alphabet_info()
    res = []
    for start, end in bounds:
        letter_features = calculate_features(img[:, start: end])
        hypothesis = create_hypothesis(alphabet_info, letter_features)
        best_hypotheses = hypothesis[-1][0]
        res.append(best_hypotheses)
    return "".join(res)


def write_res(recognized_phrase: str):
    max_len = max(len(PHRASE), len(recognized_phrase))
    orig = PHRASE.ljust(max_len)
    detected = recognized_phrase.ljust(max_len)

    with open("output/result.txt", 'w', encoding='utf-8') as f:  # Указание кодировки utf-8
        correct_letters = 0
        by_letter = ["original | detected | is_correct"]
        for i in range(max_len):
            is_correct = orig[i] == detected[i]
            by_letter.append(f"{orig[i]}\t{detected[i]}\t{is_correct}")
            correct_letters += int(is_correct)
        f.write("\n".join([
            f"phrase:      {orig}",
            f"detected:    {detected}",
            f"correct:     {math.ceil(correct_letters / max_len * 100)}%\n\n"
        ]))
        f.write("\n".join(by_letter))



if __name__ == "__main__":
    img = np.array(Image.open(f'input/original_phrase copy.bmp').convert('L'))
    bounds = segment_letters(img)
    recognized_phrase = get_phrase_from_hypothesis(img, bounds)
    write_res(recognized_phrase)