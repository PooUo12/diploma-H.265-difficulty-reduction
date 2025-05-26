import math
import pickle
import sys
import time

import cv2
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass

from intra_coding import intra_predict, intra_predict_decode


@dataclass
class Ctu:
    pixels: np.array
    last_pixel_x: int
    last_pixel_y: int
    encoded: bool = False


@dataclass
class Area:
    last_pixel_x: int
    last_pixel_y: int
    ctus: list


ctu_arr = []
CTU_ACCURACY = 10


def frame_is_similar(first: np.array, second: np.array, accuracy: float) -> bool:
    if abs(first[0] - second[0]) >= (first[0] / accuracy):
        return False
    if abs(first[1] - second[1]) >= (first[1] / accuracy):
        return False
    if abs(first[2] - second[2]) >= (first[2] / accuracy):
        return False
    return True


def divide_to_ctu(cu: np.array, ctus: list, last_pixel_x: int, last_pixel_y: int) -> list:
    half_size = len(cu) // 2
    if (half_size * 2) == 16:
        ctus.append(Ctu(cu, last_pixel_x, last_pixel_y))
        return ctus
    cu_avg = get_cu_avg(cu)
    middle_pixel = cu[half_size, half_size]
    if (1 - min(cu_avg[0], middle_pixel[0]) / max(cu_avg[0], middle_pixel[0])) > CTU_ACCURACY / 100:
        divide_to_ctu(cu[0: half_size, 0: half_size], ctus, last_pixel_x - half_size, last_pixel_y - half_size)
        divide_to_ctu(cu[half_size:, 0: half_size], ctus, last_pixel_x - half_size, last_pixel_y)
        divide_to_ctu(cu[0: half_size, half_size:], ctus, last_pixel_x, last_pixel_y - half_size)
        divide_to_ctu(cu[half_size:, half_size:], ctus, last_pixel_x, last_pixel_y)
    else:
        ctus.append(Ctu(cu, last_pixel_x, last_pixel_y))
        return ctus


def get_image_avg(image_id: int) -> np.array:
    img = mpimg.imread('frame%d.jpg' % image_id)
    img = img.reshape(img.size // 3, 3)
    return np.mean(img, 0)


def get_cu_avg(ctu: np.array) -> np.array:
    ctu = ctu.reshape(ctu.size // 3, 3)
    return np.mean(ctu, 0)


def bin_frame_search(end: int) -> list[int]:
    checkpoints = []
    start = 0
    accuracy = 20
    while accuracy > 5:
        while not (frame_is_similar(get_image_avg(start), get_image_avg(end), 100 / accuracy)):
            current = (start + end) // 2 + 1
            left_pointer = start
            right_pointer = end
            while current != right_pointer:
                if frame_is_similar(get_image_avg(start), get_image_avg(current), 100 / accuracy):
                    left_pointer = current
                    current = (left_pointer + right_pointer) // 2 + 1
                else:
                    right_pointer = current
                    current = (left_pointer + right_pointer) // 2 + 1
            if frame_is_similar(get_image_avg(current), get_image_avg(current - 1), 100 / accuracy):
                current -= 1
            start = current
            checkpoints.append(current)
        print("Accuracy: ", 100 / accuracy)
        print(checkpoints)
        accuracy -= 0.5
    return checkpoints


def divide_to_frames(name: str) -> int:
    vidcap = cv2.VideoCapture(name)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
    return count


def divide_to_cu(frame: int, ctus: list,
                 areas: list, bottom_areas: list) -> None:
    img = mpimg.imread('frame%d.jpg' % frame)
    for i in range(img.shape[0] // 64):
        for j in range(img.shape[1] // 64):
            ctus_temp = []
            divide_to_ctu(img[i * 64:i * 64 + 64, j * 64:j * 64 + 64], ctus_temp, j * 64 + 64, i * 64 + 64)
            areas.append(Area(i * 64 + 64, j * 64 + 64, [i for i in range(len(ctus), len(ctus) + len(ctus_temp))]))
            ctus.extend(ctus_temp)
    if (img.shape[0] % 64) != 0:
        for j in range(img.shape[1] // 16):
            ctus_temp = []
            last_pixels = img.shape[0] - img.shape[0] % 64
            divide_to_ctu(img[last_pixels: img.shape[0], j * 16:j * 16 + 16], ctus_temp, j * 16 + 16,
                          img.shape[0])
            bottom_areas.append(
                Area(j * 16 + 16, img.shape[0], [i for i in range(len(ctus), len(ctus) + len(ctus_temp))]))
            ctus.extend(ctus_temp)


def find_area(x_val: int, y_val: int, areas: list, bottom_areas: list) -> Area:
    if (x_val >= 1280) or (y_val >= 720):
        return Area(-1, -1, [])
    if y_val >= 704:
        return bottom_areas[x_val // 16]
    return areas[(y_val // 64) * 20 + (x_val // 64)]


def find_ref_pixels(pixels_x_first_index: int,
                    pixels_y_first_index: int,
                    ctus: list, areas: list, length: int,
                    bottom_areas: list) -> tuple[list, list]:
    pixels_y = [None for i in range(length * 2 + 1)]
    pixels_x = [None for i in range(length * 2 + 1)]
    index = 0

    if pixels_x_first_index != -1:
        if pixels_y_first_index == -1:
            index += 1
            pixels_y_first_index += 1
        if ((pixels_y_first_index // 64) == ((pixels_y_first_index + length * 2) // 64)) or (pixels_y_first_index == 0):
            area = find_area(pixels_x_first_index, pixels_y_first_index, areas, bottom_areas)
            extract_pixels_y(area, ctus, index, length, pixels_x_first_index, pixels_y, pixels_y_first_index)
        else:
            area_1 = find_area(pixels_x_first_index, pixels_y_first_index, areas, bottom_areas)
            index = extract_pixels_y(area_1, ctus, index, length, pixels_x_first_index, pixels_y, pixels_y_first_index)
            if pixels_y_first_index != 703:
                area_2 = find_area(pixels_x_first_index, pixels_y_first_index + 64, areas, bottom_areas)
                extract_pixels_y(area_2, ctus, index, length, pixels_x_first_index, pixels_y, pixels_y_first_index)
    index = 0
    if pixels_y_first_index == 0:
        pixels_y_first_index -= 1
    if pixels_y_first_index != -1:
        if pixels_x_first_index == -1:
            index += 1
            pixels_x_first_index += 1
        if ((pixels_x_first_index // 64) == ((pixels_x_first_index + length * 2) // 64)) or (pixels_x_first_index == 0):
            area = find_area(pixels_x_first_index, pixels_y_first_index, areas, bottom_areas)
            extract_pixels_x(area, ctus, index, length, pixels_x_first_index, pixels_x, pixels_y_first_index)
        else:
            area_1 = find_area(pixels_x_first_index, pixels_y_first_index, areas, bottom_areas)
            index = extract_pixels_x(area_1, ctus, index, length, pixels_x_first_index, pixels_x, pixels_y_first_index)
            if pixels_y_first_index == 703:
                pixels_x_first_index += 16
            else:
                pixels_x_first_index += 64
            area_2 = find_area(pixels_x_first_index, pixels_y_first_index, areas, bottom_areas)
            extract_pixels_x(area_2, ctus, index, length, pixels_x_first_index, pixels_x, pixels_y_first_index)
    return pixels_x, pixels_y


def extract_pixels_y(area: Area, ctus: list, index: int, length: int, pixels_x_first_index: int, pixels_y: list,
                     pixels_y_first_index: int) -> int:
    pixels_y_index = pixels_y_first_index
    for ctu_index in area.ctus:
        ctu = ctus[ctu_index]
        if ((ctu.last_pixel_x >= pixels_x_first_index)
                and ((ctu.last_pixel_x - len(ctu.pixels)) <= pixels_x_first_index)
                and (ctu.last_pixel_y >= pixels_y_index)
                and ((ctu.last_pixel_y - len(ctu.pixels)) <= pixels_y_index)):
            in_ctu_index_x = pixels_x_first_index - (ctu.last_pixel_x - len(ctu.pixels))
            in_ctu_index_y = pixels_y_index - (ctu.last_pixel_y - len(ctu.pixels))
            in_ctu_index_y_length = len(ctu.pixels) - in_ctu_index_y
            last_index = min(in_ctu_index_y_length + index, (length * 2 + 1))
            if not ctu.encoded:
                break
            if last_index == (in_ctu_index_y_length + index):
                if (index == 0) and (last_index == 1):
                    pixels_y[0] = ctu.pixels[in_ctu_index_y, in_ctu_index_x]
                else:
                    pixels_y[index: last_index] = \
                        ctu.pixels[in_ctu_index_y:, in_ctu_index_x]
                index = last_index
                pixels_y_index += last_index
                if index == (length * 2):
                    break
            else:
                pixels_y[index: last_index] = \
                    ctu.pixels[in_ctu_index_y: in_ctu_index_y + last_index - index, in_ctu_index_x]
                return index
    return index


def extract_pixels_x(area: Area, ctus: list, index: int, length: int, pixels_x_first_index: int, pixels_x: list,
                     pixels_y_first_index: int) -> int:
    pixels_x_index = pixels_x_first_index
    for ctu_index in area.ctus:
        ctu = ctus[ctu_index]
        if ((ctu.last_pixel_x >= pixels_x_index)
                and ((ctu.last_pixel_x - len(ctu.pixels)) <= pixels_x_index)
                and (ctu.last_pixel_y >= pixels_y_first_index)
                and ((ctu.last_pixel_y - len(ctu.pixels)) <= pixels_y_first_index)):
            in_ctu_index_x = pixels_x_index - (ctu.last_pixel_x - len(ctu.pixels))
            in_ctu_index_y = pixels_y_first_index - (ctu.last_pixel_y - len(ctu.pixels))
            in_ctu_index_x_length = len(ctu.pixels) - in_ctu_index_x
            last_index = min(in_ctu_index_x_length + index, (length * 2 + 1))
            if not ctu.encoded:
                break
            if last_index == (in_ctu_index_x_length + index):
                if (index == 0) and (last_index == 1):
                    pixels_x[0] = ctu.pixels[in_ctu_index_y, in_ctu_index_x]
                else:
                    pixels_x[index: last_index] = \
                        ctu.pixels[in_ctu_index_y, in_ctu_index_x:]
                index += last_index
                pixels_x_index += last_index
                if index == (length * 2):
                    break
            else:
                pixels_x[index: last_index] = \
                    ctu.pixels[in_ctu_index_y, in_ctu_index_x: in_ctu_index_x + last_index - index]
                return index
    return index


def intra_coding(ctus: list, areas: list, bottom_areas: list, ctu: Ctu) -> tuple[list, int, list, int, int]:
    length = len(ctu.pixels)
    pixels_x_first_index = ctu.last_pixel_x - length - 1
    pixels_y_first_index = ctu.last_pixel_y - length - 1
    pixels_x, pixels_y = find_ref_pixels(pixels_x_first_index, pixels_y_first_index, ctus, areas, length, bottom_areas)
    predicted_block, error_list, mode = intra_predict(pixels_x, pixels_y, ctu.pixels)
    ctu.encoded = True
    return predicted_block, mode, error_list, ctu.last_pixel_x, ctu.last_pixel_y


def find_area_to_ctu(ctu: Ctu, areas: list, bottom_areas: list, index: int, x_size: int, y_size: int) -> None:
    if ctu.last_pixel_y == y_size:
        bottom_areas[(ctu.last_pixel_x - 1) // 16].ctus.append(index)
        return
    areas[((ctu.last_pixel_y - 1) // 64) * 20 + ((ctu.last_pixel_x - 1) // 64)].ctus.append(index)


def unite_ctus(ctus: list, x_size: int, y_size: int) -> np.array:
    frame = np.full((y_size, x_size, 3), 0)
    for ctu in ctus:
        length = len(ctu.pixels)
        top_left_y = ctu.last_pixel_y - length
        top_left_x = ctu.last_pixel_x - length
        frame[top_left_y: ctu.last_pixel_y, top_left_x: ctu.last_pixel_x] = ctu.pixels
    return frame


def intra_decoding_main(ctu_info_filename: str,
                        error_lists_bin_filename: str,
                        x_size: int, y_size: int) -> None:
    areas = []
    bottom_areas = []
    ctus = []
    for i in range(y_size // 64):
        for j in range(x_size // 64):
            areas.append(Area(j * 64 + 64, i * 64 + 64, []))
    if (y_size % 64) != 0:
        for j in range(x_size // 16):
            bottom_areas.append(
                Area(j * 16 + 16, y_size, []))
    with open(error_lists_bin_filename, "rb") as f:
        error_lists = pickle.load(f)
    with open(ctu_info_filename, "r") as f:
        data = f.read().splitlines()
        for i in range(len(data) - 1):
            ctu_info = data[i + 1].split(" ")
            error_list = error_lists[i].astype(np.int64)
            length = len(error_list)
            last_pixel_x = int(ctu_info[1])
            last_pixel_y = int(ctu_info[2])
            mode = int(ctu_info[0])
            pixels_x_first_index = last_pixel_x - length - 1
            pixels_y_first_index = last_pixel_y - length - 1
            pixels_x, pixels_y = find_ref_pixels(pixels_x_first_index, pixels_y_first_index, ctus, areas, length,
                                                     bottom_areas)
            predicted_block = intra_predict_decode(pixels_x, pixels_y, mode)
            real_block = (predicted_block + error_list).astype(int)
            ctus.append(Ctu(real_block, last_pixel_x, last_pixel_y, True))
            find_area_to_ctu(ctus[i], areas, bottom_areas, i, x_size, y_size)
    frame = unite_ctus(ctus, x_size, y_size, areas)
    plt.imshow(frame)
    plt.show()


def intra_coding_main(frame: int) -> None:
    ctus = []
    areas = []
    bottom_areas = []
    divide_to_cu(frame, ctus, areas, bottom_areas)

    with open("byte_array.txt", "w") as f:
        f.write("Byte array imitation \n")

    all_error_lists = []
    with open("byte_array.txt", "a") as f:
        for i in tqdm(range(len(ctus))):
            predicted_block, mode, error_list, last_pixel_x, last_pixel_y = intra_coding(ctus, areas, bottom_areas,
                                                                                         ctus[i])
            f.write(str(mode) + " " + str(last_pixel_x) + " " + str(last_pixel_y) + "\n")
            all_error_lists.append(error_list)
    with open("errors.bin", "wb") as f:
        pickle.dump(all_error_lists, f)


def find_same_pixels(ctus_old: list,
                     areas_old: list,
                     bottom_areas_old: list,
                     ctu_new: Ctu) -> np.array:
    ctu_area = find_area(ctu_new.last_pixel_x - 1, ctu_new.last_pixel_y - 1, areas_old, bottom_areas_old)
    unite_ctu_pixels = np.full((len(ctu_new.pixels), len(ctu_new.pixels), 3), 0)
    for ctu_i in ctu_area.ctus:
        ctu = ctus_old[ctu_i]
        if len(ctu.pixels) == len(ctu_new.pixels):
            if (ctu.last_pixel_x == ctu_new.last_pixel_x) and (ctu.last_pixel_y == ctu_new.last_pixel_y):
                return ctu.pixels
        elif len(ctu.pixels) > len(ctu_new.pixels):
            if ((ctu.last_pixel_x >= ctu_new.last_pixel_x)
                    and ((ctu.last_pixel_x - len(ctu.pixels)) <= ctu_new.last_pixel_x)
                    and (ctu.last_pixel_y >= ctu_new.last_pixel_y)
                    and ((ctu.last_pixel_y - len(ctu.pixels)) <= ctu_new.last_pixel_y)):
                last_index_x = ctu_new.last_pixel_x - ctu.last_pixel_x + len(ctu.pixels)
                last_index_y = ctu_new.last_pixel_y - ctu.last_pixel_y + len(ctu.pixels)
                return ctu.pixels[(last_index_y - len(ctu_new.pixels)): last_index_y, (last_index_x - len(ctu_new.pixels)): last_index_x]
        else:
            if ((ctu_new.last_pixel_x >= ctu.last_pixel_x)
                    and ((ctu_new.last_pixel_x - len(ctu_new.pixels)) < ctu.last_pixel_x)
                    and (ctu_new.last_pixel_y >= ctu.last_pixel_y)
                    and ((ctu_new.last_pixel_y - len(ctu_new.pixels)) < ctu.last_pixel_y)):
                last_index_x = ctu.last_pixel_x - ctu_new.last_pixel_x + len(ctu_new.pixels)
                last_index_y = ctu.last_pixel_y - ctu_new.last_pixel_y + len(ctu_new.pixels)
                unite_ctu_pixels[(last_index_y - len(ctu.pixels)): last_index_y, (last_index_x - len(ctu.pixels)): last_index_x] = ctu.pixels
    return unite_ctu_pixels


def find_motion_vector(pixels: np.array,
                       ref_pixels: np.array) \
        -> tuple[int, int]:
    length = len(pixels)
    if length != 16:
        return None, None
    for i in range(length - 1):
        for j in range(length - 1):
            check_low = (ref_pixels[length // 2 - 1: length // 2 + 1, length // 2 - 1: length // 2 + 1] <= (pixels[i: i + 2, j: j + 2] + 5))
            check_high = (ref_pixels[length // 2 - 1: length // 2 + 1, length // 2 - 1: length // 2 + 1] >= (pixels[i: i + 2, j: j + 2] - 5))
            if check_low.all() and check_high.all():
                mv_x = j - length // 2 - 1
                mv_y = i - length // 2 - 1
                return mv_x, mv_y
    return None, None


def expand_ctu(ctu_new: Ctu, ctus: list, areas: list, bottom_areas: list) -> np.array:
    length = len(ctu_new.pixels)
    expanded_pixels = np.full((length * 3, length * 3, 3), 127)
    ctu_area = find_area(ctu_new.last_pixel_x - 1, ctu_new.last_pixel_y - 1, areas, bottom_areas)
    combine_pixels_from_area(ctu_area, ctu_new, ctus, expanded_pixels, length)
    flag = 0
    if ((ctu_new.last_pixel_x + length) // 64) != (ctu_new.last_pixel_x // 64):
        ctu_area = find_area(ctu_new.last_pixel_x + length, ctu_new.last_pixel_y - 1, areas, bottom_areas)
        combine_pixels_from_area(ctu_area, ctu_new, ctus, expanded_pixels, length)
        flag += 1
    if ((ctu_new.last_pixel_y + length) // 64) != (ctu_new.last_pixel_y // 64):
        ctu_area = find_area(ctu_new.last_pixel_x -1, ctu_new.last_pixel_y + length - 1, areas, bottom_areas)
        combine_pixels_from_area(ctu_area, ctu_new, ctus, expanded_pixels, length)
        flag += 1
    if flag == 2:
        ctu_area = find_area(ctu_new.last_pixel_x + length - 1, ctu_new.last_pixel_y + length - 1, areas, bottom_areas)
        combine_pixels_from_area(ctu_area, ctu_new, ctus, expanded_pixels, length)
    return expanded_pixels


def combine_pixels_from_area(ctu_area, ctu_new, ctus, expanded_pixels, length):
    for ctu_i in ctu_area.ctus:
        ctu = ctus[ctu_i]
        if ((ctu.last_pixel_x > (ctu_new.last_pixel_x - 2 * length)) and
                (ctu.last_pixel_x <= (ctu_new.last_pixel_x + length)) and
                (ctu.last_pixel_y > (ctu_new.last_pixel_y - 2 * length)) and
                (ctu.last_pixel_y <= (ctu_new.last_pixel_y + length))):
            last_index_x = ctu.last_pixel_x - ctu_new.last_pixel_x + length * 2
            last_index_y = ctu.last_pixel_y - ctu_new.last_pixel_y + length * 2

            if len(ctu.pixels) == 16:
                expanded_pixels[(last_index_y - len(ctu_new.pixels)): last_index_y, (last_index_x - len(ctu_new.pixels)): last_index_x] = ctu.pixels
            else:
                expanded_pixels[(last_index_y - len(ctu_new.pixels)): last_index_y, (last_index_x - len(ctu_new.pixels)): last_index_x] = ctu.pixels[len(ctu.pixels) //2 :, len(ctu.pixels) //2:]


def inter_coding_main(frame_old: int,
                      frame_new: int) -> None:
    ctus_old = []
    areas_old = []
    bottom_areas_old = []
    divide_to_cu(frame_old, ctus_old, areas_old, bottom_areas_old)

    ctus_new = []
    areas_new = []
    bottom_areas_new = []
    divide_to_cu(frame_new, ctus_new, areas_new, bottom_areas_new)

    all_error_lists = []
    with open("byte_array_ctu_skeleton.txt", "w") as f:
        for ctu in ctus_new:
            f.write(str(ctu.last_pixel_x) + " " + str(ctu.last_pixel_y) + " " + str(len(ctu.pixels)) + "\n")
    with open("byte_array_intra.txt", "w") as f:
        for ctu in ctus_new:
            ref_pixels = find_same_pixels(ctus_old, areas_old, bottom_areas_old, ctu)
            length = len(ref_pixels)
            errs = np.equal(ref_pixels, ctu.pixels).sum()
            if errs <= (length - 2):
                mv_x, mv_y = find_motion_vector(ctu.pixels, ref_pixels)
                if mv_x is None:
                    errors = np.int64(ctu.pixels.astype(int) - ref_pixels.astype(int))
                    all_error_lists.append(errors)
                    f.write(str(ctu.last_pixel_x) + " " + str(ctu.last_pixel_y) + " None None\n")
                else:
                    f.write(str(ctu.last_pixel_x) + " " + str(ctu.last_pixel_y) + " " + str(mv_x) + " " + str(mv_y) + "\n")
                    expanded_pixels = expand_ctu(ctu, ctus_old, areas_old, bottom_areas_old)
                    expanded_pixels[length: (length * 2), length: (length * 2)] = ref_pixels
                    predicted_pixels = expanded_pixels[(length + mv_y): (length * 2 + mv_y), (length + mv_x): (length * 2 + mv_x)]
                    errors = np.int64(ctu.pixels.astype(int) - predicted_pixels.astype(int))
                    all_error_lists.append(errors)
    with open("errors_intra.bin", "wb") as f:
        pickle.dump(all_error_lists, f)


def inter_decoding_main() -> None:
    x_size = 1280
    y_size = 720

    ctus = []
    areas = []
    bottom_areas = []
    for i in range(y_size // 64):
        for j in range(x_size // 64):
            areas.append(Area(j * 64 + 64, i * 64 + 64, []))
    if (y_size % 64) != 0:
        for j in range(x_size // 16):
            bottom_areas.append(
                Area(j * 16 + 16, y_size, []))
    ctus_old = []
    areas_old = []
    bottom_areas_old = []
    divide_to_cu(0, ctus_old, areas_old, bottom_areas_old)
    with open("byte_array_ctu_skeleton.txt", "r") as f:
        data = f.read().splitlines()
        for i in range(len(data)):
            ctu_info = data[i].split(" ")
            last_pixel_x = int(ctu_info[0])
            last_pixel_y = int(ctu_info[1])
            length = int(ctu_info[2])
            ctus.append(Ctu(np.full((length, length, 3), 0), last_pixel_x, last_pixel_y))
            ctus[i].pixels = find_same_pixels(ctus_old, areas_old, bottom_areas_old, ctus[i])
            find_area_to_ctu(ctus[i], areas, bottom_areas, i, x_size, y_size)
    with open("errors_intra.bin", "rb") as f:
        error_lists = pickle.load(f)
    with open("byte_array_intra.txt", "r") as f:
        data = f.read().splitlines()
        for i in range(len(data)):
            cur_ctu = -1
            ctu_info = data[i].split(" ")
            error_list = error_lists[i].astype(np.int64)
            last_pixel_x = int(ctu_info[0])
            last_pixel_y = int(ctu_info[1])
            mv_x = ctu_info[2]
            mv_y = ctu_info[3]
            area = find_area(last_pixel_x - 1, last_pixel_y - 1, areas, bottom_areas)
            for ctu_i in area.ctus:
                ctu = ctus[ctu_i]
                if (ctu.last_pixel_x == last_pixel_x) and (ctu.last_pixel_y == last_pixel_y):
                    cur_ctu = ctu
            if mv_x == "None":
                new_pixels = cur_ctu.pixels + error_list
                cur_ctu.pixels = new_pixels
            else:
                mv_x = int(mv_x)
                mv_y = int(mv_y)
                expanded_pixels = expand_ctu(cur_ctu, ctus_old, areas_old, bottom_areas_old)
                expanded_pixels[length: (length * 2), length: (length * 2)] = cur_ctu.pixels
                predicted_pixels = expanded_pixels[(length + mv_y): (length * 2 + mv_y), (length + mv_x): (length * 2 + mv_x)]
                new_pixels = predicted_pixels + error_list
                cur_ctu.pixels = new_pixels
    frame = unite_ctus(ctus, x_size, y_size)
    plt.imshow(frame)
    plt.show()

start_time = time.time()
inter_coding_main(1,4)
end_time = time.time()
execution_time = end_time - start_time
print(str(execution_time) + "c")
# intra_decoding_main("byte_array.txt", "errors.bin", 1280, 720)
# count = divide_to_frames('SampleVideo_1280x720_5mb.mp4')
# bin_frame_search(738) #324
