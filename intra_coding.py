import math
from typing import Tuple

import numpy as np

ANGULAR_A = [32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13, -9, -5, -2, 0, 2,
             5, 9, 13, 17, 21, 26, 32]
angular_a_to_b_dict = {
    -32: -256,
    -26: -315,
    -21: -390,
    -17: -482,
    -13: -630,
    -9: -910,
    -5: -1638,
    -2: -4096
}


def substitution(pixels_x: list, pixels_y: list) -> tuple[list, list]:

    length = len(pixels_x)
    if (pixels_y[0] is None) and (pixels_y[1] is None) and (pixels_x[1] is None):
        for i in range(length):
            pixels_x[i] = np.array([127, 127, 127])
            pixels_y[i] = np.array([127, 127, 127])
        return pixels_x, pixels_y
    if (pixels_y[0] is None) and (pixels_y[1] is None):
        pixels_y[0] = pixels_x[1]
        pixels_x[0] = pixels_x[1]
    elif pixels_y[0] is None:
        pixels_y[0] = pixels_y[1]
        pixels_x[0] = pixels_y[1]

    unavailable_pixels_y = length
    unavailable_pixels_x = length
    for i in range(1, length):
        if pixels_y[i] is None:
            unavailable_pixels_y = i
            break
    for i in range(1, length):
        if pixels_x[i] is None:
            unavailable_pixels_x = i
            break
    if unavailable_pixels_y != length:
        for i in range(unavailable_pixels_y, length):
            pixels_y[i] = pixels_y[unavailable_pixels_y - 1]
    if unavailable_pixels_x != length:
        for i in range(unavailable_pixels_x, length):
            pixels_x[i] = pixels_x[unavailable_pixels_x - 1]
    return pixels_x, pixels_y


def filtering(pixels_x: list, pixels_y: list) -> tuple[list, list]:
    length = len(pixels_x)
    pixels_x[0] = (2 * pixels_x[0] + pixels_x[1] + pixels_y[1]) / 4
    pixels_y[0] = pixels_x[0]
    for i in range(1, length - 1):
        pixels_y[i] = (2 * pixels_y[i] + pixels_y[i - 1] + pixels_y[i + 1]) / 4
        pixels_x[i] = (2 * pixels_x[i] + pixels_x[i - 1] + pixels_x[i + 1]) / 4
    return pixels_x, pixels_y


def dc_prediction(pixels_x: list, pixels_y: list) -> np.array:
    sum_x = 0
    sum_y = 0
    block_length = len(pixels_x) // 2
    predicted_block = [[np.full(3, 0) for x in range(block_length)] for y in range(block_length)]
    for i in range(1, block_length + 1):
        sum_x += pixels_x[i]
        sum_y += pixels_y[i]
    dc_val = (sum_x + sum_y + block_length) >> (math.floor(math.log(block_length, 2)) + 1)
    for i in range(block_length):
        for j in range(block_length):
            if (i == 0) and (j == 0):
                predicted_block[i][j] = np.reshape((pixels_x[1] + pixels_y[1] + 2 * dc_val) >> 2, -1)
                continue
            if i == 0:
                predicted_block[i][j] = np.reshape((pixels_x[j + 1] + 3 * dc_val + 2) >> 2, -1)
                continue
            if j == 0:
                predicted_block[i][j] = np.reshape((pixels_y[i + 1] + 3 * dc_val + 2) >> 2, -1)
                continue
            predicted_block[i][j] = np.full(3, dc_val)
    return np.array(predicted_block)


def planar_prediction(pixels_x: list, pixels_y: list) -> np.array:
    block_length = len(pixels_x) // 2
    predicted_block = [[np.full(3, 0) for x in range(block_length)] for y in range(block_length)]
    for i in range(block_length):
        for j in range(block_length):
            hor = (block_length - 1 - i) * pixels_y[j + 1] + (i + 1) * pixels_x[block_length + 1]
            ver = (block_length - 1 - j) * pixels_x[i + 1] + (j + 1) * pixels_y[block_length + 1]
            predicted_block[i][j] = np.reshape((hor + ver + block_length) >> (math.floor(math.log(block_length, 2)) + 1), -1)
    return np.array(predicted_block)


def find_r_x(mode: int, x: int, pixels_x: list, pixels_y: list) -> np.array:
    a = ANGULAR_A[mode - 2]
    if x < 0:
        b = angular_a_to_b_dict[a]
        if mode < 18:
            r_x = pixels_x[((x * b + 128) >> 8)]
        else:
            r_x = pixels_y[((x * b + 128) >> 8)]
    else:
        if mode < 18:
            r_x = pixels_y[x]
        else:
            r_x = pixels_x[x]
    return r_x


def angular_prediction(pixels_x: list, pixels_y: list, mode: int) -> np.array:
    block_length = len(pixels_x) // 2
    predicted_block = [[np.full(3, 0) for x in range(block_length)] for y in range(block_length)]
    for i in range(block_length):
        for j in range(block_length):
            if mode < 18:
                ang_i = ((j + 1) * ANGULAR_A[mode - 2]) >> 5
                ang_f = ((j + 1) * ANGULAR_A[mode - 2]) & 31
                if (mode == 10) and (i == 0):
                    predicted_block[i][j] = np.reshape(np.clip((pixels_y[i] + (pixels_x[j] - pixels_x[0]) >> 1), 0, 255), -1)
                    continue
                if ang_f == 0:
                    predicted_block[i][j] = np.reshape(find_r_x(mode, (i + ang_i + 1), pixels_x, pixels_y), -1)
                else:
                    block = (((32 - ang_f) * find_r_x(mode, (i + ang_i + 1), pixels_x, pixels_y) +
                         ang_f * find_r_x(mode, (i + ang_i + 2), pixels_x,
                                          pixels_y) + 16) / 32).astype(float)
                    predicted_block[i][j] = np.reshape(np.round(block).astype(int), -1)
            else:
                ang_i = ((i + 1) * ANGULAR_A[mode - 2]) >> 5
                ang_f = ((i + 1) * ANGULAR_A[mode - 2]) & 31
                if (mode == 26) and (j == 0):
                    predicted_block[i][j] = np.reshape(np.clip((pixels_x[j] + (pixels_y[i] - pixels_x[0]) >> 1), 0, 255), -1)
                    continue
                if ang_f == 0:
                    predicted_block[i][j] = np.reshape(find_r_x(mode, (j + ang_i + 1), pixels_x, pixels_y), -1)

                else:
                    block = (((32 - ang_f) * find_r_x(mode, (j + ang_i + 1), pixels_x, pixels_y) +
                         ang_f * find_r_x(mode, (j + ang_i + 2), pixels_x,
                                          pixels_y) + 16) / 32).astype(float)
                    predicted_block[i][j] = np.reshape(np.round(block).astype(int), -1)
    return np.array(predicted_block, dtype='object')


def intra_predict(pixels_x: list,
                  pixels_y: list,
                  original_block: np.array) \
        -> tuple[np.array, np.array, int]:
    mode = 0
    pixels_x, pixels_y = substitution(pixels_x, pixels_y)

    predicted_block = dc_prediction(pixels_x, pixels_y)
    error_list = np.int64(original_block - predicted_block)
    residual_cost = sum(np.reshape(error_list**2, -1))

    # predicted_block_temp = planar_prediction(pixels_x, pixels_y)
    # error_list_temp = np.int64(original_block - predicted_block_temp)
    # residual_cost_temp = sum(np.reshape(error_list_temp**2, -1))
    #
    # if residual_cost_temp < residual_cost:
    #     residual_cost = residual_cost_temp
    #     error_list = error_list_temp
    #     predicted_block = predicted_block_temp
    #     mode = 1

    modes = [2, 10, 18, 26, 34]

    for i in modes:
        predicted_block_temp = angular_prediction(pixels_x, pixels_y, i)
        error_list_temp = np.int64(original_block - predicted_block_temp)
        residual_cost_temp = sum(np.reshape(error_list_temp**2, -1))
        if residual_cost_temp < residual_cost:
            residual_cost = residual_cost_temp
            error_list = error_list_temp
            predicted_block = predicted_block_temp
            mode = i

    return predicted_block, error_list, mode


def intra_predict_decode(pixels_x: list, pixels_y: list, mode: int) -> np.array:
    pixels_x, pixels_y = substitution(pixels_x, pixels_y)
    if mode == 0:
        predicted_block = dc_prediction(pixels_x, pixels_y)
    elif mode == 1:
        predicted_block = planar_prediction(pixels_x, pixels_y)
    else:
        predicted_block = angular_prediction(pixels_x, pixels_y, mode)
    if (predicted_block[0][0][0] > 255) or (predicted_block[0][0][0] < 0):
        print(mode)
    return predicted_block



# print(np.array(angular_prediction(test_1, test_2, 26)))
