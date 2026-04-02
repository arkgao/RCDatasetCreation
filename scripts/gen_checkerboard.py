"""Generate checkerboard calibration pattern image."""
import argparse
import numpy as np
import imageio


def gen_checkerboard(pixel_size, row_num, col_num):
    """Generate a black-white checkerboard pattern.

    Args:
        pixel_size: pixel size per grid cell
        row_num: number of rows
        col_num: number of columns

    Returns:
        numpy array of shape (H, W, 3) with values in [0, 1]
    """
    H = row_num * pixel_size
    W = col_num * pixel_size
    pattern = np.ones([H, W, 3])
    row_flag = True
    for i in range(row_num):
        col_flag = row_flag
        for j in range(col_num):
            color = np.array([0, 0, 0]) if col_flag else np.array([1, 1, 1])
            grid = np.ones([pixel_size, pixel_size, 3]) * color
            pattern[i*pixel_size:(i+1)*pixel_size, j*pixel_size:(j+1)*pixel_size, :] = grid
            col_flag = not col_flag
        row_flag = not row_flag
    return pattern


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate checkerboard pattern image')
    parser.add_argument('--row', type=int, default=9, help='number of rows')
    parser.add_argument('--col', type=int, default=12, help='number of columns')
    parser.add_argument('--pixel_size', type=int, default=50, help='pixel size per grid cell')
    parser.add_argument('--output', type=str, required=True, help='output image path')
    args = parser.parse_args()

    pattern = gen_checkerboard(args.pixel_size, args.row, args.col)
    imageio.imwrite(args.output, (255 * pattern).astype(np.uint8))
    print(f'Saved checkerboard ({args.row}x{args.col}) to {args.output}')
