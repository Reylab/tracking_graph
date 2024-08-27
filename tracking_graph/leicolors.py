leicolors_list = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.620690, 0.0, 0.0], [0.413793, 0.0, 0.758621],
                [0.965517, 0.517241, 0.034483], [0.448276, 0.379310, 0.241379], [1.0, 0.103448, 0.724138],
                [0.545, 0.545, 0.545], [0.586207, 0.827586, 0.310345], [0.965517, 0.620690, 0.862069],
                [0.620690, 0.758621, 1.]] #silly name, just colors that look different enough
def leicolors(x):
    return leicolors_list[int(x) % len(leicolors_list)]

def hex_leicolors(x):
    rgb = leicolors_list[int(x) % len(leicolors_list)]
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))