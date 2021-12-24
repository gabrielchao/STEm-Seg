# Adapted from "Two-in-One Refinement for Interactive Segmentation"

from scipy.ndimage.measurements import center_of_mass
import numpy as np
from scipy import ndimage

# broken gts
from scipy.ndimage.measurements import label
from skimage.measure import regionprops


def simulate_single_centric(object_mask):
    """
    Simulates the object selection click at the center of the object. If the center lies outside the object, pick the
    pixel which is farthest from the boundary.
    """
    cx_adjusted = False

    try:
        cx_click = np.array([center_of_mass(object_mask)], dtype=int)
    except:
        raise IndexError('Center of mass computation failed for given instance mask')

    if object_mask[cx_click[0][0], cx_click[0][1]] == 0.:
        label_blobs, num_blobs = label(object_mask)
        if num_blobs > 1:
            object_mask_props = regionprops(label_blobs)
            sort_prop_id = np.argsort(np.array([-p.area for p in object_mask_props], dtype=int))
            sort_prop_id = sort_prop_id[0]  # :min(pick, sort_prop_id.size)]
            sort_prop_id = sort_prop_id + 1
            _blob = (label_blobs == sort_prop_id) * 1.
            _blob = _blob.astype(np.float32)
            dt = ndimage.distance_transform_edt(_blob)
            cx_click = np.array(np.where(dt == dt.max())).transpose()
        return cx_click, True

    return cx_click, cx_adjusted


def gen_circular_map(clicks, size, radius=5, d_type=np.float32):
    guidance_map = np.zeros(size, dtype=bool)
    for i in range(len(clicks)):
        x, y = np.ogrid[:size[0], :size[1]]
        dist_from_center = (x - clicks[i][0]) ** 2 + (y - clicks[i][1]) ** 2
        mask = dist_from_center <= radius ** 2
        guidance_map = np.logical_or(guidance_map, mask)

    return guidance_map.astype(d_type)


def simulate_random(object_mask):
    dt = ndimage.distance_transform_edt(object_mask)
    dt_min = min(30, np.max(dt))
    candidates = np.argwhere(dt >= dt_min)
    clicks_index = np.random.choice(np.array(range(len(candidates))), 1)
    clicks = candidates[clicks_index]
    return clicks


def gen_gaussian_map(clicks, sk, size):
    """
    :param clicks: Co-ordinates of the clicks for the given image
    :param size: Size of the resultant map
    :param sk: Size of the Gaussian Kernel - (2*sk + 1) x (2*sk + 1)
    :return: Map
    """
    dim1 = size[0] + 2 * sk
    dim2 = size[1] + 2 * sk
    guidance_map = np.zeros((dim1, dim2))
    for i in range(len(clicks)):
        g_k = make_gaussian((sk * 2 + 1, sk * 2 + 1))
        guidance_map[(sk + clicks[i][0] - sk):(sk + clicks[i][0] + sk + 1),
        (sk + clicks[i][1] - sk):(sk + clicks[i][1] + sk + 1)] = \
            guidance_map[(sk + clicks[i][0] - sk):(sk + clicks[i][0] + sk + 1),
            (sk + clicks[i][1] - sk):(sk + clicks[i][1] + sk + 1)] + g_k
    guidance_map = crop_center(guidance_map, size[1], size[0])
    return guidance_map


def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)


def crop_center(img, crop_x, crop_y):
    y, x = img.shape
    start_x = x // 2 - (crop_x // 2)
    start_y = y // 2 - (crop_y // 2)
    return img[start_y:start_y + crop_y, start_x:start_x + crop_x]


def gen_euclidean_map(clicks, size):
    """
    Generate the euclidean distance transformation given the user click
    """
    guidance_map = np.zeros(size)
    clicks = np.array(clicks)
    guidance_map[tuple([clicks[:, 0], clicks[:, 1]])] = 1
    dt = ndimage.distance_transform_edt(np.logical_not(guidance_map > 0))
    dt[dt > 255.] = 255.
    return dt
