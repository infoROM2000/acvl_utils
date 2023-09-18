from copy import deepcopy
from typing import List, Tuple, Union, Callable
import numpy as np
import torch
from torch.nn import functional as F
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.morphology import ball, dilation
from acvl_utils.instance_segmentation.instance_as_semantic_seg import label_with_component_sizes
from acvl_utils.cropping_and_padding.bounding_boxes import regionprops_bbox_to_proper_bbox, bounding_box_to_slice
from acvl_utils.morphology.morphology_helper import generic_filter_components
BORDER_LABEL = 2
CENTER_LABEL = 1
def pad_bbox(bounding_box: Union[List[List[int]], Tuple[Tuple[int, int]]], pad_amount: Union[int, List[int]],
             array_shape: Tuple[int, ...] = None) -> List[List[int]]:
    """
    """
    if isinstance(bounding_box, tuple):
        # convert to list
        bounding_box = [list(i) for i in bounding_box]
    else:
        # needed because otherwise we overwrite input which could have unforseen consequences
        bounding_box = deepcopy(bounding_box)

    if isinstance(pad_amount, int):
        pad_amount = [pad_amount] * len(bounding_box)

    for i in range(len(bounding_box)):
        new_values = [max(0, bounding_box[i][0] - pad_amount[i]), bounding_box[i][1] + pad_amount[i]]
        if array_shape is not None:
            new_values[1] = min(array_shape[i], new_values[1])
        bounding_box[i] = new_values

    return bounding_box

def regionprops_bbox_to_proper_bbox(regionprops_bbox: Tuple[int, ...]) -> List[List[int]]:
    """
    regionprops_bbox is what you get from `from skimage.measure import regionprops`
    """
    dim = len(regionprops_bbox) // 2
    return [[regionprops_bbox[i], regionprops_bbox[i + dim]] for i in range(dim)]


def bounding_box_to_slice(bounding_box: List[List[int]]):
    return tuple([slice(*i) for i in bounding_box])

def get_bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
    """
    this implementation uses less ram than the np.where one and is faster as well IF we expect the bounding box to
    be close to the image size. If it's not it's likely slower!
    :param mask:
    :param outside_value:
    :return:
    """
    Z, X, Y = mask.shape
    minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y
    zidx = list(range(Z))
    for z in zidx:
        if np.any(mask[z]):
            minzidx = z
            break
    for z in zidx[::-1]:
        if np.any(mask[z]):
            maxzidx = z + 1
            break

    xidx = list(range(X))
    for x in xidx:
        if np.any(mask[:, x]):
            minxidx = x
            break
    for x in xidx[::-1]:
        if np.any(mask[:, x]):
            maxxidx = x + 1
            break

    yidx = list(range(Y))
    for y in yidx:
        if np.any(mask[:, :, y]):
            minyidx = y
            break
    for y in yidx[::-1]:
        if np.any(mask[:, :, y]):
            maxyidx = y + 1
            break
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def get_bbox_from_mask_npwhere(mask: np.ndarray) -> List[List[int]]:
    where = np.array(np.where(mask))
    mins = np.min(where, 1)
    maxs = np.max(where, 1) + 1
    return [[i, j] for i, j in zip(mins, maxs)]

def pad_nd_image(image: Union[torch.Tensor, np.ndarray], new_shape: Tuple[int, ...] = None,
                 mode: str = "constant", kwargs: dict = None, return_slicer: bool = False,
                 shape_must_be_divisible_by: Union[int, Tuple[int, ...], List[int]] = None) -> \
        Union[Union[torch.Tensor, np.ndarray], Tuple[Union[torch.Tensor, np.ndarray], Tuple]]:
    """
    One padder to pad them all. Documentation? Well okay. A little bit
    Padding is done such that the original content will be at the center of the padded image. If the amount of padding
    needed it odd, the padding 'above' the content is larger,
    Example:
    old shape: [ 3 34 55  3]
    new_shape: [3, 34, 96, 64]
    amount of padding (low, high for each axis): [[0, 0], [0, 0], [20, 21], [30, 31]]
    :param image: can either be a numpy array or a torch.Tensor. pad_nd_image uses np.pad for the former and
           torch.nn.functional.pad for the latter
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
           len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in
           any of the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
           Example:
           image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
           image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).
    :param mode: will be passed to either np.pad or torch.nn.functional.pad depending on what the image is. Read the
           respective documentation!
    :param return_slicer: if True then this function will also return a tuple of python slice objects that you can use
           to crop back to the original image (reverse padding)
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
           divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match
           that will be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation (numpy) or torch.nn.functional.pad (torch)
    :returns: if return_slicer=False, this function returns the padded numpy array / torch Tensor. If
              return_slicer=True it will also return a tuple of slice objects that you can use to revert the padding:
              output, slicer = pad_nd_image(input_array, new_shape=XXX, return_slicer=True)
              reversed_padding = output[slicer] ## this is now the same as input_array, padding was reversed
    """
    if kwargs is None:
        kwargs = {}
    old_shape = np.array(image.shape)
    if shape_must_be_divisible_by is not None:
        assert isinstance(shape_must_be_divisible_by, (int, list, tuple, np.ndarray))
        if isinstance(shape_must_be_divisible_by, int):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(image.shape)
        else:
            if len(shape_must_be_divisible_by) < len(image.shape):
                shape_must_be_divisible_by = [1] * (len(image.shape) - len(shape_must_be_divisible_by)) + \
                                             list(shape_must_be_divisible_by)
    if new_shape is None:
        assert shape_must_be_divisible_by is not None
        new_shape = image.shape
    if len(new_shape) < len(image.shape):
        new_shape = list(image.shape[:len(image.shape) - len(new_shape)]) + list(new_shape)
    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)

        if len(shape_must_be_divisible_by) < len(new_shape):
            shape_must_be_divisible_by = [1] * (len(new_shape) - len(shape_must_be_divisible_by)) + \
                                         list(shape_must_be_divisible_by)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] %
                              shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [list(i) for i in zip(pad_below, pad_above)]

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        if isinstance(image, np.ndarray):
            res = np.pad(image, pad_list, mode, **kwargs)
        elif isinstance(image, torch.Tensor):
            # torch padding has the weirdest interface ever. Like wtf? Y u no read numpy documentation? So much easier
            torch_pad_list = [i for j in pad_list for i in j[::-1]][::-1]
            res = F.pad(image, torch_pad_list, mode, **kwargs)
    else:
        res = image
    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = tuple(slice(*i) for i in pad_list)
        return res, slicer
    
def _internal_convert_semantic_to_instance_mp(cropped_core_instances, cropped_border, spacing):
    cropped_current = np.copy(cropped_core_instances)
    already_dilated_mm = np.array((0, 0, 0))
    cropped_final = np.copy(cropped_core_instances)
    background_mask = (cropped_border == 0) & (cropped_core_instances == 0)
    while np.sum(cropped_border) > 0:
        strel_size = [0, 0, 0]
        maximum_dilation = max(already_dilated_mm)
        for i in range(3):
            if spacing[i] == min(spacing):
                strel_size[i] = 1
                continue
            if already_dilated_mm[i] + spacing[i] / 2 < maximum_dilation:
                strel_size[i] = 1
        ball_here = ball(1)
        if strel_size[0] == 0: ball_here = ball_here[1:2]
        if strel_size[1] == 0: ball_here = ball_here[:, 1:2]
        if strel_size[2] == 0: ball_here = ball_here[:, :, 1:2]
        # print(1)
        dilated = dilation(cropped_current, ball_here)
        dilated[background_mask] = 0
        diff = (cropped_current == 0) & (dilated != cropped_current)
        cropped_final[diff & cropped_border] = dilated[diff & cropped_border]
        cropped_border[diff] = 0
        cropped_current = dilated
        already_dilated_mm = [already_dilated_mm[i] + spacing[i] if strel_size[i] == 1 else
                              already_dilated_mm[i] for i in range(3)]
    return cropped_final

def convert_semantic_to_instanceseg(arr: np.ndarray,
                                    spacing: Union[Tuple[float, ...], List[float]] = (1, 1, 1),
                                    small_center_threshold: float = 30,
                                    isolated_border_as_separate_instance_threshold: float = 15) -> np.ndarray:
    """
    :param arr:
    :param spacing:
    :param small_center_threshold: volume, as dictated by spacing! If your spacing is (2, 2, 2) then a
    small_center_threshold of 16 would correspond to 2 pixels!
    :param isolated_border_as_separate_instance_threshold: volume, as dictated by spacing! If your spacing is (2, 2, 2) then a
    isolated_border_as_separate_instance_threshold of 16 would correspond to 2 pixels!
    :return:
    """
    assert np.issubdtype(arr.dtype, np.unsignedinteger), 'instance_segmentation must be an array of type unsigned ' \
                                                         'integer (can be uint8, uint16 etc)'
    spacing = np.array(spacing)
    small_center_threshold_in_pixels = round(small_center_threshold / np.prod(spacing))
    isolated_border_as_separate_instance_threshold_in_pixels = round(isolated_border_as_separate_instance_threshold / np.prod(
        spacing))
    # we first identify centers that are too small and set them to be border. This should remove false positive instances
    labeled_image, component_sizes = label_with_component_sizes(arr == CENTER_LABEL, connectivity=1)
    remove = np.array([i for i, j in component_sizes.items() if j <= small_center_threshold_in_pixels])
    remove = np.in1d(labeled_image.ravel(), remove).reshape(labeled_image.shape)
    arr[remove] = BORDER_LABEL
    # recompute core labels
    core_instances = label(arr == CENTER_LABEL)
    # prepare empty array for results
    final = np.zeros_like(core_instances, dtype=np.uint32)
    # besides the core instances we will need the borders
    border_mask = arr == BORDER_LABEL
    # we search for connected components and then convert each connected component into instance segmentation. This should
    # prevent bleeding into neighboring instances even if the instances don't touch
    connected_components, num_components = label(arr > 0, return_num=True, connectivity=1)
    max_component_idx = np.max(core_instances)
    rp = regionprops(connected_components)
    for r in rp:
        bbox = regionprops_bbox_to_proper_bbox(r.bbox)
        slicer = bounding_box_to_slice(bbox)
        cropped_mask = connected_components[slicer] == r.label
        cropped_core_instances = np.copy(core_instances[slicer])
        cropped_border = np.copy(border_mask[slicer])
        # remove other objects from the current crop, only keep the current connected component
        cropped_core_instances[~cropped_mask] = 0
        cropped_border[~cropped_mask] = 0
        cropped_current = np.copy(cropped_core_instances)
        unique_core_idx = np.unique(cropped_core_instances)
        # do not use len(unique_core_idx) == 1 because there could be one code filling the entire thing
        if np.sum(cropped_core_instances) == 0:
            # special case no core
            if np.sum(cropped_border) > isolated_border_as_separate_instance_threshold_in_pixels:
                final[slicer][cropped_border] = max_component_idx + 1
                max_component_idx += 1
        elif len(unique_core_idx) == 2:
            # special case only one core = one object
            final[slicer][(cropped_core_instances > 0) | cropped_border] = unique_core_idx[1]
        else:
            already_dilated_mm = np.array((0, 0, 0))
            cropped_final = np.copy(cropped_core_instances)
            while np.sum(cropped_border) > 0:
                strel_size = [0, 0, 0]
                maximum_dilation = max(already_dilated_mm)
                for i in range(3):
                    if spacing[i] == min(spacing):
                        strel_size[i] = 1
                        continue
                    if already_dilated_mm[i] + spacing[i] / 2 < maximum_dilation:
                        strel_size[i] = 1
                ball_here = ball(1)
                if strel_size[0] == 0: ball_here = ball_here[1:2]
                if strel_size[1] == 0: ball_here = ball_here[:, 1:2]
                if strel_size[2] == 0: ball_here = ball_here[:, :, 1:2]
                # print(1)
                dilated = dilation(cropped_current, ball_here)
                dilated[~cropped_mask] = 0
                diff = (cropped_current == 0) & (dilated != cropped_current)
                cropped_final[diff & cropped_border] = dilated[diff & cropped_border]
                cropped_border[diff] = 0
                cropped_current = dilated
                already_dilated_mm = [
                    already_dilated_mm[i] + spacing[i] if strel_size[i] == 1 else already_dilated_mm[i] for i in
                    range(3)]
            # now place result back
            final[slicer][cropped_mask] = cropped_final[cropped_mask]
    return final

def generate_ball(radius: Union[Tuple, List], spacing: Union[Tuple, List] = (1, 1, 1), dtype=np.uint8) -> np.ndarray:
    """
    Returns a ball/ellipsoid corresponding to the specified size (radius = list/tuple of len 3 with one radius per axis)
    If you use spacing, both radius and spacing will be interpreted relative to each other, so a radius of 10 with a
    spacing of 5 will result in a ball with radius 2 pixels.
    """
    radius_in_voxels = np.array([round(i) for i in radius / np.array(spacing)])
    n = 2 * radius_in_voxels + 1
    ball_iso = ball(max(n) * 2, dtype=np.float64)
    ball_resampled = resize(ball_iso, n, 1, 'constant', 0, clip=True, anti_aliasing=False, preserve_range=True)
    ball_resampled[ball_resampled > 0.5] = 1
    ball_resampled[ball_resampled <= 0.5] = 0
    return ball_resampled.astype(dtype)

def remove_all_but_largest_component(binary_image: np.ndarray, connectivity: int = None) -> np.ndarray:
    """
    Removes all but the largest component in binary_image. Replaces pixels that don't belong to it with background_label
    """
    filter_fn = lambda x, y: [i for i, j in zip(x, y) if j == max(y)]
    return generic_filter_components(binary_image, filter_fn, connectivity)

def generic_filter_components(binary_image: np.ndarray, filter_fn: Callable[[List[int], List[int]], List[int]],
                              connectivity: int = None):
    """
    filter_fn MUST return the component ids that should be KEPT!
    filter_fn will be called as: filter_fn(component_ids, component_sizes) and is expected to return a List of int
    returns a binary array that is True where the filtered components are
    """
    labeled_image, component_sizes = label_with_component_sizes(binary_image, connectivity)
    component_ids = list(component_sizes.keys())
    component_sizes = list(component_sizes.values())
    keep = filter_fn(component_ids, component_sizes)
    return np.in1d(labeled_image.ravel(), keep).reshape(labeled_image.shape)

def filter_fn(x, y):
    maxy = max(y)
    return [i for i, j in zip(x, y) if j == maxy]