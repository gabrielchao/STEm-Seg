import numpy as np
import torch

from stemseg.utils.interaction.gen_clicks import gen_gaussian_map, simulate_single_centric
from stemseg.utils.interaction.gen_scribbles import get_scribble

def get_center_click_maps(annotations:np.ndarray, masks: np.ndarray, gaussian_size: int=10) -> np.ndarray:
    """
    Additively updates the given guidance tube with single positive center clicks on every frame of each instance.
    :param annotations: ndarray(I, T, 2, H, W) or ndarray(I, T, 1, H, W)
    :param masks: ndarray(I, T, H, W)
    :param gaussian_size: int
    :return ndarray(I, T, 2, H, W) or ndarray(I, T, 1, H, W)
    """
    I, T, H, W = masks.shape
    for i in range(I):
        for t in range(T):
            try:
                click, _ = simulate_single_centric(masks[i, t])
                map = gen_gaussian_map(click[:1], gaussian_size, (H, W)) # Get first click only
                annotations[i, t, 0] += map
            except:
                # instance not in frame
                pass
    return annotations

def get_blank_interaction_maps(shape, with_negative=True):
    """
    Get a blank positive + negative guidance map tube. Convention: the first (index 0)
    channel is the positive and the second (index 1) is the negative. If with_negative
    is set to false, the negative channel will not be included.
    :param shape: tuple(I, T, H, W)
    :param with_negative: bool
    :return ndarray(I, T, 2, H, W) or ndarray(I, T, 1, H, W)
    """
    I, T, H, W = shape
    C = 2 if with_negative else 1
    return np.zeros((I, T, C, H, W))

def get_clicks_for_all_frames(sub_targets: tuple, with_negative=True, kernel_size: int=10):
    """
    Get a guidance map tube for the given sequences with one click on each instance on all frames.
    If with_negative is set to true, a blank negative tube will be included as the second guidance channel.
    :param sub_targets: tuple(
                    dict(
                        'masks' -> tensor(1, T, H, W),
                        'category_ids' -> tensor(),
                        'labels' -> tensor(),
                        'ignore_masks' -> tensor(T, H, W)
                    )
                ) (length N)
    :param with_negative: bool
    :param kernel_size: int
    :return tensor(N, T, 2, H, W) or tensor(N, T, 1, H, W)
    """
    N = len(sub_targets)
    T, H, W = sub_targets[0]['masks'].shape[1:4]
    interactions = []
    for d in sub_targets:
        interactions.append(
            torch.Tensor( # Default FloatTensor dtype
                get_center_click_maps(
                    get_blank_interaction_maps((1, T, H, W), with_negative), d['masks'].detach().cpu().numpy(), kernel_size)
                        .squeeze(0))) # (T, 2, H, W)
    interaction_seqs = torch.stack(interactions, dim=0) # (N, T, 2, H, W)
    return interaction_seqs
