import numpy as np
import torch

from stemseg.utils.interaction.gen_clicks import gen_gaussian_map, simulate_single_centric
from stemseg.utils.interaction.gen_scribbles import get_scribble

def get_center_click_maps(annotations:np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    Additively updates the given guidance tube with single positive center clicks on every frame of each instance.
    :param annotations: ndarray(I, T, 2, H, W)
    :param masks: ndarray(I, T, H, W)
    :return ndarray(I, T, 2, H, W)
    """
    I, T, H, W = masks.shape
    for i in range(I):
        for t in range(T):
            try:
                click, _ = simulate_single_centric(masks[i, t])
            except:
                # instance not in frame
                pass
            map = gen_gaussian_map(click, 10, (H, W))
            annotations[i, t, 0] += map

    return annotations

def get_blank_interaction_maps(shape):
    """
    Get a blank positive + negative guidance map tube. Convention: the first (index 0)
    channel is the positive and the second (index 1) is the negative.
    :param shape: tuple(I, T, H, W)
    :return ndarray(I, T, 2, H, W)
    """
    I, T, H, W = shape
    return np.zeros((I, T, 2, H, W))

def get_clicks_for_all_frames(sub_targets: tuple):
    """
    Get a guidance map tube for the given sequences with one click on each instance on all frames.
    :param sub_targets: tuple(
                    dict(
                        'masks' -> tensor(1, T, H, W),
                        'category_ids' -> tensor(),
                        'labels' -> tensor(),
                        'ignore_masks' -> tensor(T, H, W)
                    )
                ) (length N)
    :return tensor(N, T, 2, H, W)
    """
    N, T, H, W = len(sub_targets), 
    torch.zeros((N, T, 2, H, W))
    interactions = []
    for d in sub_targets:
        interactions.append(
            torch.from_numpy(
                get_center_click_maps(
                    get_blank_interaction_maps((1, T, H, W)), d['masks'].detach().numpy().cpu())
                        .squeeze(0))) # (T, 2, H, W)
    interaction_seqs = torch.stack(interactions, dim=0) # (N, T, 2, H, W)
    return interaction_seqs
