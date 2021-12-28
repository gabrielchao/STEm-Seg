from stemseg.config import cfg
from stemseg.structures import ImageList
from stemseg.utils import transforms
from torch.utils.data import Dataset
from stemseg.data.interactive_video_dataset_parser import InteractiveVideoSequence, load_guidance_map
from stemseg.data.common import scale_and_normalize_images, compute_resize_params_2

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class InteractiveInferenceLoader(Dataset):
    """
    Dataset that supports the loading of both images and guidance maps from a single sequence.
    """
    def __init__(self, sequence:InteractiveVideoSequence):
        super().__init__()

        self.np_to_tensor = transforms.ToTorchTensor(format='CHW')

        if sequence.num_guided_instances != 1:
            raise ValueError("InteractiveInferenceLoader only supports loading \
                from InteractiveVideoSequences containing only one guided instance")
        self.sequence = sequence

        self.loaded_images = dict()
        self.loaded_guidance_maps = dict()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        """
        Return image, (image_width, image_height), guidance_map, index
        :return tuple(
            tensor(C, H, W),
            tuple(int, int),
            tensor(2, H, W),
            int
        )
        """
        image_path, (pos_path, neg_path) = self.sequence.get_step_paths(index)

        # ----- Images -----
        if index not in self.loaded_images:
            self.loaded_images[index] = cv2.imread(os.path.join(self.sequence.base_dir, image_path), cv2.IMREAD_COLOR)
        image = self.loaded_images[index]

        image_height, image_width = image.shape[:2]

        # convert image to tensor
        image = self.np_to_tensor(image).float()

        # resize image
        new_width, new_height, _ = compute_resize_params_2((image_width, image_height), cfg.INPUT.MIN_DIM, cfg.INPUT.MAX_DIM)
        image = F.interpolate(image.unsqueeze(0), (new_height, new_width), mode='bilinear', align_corners=False)

        # compute scale factor for image resizing
        image = scale_and_normalize_images(image, cfg.INPUT.IMAGE_MEAN, cfg.INPUT.IMAGE_STD, not cfg.INPUT.BGR_INPUT, cfg.INPUT.NORMALIZE_TO_UNIT_SCALE)

        # ----- Guidance Maps -----
        if index not in self.loaded_guidance_maps:
            pos_map = load_guidance_map(os.path.join(self.sequence.guidance_dir, pos_path))
            neg_map = load_guidance_map(os.path.join(self.sequence.guidance_dir, neg_path))
            self.loaded_guidance_maps[index] = pos_map, neg_map
        else:
            pos_map, neg_map = self.loaded_guidance_maps[index]
        
        guidance_map = np.stack([pos_map, neg_map], axis=0) # (2, H, W)
        guidance_map = torch.Tensor(guidance_map)

        # resize guidance map
        guidance_map = F.interpolate(guidance_map.unsqueeze(0), (new_height, new_width), mode='bilinear', align_corners=False) # (1, 2, H, W)

        assert image.shape[2:3] == guidance_map.shape[2:3], \
            f'Image and guidance map height/width mismatch: {image.shape} and {guidance_map.shape}'

        return image.squeeze(0), (image_width, image_height), guidance_map.squeeze(0), index
    
    def preload(self):
        """
        Preload all images and guidance maps in this sequence.
        """
        for t in range(len(self.sequence)):
            image_path, (pos_path, neg_path) = self.sequence.get_step_paths(t)
            self.loaded_images[t] = cv2.imread(os.path.join(self.sequence.base_dir, image_path), cv2.IMREAD_COLOR)
            pos_map = load_guidance_map(os.path.join(self.sequence.guidance_dir, pos_path))
            neg_map = load_guidance_map(os.path.join(self.sequence.guidance_dir, neg_path))
            self.loaded_guidance_maps[t] = pos_map, neg_map


def collate_fn(samples):
    """
    Return images and guidance maps collated along the sequence dimension.
    The second (originally time) dimension is left as length 1 while the first
    (originally sequence) dimension has length T.
    :return tuple(
        ImageList(T, 1, C, H, W),
        tensor(T, 1, 2, H, W),
        int
    )
    """
    image_seqs, original_dims, guidance_maps, idxes = zip(*samples)

    image_seqs = [[im] for im in image_seqs]
    image_seqs = ImageList.from_image_sequence_list(image_seqs, original_dims) # This resizes images to nearest multiple of 32
    
    guidance_maps = torch.stack(guidance_maps, dim=0) # (T, 2, H, W)
    # Resize guidance maps to match image_seqs using same logic as in ImageList
    max_height, max_width = image_seqs.max_size
    resized_guidance = torch.zeros((guidance_maps.shape[0], 2, max_height, max_width))
    resized_guidance[:, :, :guidance_maps.shape[2], :guidance_maps.shape[3]] = guidance_maps # (T, 2, mH, mW)
    resized_guidance.unsqueeze_(1) # (T, 1, 2, mH, mW)
    
    return image_seqs, resized_guidance, idxes
