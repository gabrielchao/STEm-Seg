from pycocotools import mask as masktools

import cv2
import json
import numpy as np
import os

from stemseg.data.generic_video_dataset_parser import GenericVideoSequence

def parse_interactive_video_dataset(base_dir, dataset_json, guidance_dir):
    """
    Parse the interactive dataset given by the given locations, returning a list of
    InteractiveVideoSequences and a meta-info dictionary. The dataset must include
    images and guidance maps.
    :param base_dir: str
    :param dataset_json: str
    :param guidance_dir: str
    :return tuple(list(InteractiveVideoSequence), dict())
    """
    with open(dataset_json, 'r') as fh:
        dataset = json.load(fh)

    meta_info = dataset["meta"]

    # convert instance and category IDs from str to int
    meta_info["category_labels"] = {int(k): v for k, v in meta_info["category_labels"].items()}

    if "segmentations" in dataset["sequences"][0]:
        for seq in dataset["sequences"]:
            seq["categories"] = {int(iid): cat_id for iid, cat_id in seq["categories"].items()}
            seq["segmentations"] = [
                {
                    int(iid): seg
                    for iid, seg in seg_t.items()
                }
                for seg_t in seq["segmentations"]
            ]

            # sanity check: instance IDs in "segmentations" must match those in "categories"
            seg_iids = set(sum([list(seg_t.keys()) for seg_t in seq["segmentations"]], []))
            assert seg_iids == set(seq["categories"].keys()), "Instance ID mismatch: {} vs. {}".format(
                seg_iids, set(seq["categories"].keys())
            )
    
    guidance_maps = dict()
    # get guidance map paths
    for sequence_name in os.listdir(guidance_dir):
        if not os.path.isdir(sequence_name):
            continue

        instances = dict()
        for i, instance_name in enumerate(sorted(os.listdir(os.path.join(guidance_dir, sequence_name)))):
            files = os.listdir(os.path.join(guidance_dir, sequence_name, instance_name))
            # Don't include guidance_dir in guidance paths, same as image_paths not including base_dir
            files = [os.path.join(sequence_name, instance_name, file_name) for file_name in files]
            instances[i] = files
        
        first_length = len(instances[0])
        for instance in instances:
            if len(instance) != first_length:
                raise ValueError(f"Mismatch in number of guidance time steps per instance for sequence: {sequence_name}")

        guidance_maps[sequence_name] = instances
    
    # insert guidance map paths into dataset
    for seq in dataset["sequences"]:
        if seq['id'] not in guidance_maps:
            raise ValueError(f"No guidance maps found for sequence: {seq['id']}")
        if seq['length'] != len(guidance_maps[seq['id']][0]):
            raise ValueError(f"Expected {seq['length']} guidance time steps for sequence {seq['id']}, found {len(guidance_maps[seq['id']][0])}")
        seq['guidance_paths'] = guidance_maps[seq['id']] # dict(int -> list(str) (length T)) (length I)

    seqs = [InteractiveVideoSequence(seq, base_dir, guidance_dir) for seq in dataset["sequences"]]

    return seqs, meta_info


class InteractiveVideoSequence(GenericVideoSequence):
    """
    Class that extends GenericVideoSequence to support the loading of guidance maps.
    """
    def __init__(self, seq_dict, base_dir, guidance_dir):
        """
        Initialize an InteractiveVideoSequence.
        :param seq_dict: dict()
        :param base_dir: str
        :param guidance_dir: str
        """
        super().__init__(seq_dict, base_dir)

        self.guidance_dir = guidance_dir
        self.guidance_paths = seq_dict["guidance_paths"] # dict(int -> list(str) (length T)) (length I)
    
    def load_guidance_maps(self, frame_idxes=None):
        """
        Load guidance maps from disk.
        :param frame_idxes: list()
        :return: list(list(ndarray) (length T)) (length I)
        """
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))
        
        guidance_maps = []
        for instance in self.guidance_paths.values():
            guidance_i = []
            
            for t in frame_idxes:
                guidance_map = cv2.imread(os.path.join(self.guidance_dir, instance[t]), cv2.IMREAD_GRAYSCALE).astype('float64')
                if guidance_map is None:
                    raise ValueError("No guidance map found at path: {}".format(os.path.join(self.guidance_dir, instance[t])))
                guidance_map = guidance_map.astype('float64')
                guidance_map /= 255
                guidance_i.append(guidance_map)

        return guidance_maps

    def filter_zero_instance_frames(self):
        t_to_keep = [t for t in range(len(self)) if len(self.segmentations[t]) > 0]
        self.image_paths = [self.image_paths[t] for t in t_to_keep]
        self.segmentations = [self.segmentations[t] for t in t_to_keep]
        self.guidance_paths = {
            iid: [self.guidance_path[iid][t] for t in t_to_keep] 
            for iid in self.guidance_paths.keys()
        }

    def extract_subsequence(self, frame_idxes, new_id=""):
        assert all([t in range(len(self)) for t in frame_idxes])
        instance_ids_to_keep = set(sum([list(self.segmentations[t].keys()) for t in frame_idxes], []))

        subseq_dict = {
            "id": new_id if new_id else self.id,
            "height": self.image_dims[0],
            "width": self.image_dims[1],
            "image_paths": [self.image_paths[t] for t in frame_idxes],
            "categories": {iid: self.instance_categories[iid] for iid in instance_ids_to_keep},
            "segmentations": [
                {
                    iid: segmentations_t[iid]
                    for iid in segmentations_t if iid in instance_ids_to_keep
                }
                for t, segmentations_t in enumerate(self.segmentations) if t in frame_idxes
            ],
            "guidance_maps": {
                iid: [self.guidance_paths[iid][t] for t in frame_idxes]
                for iid in instance_ids_to_keep
            }
        }

        return self.__class__(subseq_dict, self.base_dir, self.guidance_dir)
