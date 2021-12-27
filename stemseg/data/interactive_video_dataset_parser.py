from pycocotools import mask as masktools

import cv2
import json
import numpy as np
import os
from copy import deepcopy

from stemseg.data.generic_video_dataset_parser import GenericVideoSequence

def parse_interactive_video_dataset(base_dir, dataset_json, guidance_dir):
    """
    Parse the interactive dataset given by the given locations, returning a list of
    InteractiveVideoSequences and a meta-info dictionary. The dataset must include
    images and guidance maps. The expected directory structure is that generated 
    by stemseg.utils.interaction.gen_files.
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
    
    guidance_maps = dict() # dict(str -> dict(int -> dict('positive' -> list(str), 'negative' -> list(str) (length T))) (length I)) (length N)
    # get guidance map paths
    for sequence_name in os.listdir(guidance_dir):
        if not os.path.isdir(os.path.join(guidance_dir, sequence_name)):
            continue

        instances = dict()
        for i, instance_name in enumerate(sorted(os.listdir(os.path.join(guidance_dir, sequence_name)))):
            # Get positive maps
            try:
                pos_files = os.listdir(os.path.join(guidance_dir, sequence_name, instance_name, 'positive'))
            except FileNotFoundError:
                raise ValueError(f"Could not find positive guidance directory for sequence {sequence_name} {instance_name}")
            # Don't include guidance_dir in guidance paths, same as image_paths not including base_dir
            pos_files = [os.path.join(sequence_name, instance_name, 'positive', file_name) for file_name in pos_files]
            
            # Get negative maps
            try:
                neg_files = os.listdir(os.path.join(guidance_dir, sequence_name, instance_name, 'negative'))
            except FileNotFoundError:
                raise ValueError(f"Could not find negative guidance directory for sequence: {sequence_name} {instance_name}")
            neg_files = [os.path.join(sequence_name, instance_name, 'negative', file_name) for file_name in neg_files]

            if len(pos_files) != len(neg_files):
                raise ValueError(f"Mismatch in number of positive and negative guidance time steps for sequence {sequence_name} {instance_name}")

            instances[i] = {
                'positive': pos_files,
                'negative': neg_files
            }
        
        first_length = len(instances[0]['positive'])
        for instance in instances.values():
            if len(instance['positive']) != first_length:
                raise ValueError(f"Mismatch in number of guidance time steps per instance for sequence: {sequence_name}")

        guidance_maps[sequence_name] = instances
    
    # insert guidance map paths into dataset
    for seq in dataset["sequences"]:
        if seq['id'] not in guidance_maps:
            raise ValueError(f"No guidance maps found for sequence: {seq['id']}")
        if seq['length'] != len(guidance_maps[seq['id']][0]['positive']):
            raise ValueError(f"Expected {seq['length']} guidance time steps for sequence {seq['id']}, found {len(guidance_maps[seq['id']][0]['positive'])}")
        seq['guidance_paths'] = guidance_maps[seq['id']] # dict(int -> dict('positive' -> list(str), 'negative' -> list(str) (length T))) (length I)

    seqs = [InteractiveVideoSequence(seq, base_dir, guidance_dir) for seq in dataset["sequences"]]

    return seqs, meta_info


class InteractiveVideoSequence(GenericVideoSequence):
    """
    Class that extends GenericVideoSequence to support the loading of guidance maps.
    """
    def __init__(self, seq_dict, base_dir, guidance_dir, is_split=False):
        """
        Initialize an InteractiveVideoSequence.
        :param seq_dict: dict()
        :param base_dir: str
        :param guidance_dir: str
        :param is_split: bool. True if this sequence has already been split.
        """
        super().__init__(seq_dict, base_dir)

        self.guidance_dir = guidance_dir
        self.guidance_paths = seq_dict["guidance_paths"] # dict(int -> dict('positive' -> list(str), 'negative' -> list(str) (length T))) (length I)
        self.is_split = is_split

    @property
    def num_guided_instances(self):
        """
        The number of instances with guidance maps in this sequence.
        """
        return len(self.guidance_paths)
    
    def load_guidance_maps(self, frame_idxes=None):
        """
        Load guidance tubes from disk.
        :param frame_idxes: list()
        :return: list(ndarray(T, 2, H, W)) (length I)
        """
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))
        
        guidance_maps = [] # list(ndarray(T, 2, H, W)) (length I)
        for instance in self.guidance_paths.values():
            pos_i = [] # list(ndarray(H, W)) (length T)
            neg_i = [] # list(ndarray(H, W)) (length T)
            
            for t in frame_idxes:
                pos = cv2.imread(os.path.join(self.guidance_dir, instance['positive'][t]), cv2.IMREAD_GRAYSCALE).astype('float64')
                if pos is None:
                    raise ValueError("No positive guidance map found at path: {}".format(os.path.join(self.guidance_dir, instance['positive'][t])))
                pos = pos.astype('float64')
                pos /= 255
                pos_i.append(pos)
            
            for t in frame_idxes:
                neg = cv2.imread(os.path.join(self.guidance_dir, instance['negative'][t]), cv2.IMREAD_GRAYSCALE).astype('float64')
                if neg is None:
                    raise ValueError("No negative guidance map found at path: {}".format(os.path.join(self.guidance_dir, instance['negative'][t])))
                neg = neg.astype('float64')
                neg /= 255
                neg_i.append(neg)
            
            pos_i = np.stack(pos_i, axis=0) # ndarray(T, H, W)
            neg_i = np.stack(neg_i, axis=0) # ndarray(T, H, W)
            guidance_map = np.stack([pos_i, neg_i], axis=1) # ndarray(T, 2, H, W)
            guidance_maps.append(guidance_map)

        return guidance_maps

    # Override
    def filter_zero_instance_frames(self):
        t_to_keep = [t for t in range(len(self)) if len(self.segmentations[t]) > 0]
        self.image_paths = [self.image_paths[t] for t in t_to_keep]
        self.segmentations = [self.segmentations[t] for t in t_to_keep]
        self.guidance_paths = {
            iid: {
                'positive': [self.guidance_paths[iid]['positive'][t] for t in t_to_keep],
                'negative': [self.guidance_paths[iid]['negative'][t] for t in t_to_keep]
            }
            for iid in self.guidance_paths.keys()
        }

    # Override
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
            "guidance_paths": {
                iid: {
                    'positive': [self.guidance_paths[iid]['positive'][t] for t in frame_idxes],
                    'negative': [self.guidance_paths[iid]['negative'][t] for t in frame_idxes]
                }
                for iid in instance_ids_to_keep
            }
        }

        return self.__class__(subseq_dict, self.base_dir, self.guidance_dir, self.is_split)

    def split_by_guided_instances(self):
        """
        Splits this sequence into InteractiveVideoSequences with the same frames but
        with only a single guided instance each. Each sequence will have the id 
        '{original_id}_instance_{instance_id}'. Can only be called if this sequence
        has not been split before.
        :return list(InteractiveVideoSequence)
        """
        if self.is_split:
            raise TypeError("Cannot split InteractiveVideoSequence that has already been split")

        sequences = []
        for iid in self.guidance_paths.keys():
            seq_dict = {
                "id": f"{self.id}_instance_{iid}",
                "height": self.image_dims[0],
                "width": self.image_dims[1],
                "image_paths": self.image_paths.copy(),
                "categories": self.instance_categories.copy(),
                "segmentations": deepcopy(self.segmentations),
                "guidance_paths": {
                    iid: deepcopy(self.guidance_paths[iid])
                }
            }
            seq = self.__class__(seq_dict, self.base_dir, self.guidance_dir, True)
            sequences.append(seq)
        return sequences


if __name__ == '__main__':
    sequences, meta_info = parse_interactive_video_dataset(
        '/home/gabriel/datasets/DAVIS/JPEGImages/480p',
        '/home/gabriel/datasets/dataset_jsons/davis_val.json',
        '/home/gabriel/datasets/DAVIS/CustomGuidance/480p')
    print([sequence.id for sequence in sequences])

    print('Split sequences:')
    print([subseq.id for seq in sequences for subseq in seq.split_by_guided_instances()])

    import timeit
    print(f'Loading guidance for sequence {sequences[0].id}...')
    print(f'Done in {timeit.timeit(lambda: print(sequences[0].load_guidance_maps()[0].shape), number=1)}')
