from collections import defaultdict, namedtuple

from stemseg.config import cfg

from stemseg.data.common import compute_resize_params_2, semseg_mask_to_instance_masks

from stemseg.inference.online_chainer import OnlineChainer
from stemseg.inference.clusterers import SequentialClustering

from stemseg.modeling.embedding_utils import get_nb_free_dims
from stemseg.modeling.model_builder import build_model

from stemseg.structures.image_list import ImageList

from stemseg.utils.constants import ModelOutput

import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractiveModel(nn.Module):
    def __init__(self, dataset='davis'):
        self.model = build_model(restore_pretrained_backbone_wts=False)
        self.chainer = OnlineChainer(self.create_clusterer(), embedding_resize_factor=resize_scale)
        
        self.EmbeddingMapEntry = namedtuple(
            "EmbeddingMapEntry", ["subseq_frames", "embeddings", "bandwidths", "seediness"])
        
        assert dataset == cfg.TRAINING.MODE == 'davis', "InteractiveModel only implemented for davis."
        self.dataset = dataset


    def model_forward(self, image_seqs: ImageList, interaction_seqs: list, targets: list) -> list:
        """
        Do a forward pass on a N-length batch of sequences of images and targets. Returns a list containing N dicts, each
        representing a sequence. Each dict is the same data type as that output by InferenceModel.
        :param image_seqs: ImageList
        :param interaction_seqs: list(BinaryMaskSequenceList)
        :param targets: List (length N) of dicts, each containing a 'masks' field containing a tensor of
        shape (I (instances), T, H, W)
        :return tuple(
                    dict(
                        ModelOutput.INFERENCE -> dict(
                            ModelOutput.EMBEDDINGS -> ,
                            ModelOutput.SEMSEG_MASKS ->
                        )
                    ),
                    list(dict('fg_masks' -> tensor(T, C, H, W), 
                        'multiclass_masks' -> tensor(T, C, H, W), 
                        'embeddings' -> list(namedtuple(
                            'subseq_frames' -> list(int), 
                            'embeddings' -> tensor, 
                            'bandwidths' -> tensor, 
                            'seediness' -> tensor))))
                )
        """
        # TODO: pass in interactions
        output = self.model(image_seqs, targets)
        all_embeddings = output[ModelOutput.INFERENCE][ModelOutput.EMBEDDINGS]
        all_semsegs = output[ModelOutput.INFERENCE][ModelOutput.SEMSEG_MASKS]

        embeddings, bandwidths, seediness = self.model.split_embeddings(all_embeddings)
        frames = list(range(image_seqs.num_frames))

        # Split batch into individual sequences
        sequences = []
        split_e, split_b, split_s = torch.split(embeddings, 1), torch.split(bandwidths, 1), torch.split(seediness, 1)
        split_sem = torch.split(all_semsegs, 1)
        for e, b, s, sem in zip(split_e, split_b, split_s, split_sem):
            # Compute fg_masks and multiclass_masks from semseg logits for this sequence         
            # Make list of semseg logits with dummy counts of 1 count each so we can reuse get_semseg_masks()
            # sem: [T, C, H, W]
            semseg_logits = [[0., 0] for _ in range(image_seqs.num_frames)]
            for t, sem_frame in enumerate(sem):
                # sem_frame: [C, H, W]
                semseg_logits[t][0] += sem_frame.unsqueeze(0) # [1, C, H, W]
                semseg_logits[t][1] += 1
                assert semseg_logits[t][1] == 1
            fg_masks, multiclass_masks = self.get_semseg_masks(semseg_logits)
            
            # Finally, create dict for this sequence
            sequences.append({
                "fg_masks": fg_masks,
                "multiclass_masks": multiclass_masks,
                "embeddings": self.EmbeddingMapEntry(frames, e, b, s)
            })
        
        # Finished emulating InferenceModel behaviour - each element in sequences is the
        # same data type as InferenceModel output.
        return output, sequences
    

    def cluster_sequence(self, all_embeddings, fg_masks, multiclass_masks, max_tracks, original_dims):
        """
        Produce a tensor containing a mask tube for each instance, predicted from 
        the embeddings and foreground masks for a single sequence.
        :param all_embeddings: list(namedtuple(
                        'subseq_frames' -> list(int), 
                        'embeddings' -> tensor, 
                        'bandwidths' -> tensor, 
                        'seediness' -> tensor))
        :param fg_masks: tensor(T, C, H, W)
        :param multiclass_masks: tensor(T, C, H, W)
        :param max_tracks: int
        :return: tensor(I, T, H, W) (I is the number of instances)
        """
        subseq_dicts = []
        for i, (subseq_frames, embeddings, bandwidths, seediness) in enumerate(all_embeddings):
            subseq_dicts.append({
                "frames": subseq_frames,
                "embeddings": embeddings,
                "bandwidths": bandwidths,
                "seediness": seediness,
            })

        (track_labels, instance_pt_counts, instance_lifetimes), framewise_mask_idxes, subseq_labels_list, \
            fg_embeddings, subseq_clustering_meta_info = self.chainer.process(
            fg_masks, subseq_dicts)
        
        masks = self.process_output(
            original_dims, framewise_mask_idxes, track_labels, instance_pt_counts, instance_lifetimes, multiclass_masks,
            fg_masks.shape[-2:], 4.0, max_tracks, device=self.clustering_device
        )

        instance_masks = semseg_mask_to_instance_masks(masks)
        return instance_masks
        

    def process_output(self, original_dims, track_mask_idxes, track_mask_labels, instance_pt_counts, instance_lifetimes,
                         category_masks, mask_dims, mask_scale, max_tracks, device="cpu"):
        """
        Given a list of mask indices per frame, creates a sequence of masks for the entire sequence.
        :param original_dims: tuple(int, int) (width, height)
        :param track_mask_idxes: list(tuple(tensor, tensor))
        :param track_mask_labels: list(tensor)
        :param instance_pt_counts: dict(int -> int)
        :param instance_lifetimes: 
        :param category_masks: irrelevant
        :param mask_dims: tuple(int, int) (height, width)
        :param mask_scale: int
        :param max_tracks: int
        :param device: str
        :return: tensor(T, H, W)
        """
        mask_height, mask_width = mask_dims
        image_width, image_height = original_dims

        assert len(track_mask_idxes) == len(track_mask_labels)
        assert max_tracks < 256

        # filter out small/unstable instances
        instances_to_keep = [
                                instance_id for instance_id, _ in sorted(
                [(k, v) for k, v in instance_lifetimes.items()], key=lambda x: x[1], reverse=True
            ) if instance_id != self.outlier_label
        ]

        instances_to_keep = instances_to_keep[:max_tracks]
        num_tracks = len(instances_to_keep)

        print("Number of instances: ", len(instances_to_keep))

        # move tensors to the target device
        track_mask_labels = [x.to(device=device) for x in track_mask_labels]
        track_mask_idxes = [(coords[0].to(device=device), coords[1].to(device=device)) for coords in track_mask_idxes]

        masks = torch.zeros(len(track_mask_idxes), image_height, image_width, dtype=torch.long, device=device)

        for t in range(len(track_mask_idxes)):
            mask_t = torch.zeros(mask_height, mask_width, dtype=torch.long, device=device)
            mask_t[track_mask_idxes[t]] = track_mask_labels[t]

            mask_t = torch.stack([mask_t == ii for ii in instances_to_keep], 0)

            # to obtain the mask in the original image dims:
            # 1. up-sample mask to network input size
            # 2. remove zero padding from right/bottom
            # 3. resize to original image dims

            mask_t = mask_t.unsqueeze(0).float()
            if not self.upscaled_inputs:
                mask_t = F.interpolate(mask_t, scale_factor=mask_scale, mode='bilinear', align_corners=False)

            # get resized network input dimensions (without zero padding)
            resized_mask_width, resized_mask_height, _ = compute_resize_params_2(
                (image_width, image_height), cfg.INPUT.MIN_DIM, cfg.INPUT.MAX_DIM)

            try:
                assert mask_t.shape[3] >= resized_mask_width
                assert mask_t.shape[2] >= resized_mask_height
            except AssertionError as _:
                raise RuntimeError("Network input dims without padding {} should be <= padded dims".format(
                    (resized_mask_width, resized_mask_height), tuple(mask_t.shape[-2:])))

            # remove extra zeros
            mask_t = mask_t[:, :, :resized_mask_height, :resized_mask_width]

            # resize to original image dims
            mask_t = (F.interpolate(mask_t, (image_height, image_width), mode='bilinear', align_corners=False) > 0.5)
            mask_t = mask_t.byte().squeeze(0)
            
            masks[t] = mask_t

        return masks
        

    def forward(self, image_seqs: ImageList, interaction_seqs: list, targets: list):
        """
        Do a forward pass on a N-length batch of sequences of images and targets.
        :param image_seqs: ImageList
        :param interaction_seqs: list(BinaryMaskSequenceList)
        :param targets: List (length N) of dicts, each containing a 'masks' field containing a tensor of
        shape (I, T, H, W) (I is the number of instances)
        :return: dict(
                        ModelOutput.INFERENCE -> dict(
                            ModelOutput.EMBEDDINGS -> ,
                            ModelOutput.SEMSEG_MASKS ->
                            ModelOutput.MASKS -> list(list(tensor(T, H, W))) N-length list with each element being an I-length list
                        )
                )
        """
        # Run model
        model_output, sequences = self.model_forward(image_seqs, interaction_seqs, targets)
        
        # Do mask postprocessing
        masks = []
        for i, output in enumerate(sequences):
            fg_masks, multiclass_masks = output['fg_masks'], output['multiclass_masks']
            if torch.is_tensor(fg_masks):
                print("Obtaining foreground mask from model's foreground mask output")
                fg_masks = (fg_masks > 0.5).byte()  # [T, H, W]
            else:
                print("Obtaining foreground mask by thresholding seediness map at {}".format(self.seediness_fg_threshold))
                fg_masks = self.get_fg_masks_from_seediness(output)

            # Process embedding ouputs into mask tubes using TrackGenerator/OnlineChainer/SequentialClustering
            instance_masks = self.cluster_sequence(output['embeddings'], fg_masks, multiclass_masks, 
                max_tracks=8, original_dims=image_seqs.original_image_sizes[i])
            masks.append(instance_masks)
            
        
        model_output[ModelOutput.INFERENCE][ModelOutput.MASKS] = masks
        return output
    

    @torch.no_grad()
    def get_semseg_masks(self, semseg_logits):
        """
        :param semseg_logits: list(tuple(tensor, int))
        :return: tensor(T, C, H, W) or tensor(T, H, W)
        """
        fg_masks, multiclass_masks = [], []
        if self._model.semseg_head is None:
            return fg_masks, multiclass_masks

        device = "cuda:0" if self.semseg_generation_on_gpu else "cpu"
        semseg_logits = torch.cat([(logits.to(device=device) / float(num_entries)) for logits, num_entries in semseg_logits], 0)

        if semseg_logits.shape[1] > 2:
            # multi-class segmentation: first N-1 channels correspond to logits for N-1 classes and the Nth channels is
            # a fg/bg mask
            multiclass_logits, fg_logits = semseg_logits.split((semseg_logits.shape[1] - 1, 1), dim=1)

            if self.semseg_output_type == "logits":
                multiclass_masks.append(multiclass_logits)
            elif self.semseg_output_type == "probs":
                multiclass_masks.append(F.softmax(multiclass_logits, dim=1))
            elif self.semseg_output_type == "argmax":
                multiclass_masks.append(multiclass_logits.argmax(dim=1))

            fg_masks.append(fg_logits.squeeze(1).sigmoid())

        else:
            # only fg/bg segmentation: the 2 channels correspond to bg and fg logits, respectively
            fg_masks.append(F.softmax(semseg_logits, dim=1)[:, 1])

        fg_masks = torch.cat(fg_masks)
        if multiclass_masks:
            multiclass_masks = torch.cat(multiclass_masks)

        return fg_masks.cpu(), multiclass_masks.cpu()


    def get_fg_masks_from_seediness(self, embeddings):
        seediness_scores = defaultdict(lambda: [0., 0.])

        for subseq_frames, _, _, subseq_seediness in embeddings:
            subseq_seediness = subseq_seediness.cuda().squeeze(0)
            for i, t in enumerate(subseq_frames):
                seediness_scores[t][0] += subseq_seediness[i]
                seediness_scores[t][1] += 1.

        fg_masks = [(seediness_scores[t][0] / seediness_scores[t][1]) for t in sorted(seediness_scores.keys())]
        return (torch.stack(fg_masks, 0) > self.seediness_fg_threshold)
    

    def create_clusterer(self):
        _cfg = cfg.CLUSTERING
        return SequentialClustering(primary_prob_thresh=_cfg.PRIMARY_PROB_THRESHOLD,
                                    secondary_prob_thresh=_cfg.SECONDARY_PROB_THRESHOLD,
                                    min_seediness_prob=_cfg.MIN_SEEDINESS_PROB,
                                    n_free_dims=get_nb_free_dims(cfg.MODEL.EMBEDDING_DIM_MODE),
                                    free_dim_stds=cfg.TRAINING.LOSSES.EMBEDDING.FREE_DIM_STDS,
                                    device=self.clustering_device)

if __name__ == '__main__':
    model = InteractiveModel()