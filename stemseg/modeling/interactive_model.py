from collections import defaultdict, namedtuple

from stemseg.config import cfg

from stemseg.data import InferenceImageLoader
from stemseg.data.inference_image_loader import collate_fn
from stemseg.inference.online_chainer import OnlineChainer, masks_to_coord_list
from stemseg.inference.clusterers import SequentialClustering
from stemseg.modeling.embedding_utils import get_nb_free_dims
from stemseg.modeling.model_builder import build_model
from stemseg.structures.image_list import ImageList
from stemseg.structures.mask import BinaryMaskSequenceList
from stemseg.utils.constants import ModelOutput
from stemseg.utils.timer import Timer

from torch.utils.data import DataLoader
from tqdm import tqdm

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractiveTrackGenerator(object):
    def __init__(self, sequences, dataset_name, output_generator, output_dir, model_ckpt_path, max_tracks, preload_images,
                 resize_scale, semseg_averaging_on_gpu, **kwargs):
        self.sequences = sequences
        self.dataset_name = dataset_name

        self.output_generator = output_generator
        if self.dataset_name == "kittimots":
            semseg_output_type = "argmax"
        elif self.dataset_name == "ytvis":
            semseg_output_type = "logits"
        else:
            semseg_output_type = None

        self.model = InferenceModel(model_ckpt_path, semseg_output_type=semseg_output_type, preload_images=preload_images,
                                    resize_scale=resize_scale, semseg_generation_on_gpu=semseg_averaging_on_gpu).cuda()

        self.resize_scale = resize_scale
        self.vis_output_dir = os.path.join(output_dir, "vis")
        self.embeddings_output_dir = os.path.join(output_dir, "embeddings")
        self.max_tracks = max_tracks

        self.save_vis = kwargs.get("save_vis", False)

        self.seediness_fg_threshold = kwargs.get("seediness_thresh", 0.25)
        self.ignore_fg_masks = kwargs.get("ignore_fg_masks", False)
        self.frame_overlap = kwargs.get("frame_overlap", -1)
        self.clustering_device = kwargs.get("clustering_device", "cuda:0")

        self.chainer = OnlineChainer(self.create_clusterer(), embedding_resize_factor=resize_scale)
        self.total_frames_processed = 0.

    def create_clusterer(self):
        _cfg = cfg.CLUSTERING
        return SequentialClustering(primary_prob_thresh=_cfg.PRIMARY_PROB_THRESHOLD,
                                    secondary_prob_thresh=_cfg.SECONDARY_PROB_THRESHOLD,
                                    min_seediness_prob=_cfg.MIN_SEEDINESS_PROB,
                                    n_free_dims=get_nb_free_dims(cfg.MODEL.EMBEDDING_DIM_MODE),
                                    free_dim_stds=cfg.TRAINING.LOSSES.EMBEDDING.FREE_DIM_STDS,
                                    device=self.clustering_device)

    def get_fg_masks_from_seediness(self, inference_output):
        seediness_scores = defaultdict(lambda: [0., 0.])

        for subseq_frames, _, _, subseq_seediness in inference_output['embeddings']:
            subseq_seediness = subseq_seediness.cuda().squeeze(0)
            for i, t in enumerate(subseq_frames):
                seediness_scores[t][0] += subseq_seediness[i]
                seediness_scores[t][1] += 1.

        fg_masks = [(seediness_scores[t][0] / seediness_scores[t][1]) for t in sorted(seediness_scores.keys())]
        return (torch.stack(fg_masks, 0) > self.seediness_fg_threshold).byte().cpu()

    def start(self, seqs_to_process):
        # iou_stats_container = IoUStatisticsContainer()

        if not isinstance(self.max_tracks, (list, tuple)):
            self.max_tracks = [self.max_tracks] * len(self.sequences)

        for i in range(len(self.sequences)):
            sequence = self.sequences[i]
            if seqs_to_process and str(sequence.seq_id) not in seqs_to_process:
                continue

            print("Performing inference for sequence {}/{}".format(i + 1, len(self.sequences)))
            self.process_sequence(sequence, self.max_tracks[i])

        print("----------------------------------------------------")
        print("Model inference speed: {:.3f} fps".format(self.total_frames_processed / Timer.get_duration("inference")))
        print("Clustering and postprocessing speed: {:.3f} fps".format(self.total_frames_processed / Timer.get_duration("postprocessing")))
        print("Overall speed: {:.3f} fps".format(self.total_frames_processed / Timer.get_durations_sum()))
        print("----------------------------------------------------")

    def process_sequence(self, sequence, max_tracks):
        embeddings, fg_masks, multiclass_masks = self.do_inference(sequence)

        self.do_clustering(sequence, embeddings, fg_masks, multiclass_masks, max_tracks)

        self.total_frames_processed += len(sequence)

    @Timer.log_duration("inference")
    def do_inference(self, sequence):
        subseq_idxes, _ = get_subsequence_frames(
            len(sequence), cfg.INPUT.NUM_FRAMES, self.dataset_name, self.frame_overlap)

        image_paths = [os.path.join(sequence.base_dir, path) for path in sequence.image_paths]
        inference_output = self.model(image_paths, subseq_idxes)

        fg_masks, multiclass_masks = inference_output['fg_masks'], inference_output['multiclass_masks']

        if torch.is_tensor(fg_masks):
            print("Obtaining foreground mask from model's foreground mask output")
            fg_masks = (fg_masks > 0.5).byte()  # [T, H, W]
        else:
            print("Obtaining foreground mask by thresholding seediness map at {}".format(self.seediness_fg_threshold))
            fg_masks = self.get_fg_masks_from_seediness(inference_output)

        return inference_output["embeddings"], fg_masks, multiclass_masks

    @Timer.log_duration("postprocessing")
    def do_clustering(self, sequence, all_embeddings, fg_masks, multiclass_masks, max_tracks):
        subseq_dicts = []

        for i, (subseq_frames, embeddings, bandwidths, seediness) in tqdm(enumerate(all_embeddings), total=len(all_embeddings)):
            subseq_dicts.append({
                "frames": subseq_frames,
                "embeddings": embeddings,
                "bandwidths": bandwidths,
                "seediness": seediness,
            })

        (track_labels, instance_pt_counts, instance_lifetimes), framewise_mask_idxes, subseq_labels_list, \
            fg_embeddings, subseq_clustering_meta_info = self.chainer.process(
            fg_masks, subseq_dicts)

        self.output_generator.process_sequence(
            sequence, framewise_mask_idxes, track_labels, instance_pt_counts, instance_lifetimes, multiclass_masks,
            fg_masks.shape[-2:], 4.0, max_tracks, device=self.clustering_device
        )


def configure_directories(args):
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(args.model_path), "inference")

    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(args.model_path), output_dir)

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_cfg(args):
    cfg_file = os.path.join(os.path.dirname(args.model_path), 'config.yaml')
    if not os.path.exists(cfg_file):
        dataset_cfgs = {
            "davis": "davis_2.yaml",
            "ytvis": "youtube_vis.yaml",
            "kittimots": "kitti_mots_2.yaml"
        }
        assert args.dataset in dataset_cfgs, \
            "Invalid '--dataset' argument. Should be either 'davis', 'ytvis' or 'kittimots'"
        cfg_file = os.path.join(RepoPaths.configs_dir(), dataset_cfgs[args.dataset])

    print("Loading config from {}".format(cfg_file))
    cfg.merge_from_file(cfg_file)


def configure_input_dims(args):
    if not args.min_dim and not args.max_dim:
        return

    elif args.min_dim and args.max_dim:
        assert args.min_dim > 0
        assert args.max_dim > 0
        cfg.INPUT.update_param("MIN_DIM", args.min_dim)
        cfg.INPUT.update_param("MAX_DIM", args.max_dim)

    elif args.min_dim and not args.max_dim:
        assert args.min_dim > 0
        dim_ratio = float(cfg.INPUT.MAX_DIM) / float(cfg.INPUT.MIN_DIM)
        cfg.INPUT.update_param("MIN_DIM", args.min_dim)
        cfg.INPUT.update_param("MAX_DIM", int(round(args.min_dim * dim_ratio)))

    elif not args.min_dim and args.max_dim:
        assert args.max_dim > 0
        dim_ratio = float(cfg.INPUT.MAX_DIM) / float(cfg.INPUT.MIN_DIM)
        cfg.INPUT.update_param("MIN_DIM", int(round(args.max_dim / dim_ratio)))
        cfg.INPUT.update_param("MAX_DIM", args.max_dim)

    else:
        raise ValueError("Should never be here")

    print("Network input image dimension limits: {}, {}".format(cfg.INPUT.MIN_DIM, cfg.INPUT.MAX_DIM))


class InteractiveModel(nn.Module):
    def __init__(self):
        self._model = build_model(restore_pretrained_backbone_wts=True, logger=None)
        self.chainer = OnlineChainer(self.create_clusterer(), embedding_resize_factor=resize_scale)

    def forward(self, image_seqs: ImageList, interaction_seqs: list(BinaryMaskSequenceList), targets: list):
        """
        Do a forward pass on a batch of sequences of images and targets.
        :param image_seqs: ImageList
        :param interaction_seqs: list(BinaryMaskSequenceList)
        :param targets: List (length N) of dicts, each containing a 'masks' field containing a tensor of
        shape (I (instances), T, H, W)
        :return dict()
        """
        
        # TODO: pass in interactions
        output = self._model.forward(image_seqs, targets)

        # TODO: process embedding ouputs into mask tubes using TrackGenerator/OnlineChainer/SequentialClustering

        fg_masks = self.get_fg_masks_from_seediness(output[ModelOutput.INFERENCE][ModelOutput.EMBEDDINGS])
        
        output[ModelOutput.INFERENCE][ModelOutput.MASKS] = masks
        return output
        

    def create_clusterer(self):
        _cfg = cfg.CLUSTERING
        return SequentialClustering(primary_prob_thresh=_cfg.PRIMARY_PROB_THRESHOLD,
                                    secondary_prob_thresh=_cfg.SECONDARY_PROB_THRESHOLD,
                                    min_seediness_prob=_cfg.MIN_SEEDINESS_PROB,
                                    n_free_dims=get_nb_free_dims(cfg.MODEL.EMBEDDING_DIM_MODE),
                                    free_dim_stds=cfg.TRAINING.LOSSES.EMBEDDING.FREE_DIM_STDS,
                                    device=self.clustering_device)

    def get_fg_masks_from_seediness(self, embeddings):
        seediness_scores = defaultdict(lambda: [0., 0.])

        for subseq_frames, _, _, subseq_seediness in embeddings:
            subseq_seediness = subseq_seediness.cuda().squeeze(0)
            for i, t in enumerate(subseq_frames):
                seediness_scores[t][0] += subseq_seediness[i]
                seediness_scores[t][1] += 1.

        fg_masks = [(seediness_scores[t][0] / seediness_scores[t][1]) for t in sorted(seediness_scores.keys())]
        return (torch.stack(fg_masks, 0) > self.seediness_fg_threshold)

if __name__ == '__main__':
    model = InteractiveModel()