import logging
import sys
import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
from davisinteractive.metrics.jaccard import batched_jaccard

from stemseg.data.generic_video_dataset_parser import parse_generic_video_dataset, GenericVideoSequence
from stemseg.data.interactive_video_dataset_parser import InteractiveVideoSequence
from stemseg.data import DavisUnsupervisedPaths as DavisPaths

def get_sequence(sequences, name) -> GenericVideoSequence:
    for seq in sequences:
        if seq.id == name:
            return seq
    return None

def load_preds_as_np(paths: list):
    masks = []
    for path in paths:
        img = Image.open(path).convert('P') # Palettised image
        mask = np.array(img.getchannel(0))
        mask = np.where(mask > 0, 1, 0)
        masks.append(mask)
    masks = np.stack(masks, axis=0) # (T, H, W)
    return masks

def get_instance_gt(all_gt: dict, oid: str, iid: int):
    masks = []
    for step in all_gt[oid]:
        masks.append(step[iid])
    masks = np.stack(masks, axis=0) # (T, H, W)
    return masks

def main(args):
    if args.dataset == "davis":
        sequences, _ = parse_generic_video_dataset(DavisPaths.trainval_base_dir(), DavisPaths.val_vds_file())
    else:
        raise ValueError("Invalid dataset name {} provided".format(args.dataset))

    # Load ground truth masks for all predicted sequences
    all_gt = dict() # dict(str -> list(list(ndarray) (length I)) (length T))
    print("Loading ground truth masks...")
    for seq_id in tqdm(list(filter(lambda p: os.path.isdir(os.path.join(args.results_dir, p)), sorted(os.listdir(args.results_dir))))):
        oid, iid = InteractiveVideoSequence.parse_split_id(seq_id)
        ori_seq = get_sequence(sequences, oid)
        assert ori_seq is not None, f"Could not find ground truth sequence {oid} for prediction {seq_id}"
        if oid not in all_gt:
            all_gt[oid] = ori_seq.load_masks()
    
    # Perform mIOU computation for each sequence
    results = dict() # dict(str -> dict(int -> int) (length I)) (length N)
    print("Loading predictions and computing jaccard indices...")
    for seq_id in tqdm(list(filter(lambda p: os.path.isdir(os.path.join(args.results_dir, p)), sorted(os.listdir(args.results_dir))))):
        oid, iid = InteractiveVideoSequence.parse_split_id(seq_id)
        pred_paths = sorted([os.path.join(args.results_dir, seq_id, name) for name in os.listdir(os.path.join(args.results_dir, seq_id))])
        y_pred = load_preds_as_np(pred_paths)
        y_true = get_instance_gt(all_gt, oid, iid)
        scores = batched_jaccard(y_true, y_pred, False, 1)
        miou = np.mean(scores)
        sub_results = results.setdefault(oid, dict())
        sub_results[iid] = miou
    
    # Get grand mean mIOU across all sequences
    num_instances = 0
    summed_miou = 0
    summed_best_miou = 0
    for sub_results in results.values():
        best = 0
        for miou in sub_results.values():
            if miou > best:
                best = miou
            summed_miou += miou
            num_instances += 1
        summed_best_miou += best
    grand_miou = summed_miou / num_instances
    best_miou = summed_best_miou / len(results)

    # Set up logging to both console and file
    file = os.path.join((args.output_dir if args.output_dir else args.results_dir), 'metrics.txt')
    targets = logging.StreamHandler(sys.stdout), logging.FileHandler(file, mode='w')
    logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=targets)

    # Print results
    logging.info("----- Sequence result breakdown -----")
    for oid, sub_results in results.items():
        logging.info(oid)
        seq_miou = 0
        for iid, miou in sub_results.items():
            logging.info(f"    {iid}: {miou}")
            seq_miou += miou
        seq_miou /= len(sub_results)
        logging.info(f"    Sequence average: {seq_miou}")
    logging.info("")
    logging.info("----- Overall results -----")
    logging.info(f"mIOU: {grand_miou}")
    logging.info(f"mIOU for best instance per sequence: {best_miou}")

    print(f"\nResults written to file {file}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('results_dir')

    parser.add_argument('--dataset',    '-d',              required=True)
    parser.add_argument('--output_dir',                    required=False)

    main(parser.parse_args())