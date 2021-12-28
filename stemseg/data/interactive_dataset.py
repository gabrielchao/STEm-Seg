from torch.utils.data import Dataset, DataLoader

from stemseg.structures.image_list import ImageList

class InteractiveDataset(Dataset):
    """
    A sub-dataset where each item is a single video sequence - instance target pair. 
    (As opposed to a sequence having multiple instance targets.) Can act as an
    adaptor between the standard training data loader and the interactive 
    training loop.
    """
    def __init__(self, image_seqs, targets):
        """
        Initialize an InteractiveDataset.
        :param image_seqs: ImageList (N, T, C, H, W)
        :param targets: tuple(
            dict(
                'masks' -> tensor(I, T, H, W),
                'category_ids' -> tensor(),
                'labels' -> tensor(),
                'ignore_masks' -> tensor(T, H, W)
            )
        ) (length N)
        """
        self.image_seqs = image_seqs
        self.targets = targets
        self.total_instances = sum([len(d['masks']) for d in targets])
    
    def __len__(self):
        return self.total_instances

    def __getitem__(self, idx):
        """
        Get a tuple of sequence, target, (original_width, original_height) representing a single video-instance pair.
        :return tuple(
            tensor(T, C, H, W),
            dict(
                'masks' -> tensor(1, T, H, W),
                'category_ids' -> tensor(),
                'labels' -> tensor(),
                'ignore_masks' -> tensor(T, H, W)
            ),
            tuple(ori_width, ori_height)
        )
        """
        prev_items = 0
        for n, d in enumerate(self.targets):
            cur_idxs = d['masks'].shape[0]
            if prev_items + cur_idxs > idx:
                mask = d['masks'][idx - prev_items]
                sequence = self.image_seqs[n]
                new_dict = d.copy()
                new_dict['masks'] = mask.unsqueeze(0)
                return sequence, new_dict, self.image_seqs.original_image_sizes[n]
            prev_items += cur_idxs

def collate_interactive(batch):
    """
    Collates InteractiveDataset items into batches that match the standard
    model forward parameter types (image_seqs, targets).
    :return tuple(
        ImageList (N, T, C, H, W),
        tuple(
            dict(
                'masks' -> tensor(I, T, H, W),
                'category_ids' -> tensor(),
                'labels' -> tensor(),
                'ignore_masks' -> tensor(T, H, W)
            )
        ) (length N)
    )
    """
    sequence_list, targets, original_image_sizes = zip(*batch)
    image_seqs = ImageList.from_image_sequence_list(sequence_list, original_image_sizes)
    return image_seqs, targets

def create_interactive_data_loader(dataset: InteractiveDataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size,
        shuffle,
        collate_fn=collate_interactive
    )

if __name__ == '__main__':
    import torch
    image_seqs = ImageList.from_image_sequence_list([[torch.zeros((3, 2, 2))]], ((3, 3),))
    targets = ({
        'masks': torch.stack((torch.zeros((2, 2)), torch.ones(2, 2))).unsqueeze(1),
        'category_ids': torch.tensor([1]),
        'labels': torch.tensor([1, 2]),
        'ignore_masks': torch.zeros((1, 2, 2))
    },)
    sub_dataset = InteractiveDataset(image_seqs, targets)
    print(sub_dataset[0][1]['masks'])
    print(sub_dataset[1][1]['masks'])
    