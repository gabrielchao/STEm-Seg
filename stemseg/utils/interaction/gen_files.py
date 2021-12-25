import os
import cv2
import numpy as np

from davisinteractive.dataset import Davis

from stemseg.utils.interaction.gen_interaction import get_center_click_maps, get_blank_interaction_maps

if __name__ == '__main__':
    """
    Generate and save guidance maps for all frames of DAVIS val
    """
    root_path = '/home/gabriel/datasets/DAVIS'
    custom_dirname = 'CustomGuidance/480p'
    davis = Davis(root_path)
    for sequence in davis.sets['trainval']:
        print(f'Processing sequence: {sequence}...')
        annotations = davis.load_annotations(sequence) # This loads the folder 'Annotations' NOT 'Annotations_unsupervised'!
        labels = set(np.unique(annotations)) - {0} # Remove background label
        instance_masks = []
        for label in labels:
            instance_masks.append(np.where(annotations == label, 1, 0))
        instance_masks = np.stack(instance_masks, axis=0)
        guidance_tube = get_center_click_maps(get_blank_interaction_maps(instance_masks.shape), instance_masks) # (I, T, 2, H, W)
        os.makedirs(os.path.join(root_path, custom_dirname, sequence), exist_ok=True)
        for i in range(guidance_tube.shape[0]):
            try:
                os.mkdir(os.path.join(root_path, custom_dirname, sequence, f'instance_{i}'))
            except FileExistsError:
                pass
            for t in range(guidance_tube.shape[1]):
                mask = guidance_tube[i, t, 0]
                cv2.imwrite(os.path.join(root_path, custom_dirname, sequence, f'instance_{i}', f'{t:05}.png'), mask*255)
