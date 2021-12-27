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
        # This loads the folder 'Annotations' NOT 'Annotations_unsupervised'!
        # Hack: just manually temporarily rename the folder before running
        annotations = davis.load_annotations(sequence) 
        labels = set(np.unique(annotations)) - {0} # Remove background label
        instance_masks = []
        for label in labels:
            instance_masks.append(np.where(annotations == label, 1, 0))
        instance_masks = np.stack(instance_masks, axis=0)
        guidance_tube = get_center_click_maps(get_blank_interaction_maps(instance_masks.shape), instance_masks) # (I, T, 2, H, W)
        os.makedirs(os.path.join(root_path, custom_dirname, sequence), exist_ok=True)
        for i in range(guidance_tube.shape[0]):
            # Save positive tube
            os.makedirs(os.path.join(root_path, custom_dirname, sequence, f'instance_{i}', 'positive'), exist_ok=True)
            for t in range(guidance_tube.shape[1]):
                pos_mask = guidance_tube[i, t, 0]
                cv2.imwrite(os.path.join(root_path, custom_dirname, sequence, f'instance_{i}', 'positive', f'{t:05}.png'), pos_mask*255)
            # Save negative tube
            os.makedirs(os.path.join(root_path, custom_dirname, sequence, f'instance_{i}', 'negative'), exist_ok=True)
            for t in range(guidance_tube.shape[1]):
                neg_mask = guidance_tube[i, t, 1]
                cv2.imwrite(os.path.join(root_path, custom_dirname, sequence, f'instance_{i}', 'negative', f'{t:05}.png'), neg_mask*255)
