import os
import random
import numpy as np
import mrcnn.model as modellib
from mrcnn import config
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon
import colorsys
import io


class SolarConfig(config.Config):
  NUM_CLASSES = 1 + 2
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
  def __init__(self):
      super().__init__()
      self.train_images_dir = 'RCNN_Masks/train/images'
      self.NAME = "solar_cfg"
      self.num_train_images = len([name for name in os.listdir(self.train_images_dir) if os.path.isfile(os.path.join(self.train_images_dir, name))])
      self.STEPS_PER_EPOCH = self.num_train_images // self.IMAGES_PER_GPU  # Adjust based on your actual number of images in the training set

class InferenceConfig(SolarConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def mrcnn_load_inference_model(filepath):
    #ipdb.set_trace()
    config = InferenceConfig()
    model_dir = 'mrcnn'
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
    model.load_weights(filepath, by_name=True)
    return model

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None,
                      figsize=(16, 16), figAx=None,
                      show_mask=True, show_bbox=True,
                      show_caption=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    """image copy for furthere analysis"""
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not figAx:
        fig,ax = plt.subplots(1, figsize=figsize)
        auto_show = True
    else:
        fig,ax = figAx

    # Generate random colors
    colors = colors or random_colors(N)

    # print("image_size is {}".format(image.shape))
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                   alpha=0.7, linestyle="dashed",
                                   edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if show_caption:
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            ax.text(x1, y1 + 25, caption,
                    color='w', size=40, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    # Remove background and axes
    ax.set_axis_off()

    # display masked image with no background
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')

    # Remove whitespace padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # save figure to byte stream
    byte_stream = io.BytesIO()
    plt.savefig(byte_stream, format='png')
    byte_stream.seek(0)

    return byte_stream


def make_mrcnn_prediction(image, results):

    class_names = ['background', 'panel', 'soil']

    r = results[0]

    # get indices where class_id is 1 and score is less than 0.95
    filtered_indices = np.where(((r['class_ids'] == 1) & (r['scores'] < 0.90)) |
                                ((r['class_ids'] == 2) & (r['scores'] < 0.80)))[0]


    # Remove the filtered indices from the data
    filtered_data = {
        'rois': np.delete(r['rois'], filtered_indices, axis=0),
        'class_ids': np.delete(r['class_ids'], filtered_indices),
        'scores': np.delete(r['scores'], filtered_indices),
        'masks': np.delete(r['masks'], filtered_indices, axis=2)
    }

    if 1 in filtered_data['class_ids']:

        inferred_img = display_instances(image, filtered_data['rois'], filtered_data['masks'], filtered_data['class_ids'],
                                class_names, filtered_data['scores'])

        return inferred_img

    return False
