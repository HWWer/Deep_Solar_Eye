import os
import json
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model

class SolarDataset(mrcnn.utils.Dataset):
    def load_dataset(self, dataset_dir, dataset_type='train'):
        """
        Load the dataset from the specified directory and dataset type (train, val, test).
        """
        # Register the classes
        self.add_class("dataset", 1, "solar_panel")
        self.add_class("dataset", 2, "soil")

        assert dataset_type in ['train', 'val', 'test'], "dataset_type must be one of: train, val, test"

        base_dir = os.path.join(dataset_dir, dataset_type)

        images_dir = os.path.join(base_dir, 'images')
        annotations_dir = os.path.join(base_dir, 'annots')

        # Iterate through all files in the images directory
        for filename in os.listdir(images_dir):
            # Image ID is the filename without the file extension
            image_id = filename[:-4]

            img_path = os.path.join(images_dir, filename)
            ann_path = os.path.join(annotations_dir, f"{image_id}_bboxes.json")

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def load_mask(self, image_id):
        """
        Generate masks for each instance in the image based on JSON annotations.
        """
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, class_ids = self.extract_boxes(path)
        h, w = info['height'], info['width']
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        for i, box in enumerate(boxes):
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1

        return masks, asarray(class_ids, dtype='int32')

    def extract_boxes(self, filename):
        """
        Extract bounding boxes and class IDs from JSON annotation file.
        """
        with open(filename) as f:
            data = json.load(f)

        boxes = []
        class_ids = []
        for item in data:
            label = item['label']
            if label in ['solar_panel', 'soil']:
                bbox = item['bbox']
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                class_ids.append(self.class_names.index(label))

        return boxes, class_ids

class SolarConfig(mrcnn.config.Config):
    train_images_dir = '/raw_data/RCNN_Masks/train/images'
    num_train_images = len([name for name in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, name))])
    NAME = "solar_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # Background + Solarpanel + Soil
    STEPS_PER_EPOCH = num_train_images // IMAGES_PER_GPU  # Adjust based on your actual number of images in the training set



# Train
train_dataset = SolarDataset()
train_dataset.load_dataset(dataset_dir='Deep_Solar_Eye/raw_data/RCNN_Masks/', dataset_type='train')
train_dataset.prepare()

# Validation
validation_dataset = SolarDataset()
validation_dataset.load_dataset(dataset_dir='Deep_Solar_Eye/raw_data/RCNN_Masks/', dataset_type='val')
validation_dataset.prepare()

# Model Configuration
solar_config = SolarConfig()

# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(mode='training', model_dir='./', config=solar_config)

model.load_weights(filepath='mask_rcnn_coco.h5', by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_dataset,
            val_dataset=validation_dataset,
            learning_rate=solar_config.LEARNING_RATE,
            epochs=1,
            layers='heads')

model_path = 'Solar_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)
