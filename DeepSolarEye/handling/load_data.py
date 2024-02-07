from tensorflow.keras.utils import image_dataset_from_directory
import os


def load_image_from_directory(image_size=(224, 224), batch_size=None):

    directory_path=os.path.dirname(os.path.dirname(os.path.abspath("")))
    image_dataset = image_dataset_from_directory(
        directory=directory_path,
        label_mode=None,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
        )


    return image_dataset
