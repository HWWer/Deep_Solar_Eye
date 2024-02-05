from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input

class ResNetModels:
    def __init__(self, model_name='ResNet50', input_shape=(224, 224, 3), pretrained=True):
        """
        Initializes a ResNet model with the option for ResNet50, ResNet101, or ResNet152.

        Parameters:
        - model_name: str, name of the ResNet model ('ResNet50', 'ResNet101', 'ResNet152').
        - input_shape: tuple, the shape of input images.
        - pretrained: bool, whether to load pretrained weights ('imagenet').
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.pretrained = 'imagenet' if pretrained else None
        self.model = self._get_model()

    def _get_model(self):
        """
        Retrieves the specified ResNet model.
        """
        models = {
            'ResNet50': ResNet50,
            'ResNet101': ResNet101,
            'ResNet152': ResNet152
        }
        if self.model_name in models:
            return models[self.model_name](include_top=False, weights=self.pretrained, input_shape=self.input_shape)
        else:
            raise ValueError(f"Unsupported ResNet model: {self.model_name}")

    def get_model(self):
        """
        Accessor for the model.
        """
        return self.model

def regression_ResNet(model_name='ResNet50', input_shape=(224, 224, 3), num_units=512, pretrained=True):
    """
    Creates a ResNet model adapted for regression, allowing choice of ResNet variant.

    Parameters:
    - model_name: str, specific ResNet model ('ResNet50', 'ResNet101', 'ResNet152').
    - input_shape: tuple, shape of input images.
    - num_units: int, number of units in the dense layer.
    - pretrained: bool, whether to initialize with pretrained weights.

    Returns:
    - A TensorFlow Keras Model adapted for regression.
    """
    # Initialize the ResNet model
    model_wrapper = ResNetModels(model_name=model_name, input_shape=input_shape, pretrained=pretrained)
    base_model = model_wrapper.get_model()

    # Make the base model non-trainable
    base_model.trainable = False

    # Build the model
    input_tensor = Input(shape=input_shape)
    x = base_model(input_tensor)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_units, activation='relu')(x)
    output_layer = Dense(1, activation='linear')(x)
    model = Model(inputs=input_tensor, outputs=output_layer)

    return model
