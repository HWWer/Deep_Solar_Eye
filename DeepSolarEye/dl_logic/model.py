from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152

class ResNetModels:
    def __init__(self, model_name='ResNet50', pretrained=True):
        """
        Initializes a ResNet model. With ResNet50 as default

        Parameters:
        - model_name: str, name of the ResNet model to initialize ('ResNet50', 'ResNet101', 'ResNet152').
        - pretrained: bool, whether to load pretrained weights ('imagenet' or None).
        """
        self.model_name = model_name
        self.pretrained = 'imagenet' if pretrained else None
        self.model = self._get_model()

    def _get_model(self):
        """
        Method to get the specified ResNet model: ResNet50, ResNet101, ResNet152
        """
        if self.model_name == 'ResNet50':
            return ResNet50(weights=self.pretrained)
        elif self.model_name == 'ResNet101':
            return ResNet101(weights=self.pretrained)
        elif self.model_name == 'ResNet152':
            return ResNet152(weights=self.pretrained)
        else:
            raise ValueError("Unsupported ResNet model: {}".format(self.model_name))

    def get_model(self):
        """
        Public method to access the model.
        """
        return self.model
