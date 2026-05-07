import logging
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Encapsulates the loading and inference of the VGG16 model for image feature extraction.
    Uses the layer just before the final classification layer.
    """
    def __init__(self):
        try:
            logger.info("Loading VGG16 feature extractor...")
            base_model = VGG16()
            # Extract features from the second-to-last layer
            self.model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
            logger.info("VGG16 model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load VGG16 model: {str(e)}")
            raise RuntimeError(f"Could not initialize feature extractor: {str(e)}")

    def extract(self, image_tensor):
        """
        Extracts features from an image tensor.
        
        Args:
            image_tensor (np.ndarray): Preprocessed image tensor of shape (1, 224, 224, 3)
            
        Returns:
            np.ndarray: Extracted feature vector
        """
        try:
            features = self.model.predict(image_tensor, verbose=0)
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise ValueError(f"Feature extraction failed: {str(e)}")
