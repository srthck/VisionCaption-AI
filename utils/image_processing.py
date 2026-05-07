import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

def preprocess_image(uploaded_file, target_size=(224, 224)) -> np.ndarray:
    """
    Reads an uploaded image file, resizes it, and preprocesses it for VGG16.
    
    Args:
        uploaded_file: Streamlit UploadedFile object or file path.
        target_size: Tuple representing (width, height) expected by VGG16.
        
    Returns:
        np.ndarray: Preprocessed image tensor of shape (1, 224, 224, 3)
    """
    try:
        # Load the image ensuring RGB format
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize(target_size)
        
        # Convert to numpy array
        img_array = img_to_array(image)
        
        # Expand dimensions to create a batch of 1
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply VGG16 specific preprocessing (mean subtraction, BGR conversion)
        processed_image = preprocess_input(img_array)
        
        return processed_image
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")
