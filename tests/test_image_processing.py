import pytest
import numpy as np
from PIL import Image
import io
from utils.image_processing import preprocess_image

def create_dummy_image(size=(300, 300), format="JPEG"):
    """Helper to create an in-memory dummy image."""
    img = Image.new("RGB", size, color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    return img_byte_arr

def test_preprocess_image_valid_input():
    """Test that a valid image is resized and preprocessed correctly for VGG16."""
    dummy_image = create_dummy_image(size=(500, 500))
    target_size = (224, 224)
    
    result = preprocess_image(dummy_image, target_size=target_size)
    
    # Check shape: (1, 224, 224, 3)
    assert result.shape == (1, *target_size, 3)
    # Check type: should be a numpy array with float values due to preprocess_input
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32

def test_preprocess_image_invalid_input():
    """Test that invalid inputs raise a ValueError."""
    # Pass a random string instead of a file-like object
    with pytest.raises(ValueError) as excinfo:
        preprocess_image("not_an_image.txt")
    
    assert "Failed to process image" in str(excinfo.value)

def test_preprocess_image_rgba_conversion():
    """Test that RGBA images are converted to RGB safely."""
    # Create an RGBA image
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    
    result = preprocess_image(img_byte_arr)
    
    # Even if it was RGBA, output should still be 3 channels for VGG16
    assert result.shape == (1, 224, 224, 3)
