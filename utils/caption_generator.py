import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.caption_model import CaptionModel

def generate_caption(caption_model: CaptionModel, image_features: np.ndarray, max_length: int = 35) -> str:
    """
    Generates a caption sequence token by token using the LSTM model.
    
    Args:
        caption_model: Instance of CaptionModel containing the loaded LSTM and Tokenizer.
        image_features: Extracted features from the VGG16 model.
        max_length: Maximum length of the generated sequence.
        
    Returns:
        str: The generated raw caption sequence including startseq/endseq.
    """
    caption = "startseq"
    
    for _ in range(max_length):
        # Convert current caption text to sequence of integers
        sequence = caption_model.tokenizer.texts_to_sequences([caption])[0]
        # Pad the sequence to match the input shape expected by the model
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict the next word index
        yhat = caption_model.model.predict([image_features, sequence], verbose=0)
        predicted_index = np.argmax(yhat)
        
        # Map index back to word
        predicted_word = caption_model.get_word_from_index(predicted_index)
        
        # Break if we hit an unknown word or endseq
        if predicted_word is None:
            break
            
        caption += " " + predicted_word
        
        if predicted_word == "endseq":
            break
            
    return caption

def clean_caption(raw_caption: str) -> str:
    """
    Cleans the raw caption by removing special tokens and capitalizing the first letter.
    
    Args:
        raw_caption: Raw string containing startseq/endseq.
        
    Returns:
        str: Cleaned human-readable caption.
    """
    cleaned = raw_caption.replace("startseq", "").replace("endseq", "").strip()
    # Replace multiple internal spaces with a single space
    cleaned = " ".join(cleaned.split())
    if cleaned:
        # Capitalize the first letter for professional appearance
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned
