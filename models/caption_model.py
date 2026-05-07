import pickle
import logging
from pathlib import Path
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

class CaptionModel:
    """
    Encapsulates the trained LSTM caption generation model and its tokenizer.
    """
    def __init__(self, model_path: Path, tokenizer_path: Path):
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.model = None
        self.tokenizer = None
        
        self._load_resources()

    def _load_resources(self):
        """Loads the LSTM model and Tokenizer from disk safely."""
        # 1. Gracefully fail if assets are entirely missing from the deployment environment
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Deployment Error: Tokenizer missing. Expected it at: {self.tokenizer_path.absolute()}\n"
                "If deploying to Streamlit Cloud, verify the file wasn't blocked by .gitignore or GitHub size limits."
            )
            
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Deployment Error: Model missing. Expected it at: {self.model_path.absolute()}\n"
                "If deploying to Streamlit Cloud, ensure Git LFS successfully pushed the .h5 file."
            )

        # 2. Load Tokenizer
        try:
            logger.info(f"Loading tokenizer from {self.tokenizer_path.name}...")
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise FileNotFoundError(f"Missing or corrupted tokenizer file at {self.tokenizer_path}.")

        # Load Model
        try:
            logger.info(f"Loading LSTM model from {self.model_path}...")
            self.model = load_model(self.model_path)
            logger.info("LSTM model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {str(e)}")
            raise FileNotFoundError(f"Missing or corrupted model file at {self.model_path}.")

    def get_word_from_index(self, index: int) -> str:
        """
        Retrieves the string word mapped to the given index.
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer is not loaded.")
        
        # Optimize: reverse indexing might be slow if done iteratively, but tokenizer.index_word usually exists
        # Falling back to items() iteration if index_word is not populated
        if hasattr(self.tokenizer, 'index_word') and index in self.tokenizer.index_word:
            return self.tokenizer.index_word[index]
            
        return next((word for word, idx in self.tokenizer.word_index.items() if idx == index), None)
