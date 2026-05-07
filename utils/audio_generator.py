import os
import uuid
import logging
from gtts import gTTS

logger = logging.getLogger(__name__)

def synthesize_speech(text: str, output_dir: str = "assets") -> str:
    """
    Converts text to speech using Google TTS and saves it as an MP3 file.
    Uses UUIDs to prevent file clashing in a concurrent web environment.
    
    Args:
        text: The text string to convert to audio.
        output_dir: Directory where the audio file will be saved.
        
    Returns:
        str: Path to the generated audio file, or None if failed.
    """
    try:
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate a unique filename
        filename = f"caption_audio_{uuid.uuid4().hex[:8]}.mp3"
        filepath = os.path.join(output_dir, filename)
        
        # Synthesize and save
        tts = gTTS(text, lang='en')
        tts.save(filepath)
        
        logger.info(f"Audio synthesized successfully at {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to synthesize audio: {str(e)}")
        return None
