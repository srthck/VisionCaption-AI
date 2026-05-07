import os
import asyncio
import tempfile
import logging
import edge_tts

logger = logging.getLogger(__name__)

def synthesize_speech(text: str) -> str:
    """
    Converts text to speech using Microsoft Edge TTS and saves it as an MP3 file.
    Uses tempfile to safely manage files in a concurrent web/Docker environment.
    
    Args:
        text: The text string to convert to audio.
        
    Returns:
        str: Path to the generated temporary audio file, or None if failed.
    """
    try:
        # Generate a secure temporary file
        temp_dir = tempfile.gettempdir()
        temp_fd, filepath = tempfile.mkstemp(suffix=".mp3", dir=temp_dir)
        os.close(temp_fd)
        
        # Configure Edge TTS
        voice = "en-US-AriaNeural"
        communicate = edge_tts.Communicate(text, voice)
        
        # Safely run asyncio in Streamlit/Threaded environments
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(communicate.save(filepath))
        loop.close()
        
        logger.info(f"Audio synthesized successfully at {filepath}")
        return filepath
    except Exception as e:
        import streamlit as st
        logger.error(f"Failed to synthesize audio: {str(e)}", exc_info=True)
        st.error(f"TTS Debug Error: {e}")
        return None
