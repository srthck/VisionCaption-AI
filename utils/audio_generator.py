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
        bytes: Raw MP3 audio data bytes, or None if failed.
    """
    try:
        voice = "en-US-AriaNeural"
        communicate = edge_tts.Communicate(text, voice)
        
        audio_data = bytearray()
        
        async def fetch_audio():
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
                    
        # Safely run asyncio in Streamlit/Threaded environments
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(fetch_audio())
        loop.close()
        
        logger.info("Audio synthesized successfully (in-memory bytes)")
        return bytes(audio_data)
    except Exception as e:
        import streamlit as st
        logger.error(f"Failed to synthesize audio: {str(e)}", exc_info=True)
        st.error(f"TTS Debug Error: {e}")
        return None
