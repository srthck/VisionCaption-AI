import os
import streamlit as st
import logging

from models.feature_extractor import FeatureExtractor
from models.caption_model import CaptionModel
from utils.image_processing import preprocess_image
from utils.caption_generator import generate_caption, clean_caption
import streamlit.components.v1 as components

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants - Hardened for Cloud Execution Contexts
from pathlib import Path

# .resolve() cuts through symlinks. .parent gets the exact folder of app.py.
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.h5"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.pkl"
MAX_CAPTION_LENGTH = 35

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Visionary - AI Image Captioning",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .title-text {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #1E1E1E;
        text-align: left;
        margin-bottom: 0px;
    }
    .subtitle-text {
        font-family: 'Inter', sans-serif;
        color: #6B7280;
        text-align: left;
        margin-bottom: 2rem;
    }
    .caption-box {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
        margin-top: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_feature_extractor():
    """Loads the VGG16 model only once and caches it."""
    return FeatureExtractor()

@st.cache_resource(show_spinner=False)
def load_captioning_model():
    """Loads the LSTM and Tokenizer only once and caches them."""
    return CaptionModel(MODEL_PATH, TOKENIZER_PATH)

def main():
    st.markdown("<h1 class='title-text'>Visionary AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>CNN-LSTM based Image Captioning System</p>", unsafe_allow_html=True)
    
    # Initialize Models
    with st.spinner("Loading deep learning models..."):
        try:
            feature_extractor = load_feature_extractor()
            caption_model = load_captioning_model()
        except Exception as e:
            st.error(f"System Error: Could not initialize AI models. {str(e)}")
            st.stop()

    # Upload Section
    st.markdown("### Upload an Image")
    uploaded_image = st.file_uploader(
        "Choose an image (JPG, JPEG, PNG)...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image for the best captioning results."
    )

    if uploaded_image is not None:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            
        with col2:
            st.markdown("### Analysis & Results")
            with st.spinner("Analyzing image features and generating caption..."):
                try:
                    # Reset file pointer to ensure Pillow can read it after st.image
                    uploaded_image.seek(0)
                    
                    # 1. Preprocess
                    processed_image = preprocess_image(uploaded_image)
                    
                    # 2. Extract Features
                    image_features = feature_extractor.extract(processed_image)
                    
                    # 3. Generate Caption
                    raw_caption = generate_caption(
                        caption_model=caption_model, 
                        image_features=image_features, 
                        max_length=MAX_CAPTION_LENGTH
                    )
                    
                    # 4. Clean Caption
                    final_caption = clean_caption(raw_caption)
                    
                    # Display Caption
                    st.markdown(f"<div class='caption-box'>\"{final_caption}\"</div>", unsafe_allow_html=True)
                    
                    # 5. Client-Side Browser Narration (Web Speech API)
                    st.markdown("**Audio Playback:**")
                    
                    # Escape caption for safe JavaScript injection
                    safe_caption = final_caption.replace("'", "\\'").replace('"', '\\"')
                    
                    js_code = f"""
                    <div style="display: flex; justify-content: flex-start;">
                        <button onclick="speakCaption()" style="background-color: #3B82F6; color: white; padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer; font-family: 'Inter', sans-serif; font-size: 15px; font-weight: 600; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: background-color 0.3s;">
                            🔊 Play Narration
                        </button>
                    </div>
                    
                    <script>
                        function speakCaption() {{
                            if ('speechSynthesis' in window) {{
                                window.speechSynthesis.cancel(); // Stop any currently playing audio
                                var msg = new SpeechSynthesisUtterance("{safe_caption}");
                                msg.lang = "en-US";
                                msg.rate = 1.0;
                                window.speechSynthesis.speak(msg);
                            }}
                        }}
                        
                        // Auto-play narration immediately when iframe renders
                        setTimeout(speakCaption, 200);
                    </script>
                    """
                    
                    components.html(js_code, height=70)
                        
                except Exception as e:
                    logger.error(f"Inference error: {str(e)}")
                    st.error(f"An error occurred during caption generation: {str(e)}")

if __name__ == "__main__":
    main()
