import streamlit as st
import os
from PIL import Image
import tempfile
from modules.frame_detector import FrameDetector
from modules.ocr_processor import OCRProcessor
from modules.tts_generator import TTSGenerator
from config import Config


class MangaAIApp:
    def __init__(self):
        self.config = Config()
        self.frame_detector = FrameDetector()
        self.ocr_processor = OCRProcessor()
        self.tts_generator = TTSGenerator()
        
        # Validate configuration on startup
        self.config_status = self.config.validate_config()
    
    def run(self):
        st.set_page_config(
            page_title="MangAI - Manga Audio Generator",
            page_icon="📚",
            layout="wide"
        )
        
        self._render_header()
        self._render_sidebar()
        self._render_main_content()
        self._render_footer()
    
    def _render_header(self):
        """Render the application header"""
        st.title("📚 MangAI - Manga to Audio")
        st.markdown("Upload a manga page and get audio narration!")
        
        # Show configuration status
        with st.expander("🔧 System Status", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Models")
                for model_type in ["frame", "panel"]:
                    status = "✅" if self.config_status.get(f"model_{model_type}", False) else "❌"
                    st.write(f"{status} {model_type.title()} Detection Model")
                
                st.subheader("Directories")
                status = "✅" if self.config_status.get("audio_output_dir", False) else "❌"
                st.write(f"{status} Audio Output Directory")
            
            with col2:
                st.subheader("Language & Processing")
                st.write("🇺🇸 English Only")
                st.write("📝 Simple Text Processing")
                st.write("🎵 Local TTS Generation")
    
    def _render_sidebar(self):
        """Render the sidebar with settings"""
        with st.sidebar:
            st.header("⚙️ Settings")
            
            self.model_type = st.selectbox(
                "Detection Model",
                ["frame", "panel"],
                index=0 if self.config.DEFAULT_YOLO_MODEL == "frame" else 1,
                help="Choose the YOLO model type for frame detection"
            )
            
            st.info("🇺🇸 Language: English Only")
            
            self.speech_rate = st.slider(
                "Speech Rate (WPM)",
                min_value=100,
                max_value=250,
                value=self.config.TTS_SPEECH_RATE,
                help="Words per minute for text-to-speech"
            )
            
            st.divider()
            
            # Advanced settings
            with st.expander("🔬 Advanced"):
                self.yolo_confidence = st.slider(
                    "YOLO Confidence",
                    min_value=0.1,
                    max_value=1.0,
                    value=self.config.YOLO_CONFIDENCE_THRESHOLD,
                    step=0.05
                )
                
                self.ocr_confidence = st.slider(
                    "OCR Confidence",
                    min_value=0.1,
                    max_value=1.0,
                    value=self.config.OCR_CONFIDENCE_THRESHOLD,
                    step=0.05
                )
    
    def _render_main_content(self):
        """Render the main content area"""
        col1, col2 = st.columns([1, 1])
        self.result_placeholder = col2.empty()
        
        with col1:
            st.header("📤 Upload Manga Page")
            
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=self.config.ALLOWED_IMAGE_TYPES,
                help=f"Upload a manga page image (max {self.config.MAX_UPLOAD_SIZE//1024//1024}MB)"
            )
            
            if uploaded_file is not None:
                # Check file size
                if len(uploaded_file.getvalue()) > self.config.MAX_UPLOAD_SIZE:
                    st.error(f"File too large! Maximum size is {self.config.MAX_UPLOAD_SIZE//1024//1024}MB")
                    return
                
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Manga Page", use_column_width=True)
                
                # Show image info
                st.caption(f"Size: {image.size[0]}x{image.size[1]} pixels | Format: {image.format}")
                
                # Process button
                if st.button("🎵 Generate Audio", type="primary", use_container_width=True):
                    self.process_manga_page(uploaded_file, col2)
        
        with col2:
            st.header("📊 Results")
            self.result_placeholder = st.empty()
            with self.result_placeholder.container():
                st.info("👈 Upload an image and click 'Generate Audio' to see results here.")
    
    def _render_footer(self):
        """Render the footer"""
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.caption(f"🚀 {self.config.APP_NAME} v{self.config.APP_VERSION}")
        
        with col2:
            if st.button("🧹 Clean Old Audio Files"):
                self.tts_generator.cleanup_old_files(self.config.MAX_AUDIO_FILE_AGE_HOURS)
                st.success("Audio files cleaned!")
        
        with col3:
            st.caption("Made with ❤️ and AI")
    
    def process_manga_page(self, uploaded_file, result_column):
        """Process the uploaded manga page through the entire pipeline"""
        with self.result_placeholder.container():
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Step 1: Frame Detection
                status_text.text("🔍 Detecting frames...")
                progress_bar.progress(25)
                self.frame_detector.detect_frames(temp_path, self.model_type)
                frames = self.frame_detector.extract_frames()

                # Step 2: OCR Processing
                status_text.text("📝 Extracting text...")
                progress_bar.progress(66)
                extracted_texts = self.ocr_processor.extract_text(frames, "en")  # English only
                
                # Step 3: Text Processing
                status_text.text("📝 Processing text...")
                progress_bar.progress(83)
                processed_text = self._process_extracted_text(extracted_texts)
                
                # Step 4: TTS Generation
                status_text.text("🎵 Generating audio...")
                self.tts_generator.configure_tts("en", self.speech_rate)  # English only
                progress_bar.progress(100)
                audio_path = self.tts_generator.generate_audio(processed_text, "en")
                
                # Display results
                status_text.text("✅ Processing complete!")
                
                self._display_results(frames, extracted_texts, processed_text, audio_path)
                
                # Clean up temporary files
                os.unlink(temp_path)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                status_text.text("❌ Processing failed")
                if self.config.DEBUG:
                    st.exception(e)
    
    def _process_extracted_text(self, extracted_texts: list[dict]) -> str:
        """
        Process extracted text fragments into a coherent narrative without LLM
        
        Args:
            extracted_texts: List of frames with extracted text from OCR
        
        Returns:
            Combined and processed text suitable for TTS
        """
        # Extract text fragments and sort by reading order
        text_fragments = []
        
        for idx, frame in enumerate(extracted_texts):
            cleaned_text = frame.get("cleaned_text", "").strip()
            if cleaned_text:
                text_fragments.append({
                    "text": cleaned_text,
                    "image_path": frame.get("image_path"),
                    "reading_order": idx,  # fallback since extract_text does not provide reading_order
                    "confidence": frame.get("ocr_confidence", 0.0)
                })
                
        if not text_fragments:
            return "No readable text was found in this manga page."
        
        # Sort by reading order
        text_fragments.sort(key=lambda x: x["reading_order"])
        
        # Combine texts with appropriate separators
        combined_texts = []
        for fragment in text_fragments:
            text = fragment["text"]
            # Clean up the text for better TTS
            text = self._clean_text_for_speech(text)
            if text:
                combined_texts.append(text)
        
        if not combined_texts:
            return "No readable text was found in this manga page."
        
        # Join with appropriate pauses
        final_text = ". ".join(combined_texts)
        
        # Final cleanup
        final_text = self._optimize_for_tts(final_text)
        
        return final_text
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text to make it more suitable for speech"""
        if not text:
            return ""
        
        # Remove extra whitespace
        cleaned = ' '.join(text.split())
        
        # Replace common symbols with speech-friendly versions
        cleaned = cleaned.replace("&", "and")
        cleaned = cleaned.replace("@", "at")
        cleaned = cleaned.replace("#", "number")
        cleaned = cleaned.replace("%", "percent")
        cleaned = cleaned.replace("$", "dollars")
        
        # Remove or replace characters that might cause TTS issues
        cleaned = cleaned.replace("*", "")
        cleaned = cleaned.replace("_", " ")
        cleaned = cleaned.replace("|", " ")
        
        return cleaned.strip()
    
    def _optimize_for_tts(self, text: str) -> str:
        """Optimize text for text-to-speech generation"""
        if not text:
            return ""
        
        import re
        
        # Clean up spacing
        optimized = re.sub(r'\s+', ' ', text)
        
        # Ensure proper sentence ending
        if optimized and not optimized.endswith(('.', '!', '?')):
            optimized += '.'
        
        # Add slight pauses for better speech flow
        optimized = optimized.replace('. ', '. ')  # Ensure space after periods
        optimized = optimized.replace('! ', '! ')  # Ensure space after exclamations
        optimized = optimized.replace('? ', '? ')  # Ensure space after questions
        
        return optimized.strip()
    
    def _display_results(self, frames, extracted_texts, processed_text, audio_path):
        """Display processing results"""
        st.success("🎉 Audio generated successfully!")
        
        # Statistics
        with st.expander("📈 Processing Statistics"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Frames Detected", len(frames))
                st.metric("Text Fragments", len([f for f in extracted_texts if f.get("cleaned_text")]))
            
            with col2:
                avg_confidence = sum(f.get("ocr_confidence", 0) for f in extracted_texts) / max(1, len(extracted_texts))
                st.metric("Avg OCR Confidence", f"{avg_confidence:.2f}")
                st.metric("Output Characters", len(processed_text))
            
            with col3:
                tts_stats = self.tts_generator.get_tts_statistics(processed_text)
                st.metric("Estimated Duration", f"{tts_stats['estimated_duration_seconds']}s")
                st.metric("Word Count", tts_stats['text_length_words'])
        
        # Show processed text
        with st.expander("📄 Generated Narrative"):
            st.text_area("Processed text:", processed_text, height=150)
        
        # Audio player
        st.subheader("🎵 Generated Audio")
        
        # Get audio info
        audio_info = self.tts_generator.get_audio_info(audio_path)
        
        if audio_info.get("is_mock", False):
            st.warning("⚠️ This is a mock audio file for demonstration. Configure TTS services for actual audio generation.")
            st.info("📝 Mock audio content preview:")
            with open(audio_path, 'r') as f:
                st.code(f.read()[:500] + "...")
        else:
            # Real audio file
            with open(audio_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav')
            
            # Download button
            st.download_button(
                label="📥 Download Audio",
                data=audio_bytes,
                file_name="manga_audio.wav",
                mime="audio/wav",
                use_container_width=True
            )


def main():
    """Main application entry point"""
    app = MangaAIApp()
    app.run()


if __name__ == "__main__":
    main()
