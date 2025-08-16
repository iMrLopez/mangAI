"""
TTS Generator Module
Converts processed text into audio using text-to-speech technology
"""

import os
import uuid
import time
import datetime
from typing import Dict, Any

from dotenv import load_dotenv
try:
    from elevenlabs import ElevenLabs, VoiceSettings
    ELEVENLABS_AVAILABLE = True
except ImportError:
    print("Warning: ElevenLabs package not properly installed")
    ElevenLabs = None
    VoiceSettings = None
    ELEVENLABS_AVAILABLE = False
    ELEVENLABS_AVAILABLE = False


class TTSGenerator:
    def __init__(self, output_dir: str = "./audio_output"):
        """Initialize the TTS generator for English only"""
        self.supported_languages = {
            "en": {"voice": "en-US", "rate": 150}  # English only
        }
        self.current_language = "en"
        self.current_rate = 150
        self.output_dir = output_dir
        
        # Load environment variables
        load_dotenv()
        
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        
        # Voice configuration
        self.voice_ids = {
            "narrator": os.getenv("ELEVEN_NARRATOR_VOICE_ID", "pNInz6obpgDQGcFmaJgB"),  # Default Adam voice
            "character": os.getenv("ELEVEN_ACTOR_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),  # Default Bella voice
            "default": "pNInz6obpgDQGcFmaJgB"  # Adam voice as fallback
        }
        
        self.client = None
        
        # Initialize ElevenLabs client if API key is available
        if self.elevenlabs_api_key and ELEVENLABS_AVAILABLE:
            try:
                self.client = ElevenLabs(api_key=self.elevenlabs_api_key)
                print("ElevenLabs client initialized successfully")
            except Exception as e:
                print(f"Error: Could not initialize ElevenLabs client: {e}")
                raise Exception(f"Failed to initialize ElevenLabs: {e}")
        else:
            if not ELEVENLABS_AVAILABLE:
                raise Exception("ElevenLabs package not available. Please install: pip install elevenlabs")
            elif not self.elevenlabs_api_key:
                raise Exception("ELEVENLABS_API_KEY not found. Please set the environment variable.")
        
        # Ensure audio output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def set_output_directory(self, output_dir: str):
        """Set a new output directory for audio files"""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Audio output directory set to: {output_dir}")
    
    def configure_tts(self, language: str, rate: int = 150):
        """Configure TTS settings"""
        if language in self.supported_languages:
            self.current_language = language
            self.current_rate = rate
        else:
            print(f"Warning: Language {language} not supported. Using English.")
            self.current_language = "en"
            self.current_rate = rate
    
    def generate_audio_from_script(self, manga_script: "list[dict]", language: str = "en") -> str:
        """
        Generate audio from structured manga script data
        
        Args:
            manga_script: List of dictionaries with 'role' and 'description' keys
            language: Language code (currently only 'en' supported)
            
        Returns:
            Path to generated audio file
        """
        if not manga_script or len(manga_script) == 0:
            raise ValueError("No script data provided for audio generation")
        
        if not self.client or not ELEVENLABS_AVAILABLE:
            raise Exception("ElevenLabs API is not configured. Please set ELEVENLABS_API_KEY environment variable.")
        
        return self._generate_multi_voice_from_structured_data(manga_script)
    
    def _has_voice_cues(self, text: str) -> bool:
        """Check if text contains narrator/character voice cues"""
        import re
        # Look for patterns like [narrator text], "Character: dialogue", etc.
        narrator_pattern = r'\[.*?\]'
        character_pattern = r'^[A-Z][a-zA-Z\s]*:\s+'
        
        has_narrator = bool(re.search(narrator_pattern, text))
        has_character = bool(re.search(character_pattern, text, re.MULTILINE))
        
        return has_narrator or has_character
    
    def _parse_script(self, text: str) -> "list[dict]":
        """
        Parse manga script into segments with role assignments.
        
        Args:
            text (str): The manga script text
            
        Returns:
            List of dictionaries with 'role', 'text', and 'voice_id' keys
        """
        import re
        
        segments = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for narrator text (enclosed in brackets)
            narrator_match = re.match(r'\[(.*?)\]', line)
            if narrator_match:
                segments.append({
                    'role': 'narrator',
                    'text': narrator_match.group(1).strip(),
                    'voice_id': self.voice_ids['narrator']
                })
                continue
            
            # Check for character dialogue (Character: text)
            character_match = re.match(r'^([A-Z][a-zA-Z\s]*?):\s*(.+)$', line)
            if character_match:
                character_name = character_match.group(1).strip()
                dialogue = character_match.group(2).strip()
                segments.append({
                    'role': 'character',
                    'character_name': character_name,
                    'text': dialogue,
                    'voice_id': self.voice_ids['character']
                })
                continue
            
            # Default: treat as narrator if no specific format
            segments.append({
                'role': 'narrator',
                'text': line,
                'voice_id': self.voice_ids['narrator']
            })
        
        return segments
    
    def _generate_multi_voice_from_structured_data(self, manga_script: "list[dict]") -> str:
        """
        Generate multi-voice audio from structured manga script data
        
        Args:
            manga_script: List of dictionaries with 'role' and 'description' keys
            
        Returns:
            Path to generated audio file
        """
        if not self.client or not ELEVENLABS_AVAILABLE or not VoiceSettings:
            raise Exception("ElevenLabs client not properly initialized for multi-voice")
            
        if not manga_script:
            raise ValueError("Empty manga script provided")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"manga_multivoice_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        print(f"Processing {len(manga_script)} structured script segments for multi-voice generation...")
        
        # For now, we'll create separate audio files and combine them as text
        # In a production environment, you'd want to use audio processing libraries
        # like pydub to properly concatenate MP3 files
        
        combined_text_parts = []
        
        for i, entry in enumerate(manga_script):
            role = entry.get('role', '').lower()
            description = entry.get('description', '').strip()
            
            print(f"Processing entry {i+1}/{len(manga_script)}: {role} - {description[:30]}...")
            
            if not description:
                print(f"  -> Skipping empty description")
                continue
                
            # Format text based on role
            if role == 'narrator':
                # Narrator provides scene description
                formatted_text = f"{description}"
                print(f"  -> Added narrator segment: {formatted_text[:50]}...")
            elif role == 'character':
                # Character dialogue - more expressive
                formatted_text = f"{description}"
                print(f"  -> Added character segment: {formatted_text[:50]}...")
            else:
                # Unknown role - treat as narrator
                formatted_text = f"{description}"
                print(f"  -> Added unknown role as narrator: {formatted_text[:50]}...")
            
            combined_text_parts.append(formatted_text)
        
        if not combined_text_parts:
            raise ValueError("No valid segments found in manga script")
        
        print(f"Combined {len(combined_text_parts)} text parts into single audio generation...")
        
        # Join with pauses for better flow
        combined_text = " ... ".join(combined_text_parts)
        
        # For multi-voice content, we'll use the narrator voice as primary
        # but with enhanced expression settings to differentiate roles
        print(f"Generating structured multi-voice audio with narrator voice...")
        print(f"Text length: {len(combined_text)} characters")
        
        response = self.client.text_to_speech.convert(
            voice_id=self.voice_ids['narrator'],
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=combined_text,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(
                stability=0.2,  # Medium stability for varied expression
                similarity_boost=0.8,
                style=0.4,  # Higher style for better character differentiation
                use_speaker_boost=True,
            ),
        )
        
        # Save the audio
        save_file_path = f"{self.output_dir}/{base_filename}.mp3"
        
        with open(save_file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)
        
        print(f"Structured multi-voice audio saved at {save_file_path}")
        
        # Create a detailed transcript file
        transcript_path = f"{self.output_dir}/{base_filename}_transcript.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("STRUCTURED MULTI-VOICE MANGA AUDIO TRANSCRIPT\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"Generated: {datetime.datetime.now()}\n")
            f.write(f"Script Segments: {len(manga_script)}\n")
            f.write(f"Text Parts Combined: {len(combined_text_parts)}\n")
            f.write(f"Narrator Voice: {self.voice_ids['narrator']}\n")
            f.write(f"Character Voice: {self.voice_ids['character']}\n\n")
            f.write(f"Combined Text ({len(combined_text)} chars):\n")
            f.write(f"{combined_text}\n\n")
            f.write("Original Structured Script:\n")
            
            for i, entry in enumerate(manga_script, 1):
                role = entry.get('role', 'unknown').upper()
                description = entry.get('description', '')
                f.write(f"{i}. [{role}]: {description}\n")
        
        return save_file_path
    
    def get_tts_statistics(self, text: str) -> Dict[str, Any]:
        """Get statistics about the text for TTS generation"""
        if not text:
            return {
                "text_length_characters": 0,
                "text_length_words": 0,
                "estimated_duration_seconds": 0
            }
        
        words = len(text.split())
        chars = len(text)
        estimated_duration = words / (self.current_rate / 60)  # Convert WPM to words per second
        
        return {
            "text_length_characters": chars,
            "text_length_words": words,
            "estimated_duration_seconds": round(estimated_duration, 1)
        }
    
    def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """Get information about generated audio file"""
        if not os.path.exists(audio_path):
            return {"exists": False}
        
        file_size = os.path.getsize(audio_path)
        
        return {
            "exists": True,
            "file_path": audio_path,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "type": "Real audio file (MP3)"
        }
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old audio files from the output directory"""
        audio_dir = self.output_dir
        if not os.path.exists(audio_dir):
            return
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        removed_files = 0
        
        for filename in os.listdir(audio_dir):
            file_path = os.path.join(audio_dir, filename)
            if os.path.isfile(file_path):
                if os.path.getmtime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        removed_files += 1
                    except Exception as e:
                        print(f"Could not remove {file_path}: {e}")
        
        print(f"Cleaned up {removed_files} old audio files")