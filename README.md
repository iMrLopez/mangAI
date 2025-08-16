# MangAI 📚🎵

<div style="display: flex; align-items: center; gap: 20px;">
   <img src="images/mangAI.png" alt="MangAI Logo" width="300">
   <div>
      A manga-to-audio application that converts English manga pages into immersive audio narratives using AI. Upload a manga image and generate a complete audio story with separate narrator and character voices using OpenAI GPT-4 Vision, GPT-4 Text, and ElevenLabs multi-voice TTS!
   </div>
</div>

## Features

- 🔍 **Smart Frame Detection**: Automatically detects manga panels using YOLO models with reading order optimization
- 📝 **Advanced OCR Processing**: Extracts English text from manga panels using PaddleOCR with confidence filtering
- 🧠 **AI Scene Analysis**: GPT-4 Vision analyzes visual context and scene descriptions for each frame
- ✍️ **Intelligent Script Generation**: GPT-4 Text creates cohesive narratives combining visual analysis with extracted text
- 🎭 **Multi-Voice TTS**: ElevenLabs generates separate narrator and character voices for immersive storytelling
- 📁 **Organized Output**: Structured processing directories with timestamped folders containing frames, OCR results, and audio files
- 🌐 **Interactive Web Interface**: User-friendly Streamlit frontend with real-time progress tracking and multi-voice audio controls
- 🇺🇸 **English Focused**: Optimized for English manga processing with advanced language models

## Architecture

📊 **For detailed interactive diagrams, see [`architecture_diagram.md`](./architecture_diagram.md)** - Contains comprehensive Mermaid diagrams showing system architecture, data flow, and component interactions.

### High-Level Flow
```
┌─────────────────────┐
│   Streamlit App     │  ← Web Frontend with Multi-Voice Controls
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Frame Detector    │  ← YOLO Models with Reading Order Detection
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   OCR Processor     │  ← PaddleOCR (English) with Confidence Filtering
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   LLM Vision        │  ← GPT-4 Vision for Scene Analysis
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   LLM Narrator      │  ← GPT-4 Text for Script Generation
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Multi-Voice TTS    │  ← ElevenLabs with Narrator & Character Voices
└─────────────────────┘
```

## Features

- 🔍 **Frame Detection**: Automatically detects manga panels using YOLO models
- 📝 **OCR Text Extraction**: Extracts English text from manga panels
- 📝 **Text Processing**: Simple text combination and cleaning for audio generation
- 🎵 **Text-to-Speech**: Generates audio from processed text
- 🌐 **Web Interface**: User-friendly Streamlit frontend for easy interaction
- 🐳 **Containerized**: Full Docker support for easy deployment
- 🇺🇸 **English Only**: Focused on English manga processing for simplicity

## Architecture

📊 **For detailed interactive diagrams, see [`architecture_diagram.md`](./architecture_diagram.md)** - Contains comprehensive Mermaid diagrams showing system architecture, data flow, and component interactions.

### High-Level Flow
```
┌─────────────────────┐
│   Streamlit App     │  ← Web Frontend
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Frame Detector    │  ← YOLO Models (integrated from yolov8Model.py)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   OCR Processor     │  ← Tesseract/OCR (English only)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Text Processor    │  ← Simple text combination
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   TTS Generator     │  ← Text-to-Speech (English)
└─────────────────────┘
```

### 🔄 Migration from Original Code

The original `yolov8Model.py` functionality has been integrated into `modules/frame_detector.py` with the following improvements:

- **Enhanced Architecture**: Better separation of concerns and modular design
- **Configuration Management**: Centralized settings and environment variables
- **Improved Error Handling**: Better exception handling and validation
- **Extended Functionality**: Additional methods for statistics and visualization
- **Backward Compatibility**: Original interface methods are preserved
- **Type Safety**: Added type hints and better documentation
- **Simplified Focus**: English-only processing for cleaner implementation

## Quick Start

### Prerequisites

Before running MangAI, you need to obtain API keys for:
- **OpenAI API**: For GPT-4 Vision and Text processing
- **ElevenLabs API**: For multi-voice TTS generation

### Local Development Setup

1. **Clone and set up the project:**
   ```bash
   git clone <repository-url>
   cd mangAI
   python3 -m venv virtualenv
   source virtualenv/bin/activate  # On macOS/Linux
   # or: virtualenv\Scripts\activate  # On Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Configure API credentials:**
   Create a `.env` file in the project root:
   ```bash
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   
   # ElevenLabs Configuration
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   ELEVENLABS_NARRATOR_VOICE_ID=voice_id_for_narrator
   ELEVENLABS_CHARACTER_VOICE_ID=voice_id_for_character
   
   # Application Settings
   DEFAULT_YOLO_MODEL=frame
   YOLO_CONFIDENCE_THRESHOLD=0.5
   OCR_CONFIDENCE_THRESHOLD=0.3
   ```

3. **Run the application:**
   ```bash
   ./start.sh
   # or directly:
   streamlit run app.py
   ```

4. **Access the application:**
   Open your browser and go to `http://localhost:8501`

### Using Virtual Environment (Recommended)

The project includes a pre-configured virtual environment in the `virtualenv/` directory:

```bash
# Activate the existing virtual environment
source virtualenv/bin/activate  # On macOS/Linux
# or: virtualenv\Scripts\activate  # On Windows

# Install any missing dependencies
pip install -r requirements.txt

# Run the application
./start.sh
```
   
   # macOS
   brew install tesseract
   ```

3. **Test the integration:**
   ```bash
   python test_integration.py
   ```

4. **Run the application:**
   ```bash
   ./start.sh
   # or
   streamlit run app.py
   ```

## Project Structure

```
mangAI/
├── app.py                    # Main Streamlit application with multi-voice interface
├── config.py                 # Configuration management with directory creation
├── requirements.txt          # Python dependencies (OpenAI, ElevenLabs, PaddleOCR)
├── start.sh                  # Startup script for virtual environment
├── architecture_diagram.md   # Comprehensive system architecture documentation
├── README.md                 # Project documentation
├── modules/                  # Core processing modules
│   ├── __init__.py
│   ├── frame_detector.py     # YOLO-based frame detection with reading order
│   ├── ocr_processor.py      # PaddleOCR text extraction with confidence filtering
│   ├── llm_processor.py      # OpenAI GPT-4 Vision and Text processing
│   └── tts_generator.py      # ElevenLabs multi-voice TTS generation
├── models/                   # YOLO model files
│   ├── yolo8l_50epochs/
│   ├── yolo8l_50epochs_frame/
│   └── yolo8s_50epochs/
├── images/                   # Test manga images
│   ├── test1.jpg
│   ├── test2.jpg
│   └── ...
├── audio_output/             # Structured processing outputs
│   ├── processed_20240101_120000/
│   │   ├── frames/           # Extracted manga frames
│   │   ├── ocr/             # OCR results and combined text
│   │   └── audio/           # Multi-voice audio files and transcript
│   └── processed_YYYYMMDD_HHMMSS/
├── logs/                     # Application logs
└── virtualenv/               # Pre-configured Python environment
    ├── bin/
    ├── lib/
    └── ...
```

## Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```bash
# OpenAI Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here

# ElevenLabs Configuration (Required)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_NARRATOR_VOICE_ID=voice_id_for_narrator
ELEVENLABS_CHARACTER_VOICE_ID=voice_id_for_character

# Application Settings
DEFAULT_YOLO_MODEL=frame
YOLO_CONFIDENCE_THRESHOLD=0.5
OCR_CONFIDENCE_THRESHOLD=0.3
```

### API Configuration

1. **OpenAI API Setup**:
   - Get your API key from [OpenAI Platform](https://platform.openai.com/)
   - The app uses GPT-4 Vision for scene analysis and GPT-4 Text for narrative generation

2. **ElevenLabs API Setup**:
   - Get your API key from [ElevenLabs](https://elevenlabs.io/)
   - Create or select voice IDs for narrator and character roles
   - The app generates separate audio tracks for different voices

3. **Model Configuration**:
   - `frame`: Best for manga frame detection
   - `yolo8l_50epochs`: Alternative YOLO model
   - `yolo8s_50epochs`: Smaller, faster model option

## Usage

1. **Upload Image**: Select an English manga page image (PNG, JPG, JPEG)
2. **Configure Settings**: 
   - Choose YOLO detection model
   - Adjust confidence thresholds if needed
3. **Generate Audio**: Click "Generate Audio" to start processing
   - Frame detection and extraction
   - OCR text extraction with confidence filtering
   - AI scene analysis using GPT-4 Vision
   - Narrative script generation using GPT-4 Text
   - Multi-voice audio generation using ElevenLabs
4. **Review Results**: 
   - View processing statistics
   - Play separate narrator/character audio or combined version
   - Download individual audio files or complete transcript
5. **Explore Output**: Browse the timestamped processing folder with organized frames, OCR results, and audio files

## Development

### Adding New Modules

Each processing module follows a consistent interface pattern:

```python
class NewProcessor:
    def __init__(self, config=None):
        """Initialize the processor with configuration"""
        self.config = config or Config()
    
    def process(self, input_data, output_dir=None):
        """Main processing method with structured output"""
        pass
    
    def get_statistics(self):
        """Return processing statistics"""
        pass
```

### Model Integration

**Adding New YOLO Models**:
1. Place model files in `./models/model_name/best.pt`
2. Update `config.py` MODEL_PATHS dictionary
3. The frame detector will automatically load and use them

**Integrating Alternative LLM Providers**:
1. Extend `LLMProcessor` class in `modules/llm_processor.py`
2. Implement vision and text processing methods
3. Add provider configuration in `config.py`
4. Update frontend provider selection

**Adding New TTS Providers**:
1. Extend `TTSGenerator` class in `modules/tts_generator.py`
2. Implement multi-voice generation methods
3. Add API configuration and voice settings
4. Update frontend voice configuration options

### Directory Structure Standards

All processing modules should use the structured directory pattern:
```
processed_YYYYMMDD_HHMMSS/
├── frames/          # Input frames and extraction results
├── ocr/            # OCR results and text processing
└── audio/          # Audio files and transcripts
```

## Performance

- **Frame Detection**: ~1-3 seconds per image (YOLO processing)
- **OCR Processing**: ~2-5 seconds per frame (PaddleOCR with confidence filtering)
- **LLM Vision Analysis**: ~3-8 seconds per frame (GPT-4 Vision processing)
- **LLM Script Generation**: ~5-15 seconds (GPT-4 Text narrative creation)
- **Multi-Voice TTS Generation**: ~10-30 seconds (ElevenLabs narrator + character voices)

**Total Processing Time**: ~2-5 minutes per manga page (depending on frame count and text complexity)

## Troubleshooting

### Common Issues

1. **API Configuration**:
   - Ensure OpenAI API key is valid and has GPT-4 access
   - Verify ElevenLabs API key and voice IDs are correct
   - Check API rate limits and quotas

2. **Model Files**:
   - Ensure YOLO models are in the correct paths (`./models/*/best.pt`)
   - Check model file permissions and accessibility

3. **Processing Errors**:
   - OCR confidence too low: Adjust `OCR_CONFIDENCE_THRESHOLD`
   - Frame detection issues: Try different YOLO models or adjust confidence
   - LLM processing failures: Check API keys and rate limits

4. **Audio Generation**:
   - Voice ID not found: Verify ElevenLabs voice IDs in configuration
   - Audio quality issues: Check voice settings and text preprocessing
   - File output errors: Ensure write permissions to `audio_output/` directory

### Logs and Debugging

Check processing logs:
```bash
# View application logs
tail -f logs/app.log

# Check specific processing folder
ls -la audio_output/processed_*/

# Verify API connectivity
python -c "import openai; print('OpenAI OK')"
python -c "import elevenlabs; print('ElevenLabs OK')"
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes following the module interface patterns
4. Test with sample manga images
5. Update documentation if needed
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **OpenAI** for GPT-4 Vision and Text models
- **ElevenLabs** for high-quality multi-voice TTS
- **Ultralytics** for YOLO object detection models
- **PaddleOCR** for robust OCR text extraction
- **Streamlit** for the interactive web interface

---

**MangAI** transforms static manga pages into immersive audio experiences using cutting-edge AI technologies. From visual analysis to multi-voice narration, experience your favorite manga like never before! 🎭📚🎵
