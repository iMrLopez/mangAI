# MangAI 📚🎵

A manga-to-audio application that converts English manga pages into audio narratives using AI. Upload a manga### Configuration

Create a `.env` file from `.env.example`:

```bash
# Application settings (English only)
DEFAULT_YOLO_MODEL=frame
TTS_SPEECH_RATE=150

# Model and processing settings
YOLO_CONFIDENCE_THRESHOLD=0.5
OCR_CONFIDENCE_THRESHOLD=0.3
```an audio story!

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

### Vanilla 

1. **Create python local env:**
```bash
 git clone <repository-url>
 python3 -m venv venv
 source venv/bin/activate
 pip install --upgrade pip
 pip install -r requirements.txt
```
2. **Run the application:**
   ```bash
   ./start.sh
   # or
   streamlit run app.py
   ```

### Using Docker (Recommended)

1. **Clone and navigate to the project:**
   ```bash
   git clone <repository-url>
   cd mangAI
   ```

2. **Configure environment (optional):**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   Open your browser and go to `http://localhost:8501`

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install system dependencies (for OCR):**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
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
├── app.py                 # Main Streamlit application
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── start.sh              # Startup script
├── modules/              # Core processing modules
│   ├── frame_detector.py # YOLO-based frame detection
│   ├── ocr_processor.py  # OCR text extraction (English)
│   └── tts_generator.py  # Text-to-speech generation
├── models/               # YOLO model files
│   ├── yolo8l_50epochs_frame/
│   └── yolo8l_50epochs/
├── processed_>date</ # outputs
│   ├── audio/
│   └── ocr/
│   └── frames/
```

## Configuration

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
# OpenAI API Key (for LLM processing)
OPENAI_API_KEY=your_openai_api_key_here

# Azure Speech Services (alternative TTS)
AZURE_SPEECH_KEY=your_azure_speech_key_here
AZURE_SPEECH_REGION=your_azure_region_here

# Application settings
DEFAULT_LANGUAGE=en
TTS_SPEECH_RATE=150
DEFAULT_YOLO_MODEL=frame
```

### Supported Languages

- **English** (`en`) - Full support

## Usage

1. **Upload Image**: Select an English manga page image (PNG, JPG, JPEG)
2. **Configure Settings**: Choose detection model and speech rate
3. **Generate Audio**: Click "Generate Audio" to process
4. **Listen & Download**: Play the generated audio or download it

## Development

### Adding New Modules

Each processing module follows a consistent interface:

```python
class NewProcessor:
    def __init__(self):
        """Initialize the processor"""
        pass
    
    def process(self, input_data):
        """Main processing method"""
        pass
```

### Model Integration

To integrate new YOLO models:

1. Place model files in `./models/model_name/`
2. Update `config.py` MODEL_PATHS
3. The frame detector will automatically load them

To add new LLM or TTS providers:

1. Extend the respective processor classes
2. Add configuration in `config.py`
3. Update the frontend options

**Note**: LLM processing has been removed for simplicity. Text is now processed using simple combination and cleaning.

## Performance

- **Frame Detection**: ~1-3 seconds per image
- **OCR Processing**: ~2-5 seconds per frame
- **Text Processing**: ~1-2 seconds (simple combination)
- **TTS Generation**: ~2-8 seconds (depending on text length)

## Troubleshooting

### Common Issues

1. **Model files not found**: Ensure YOLO models are in the correct paths
2. **OCR not working**: Install Tesseract (English language pack included by default)
3. **TTS quality**: Configure voice settings in the TTS module
4. **Memory issues**: Reduce image size or use smaller YOLO models

### Logs

Check application logs:
```bash
# Docker
docker-compose logs mangai-app

# Local
tail -f logs/app.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLO models for frame detection
- Tesseract OCR for text extraction
- Streamlit for the web interface

---

**Note**: This implementation focuses on English manga processing for simplicity. LLM processing has been removed in favor of direct text combination and cleaning.
