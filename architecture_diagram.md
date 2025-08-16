# MangAI Architecture Diagrams

This document contains comprehensive Mermaid diagrams showing the architecture and data flow of the MangAI application.

## 1. High-Level System Architecture

```mermaid
graph TD
    A[User Interface - Streamlit] --> B[Frame Detector Module]
    B --> C[OCR Processor Module]
    C --> D[LLM Vision Processor]
    D --> E[LLM Narrator Module]
    E --> F[Multi-Voice TTS Generator]
    F --> G[Structured Audio Output]
    
    H[Config Management] --> A
    H --> B
    H --> C
    H --> D
    H --> E
    H --> F
    
    I[YOLO Models] --> B
    J[PaddleOCR Engine] --> C
    K[OpenAI GPT Models] --> D
    K --> E
    L[ElevenLabs API] --> F
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#f1f8e9
    style G fill:#fff8e1
    style H fill:#e3f2fd
```

## 2. Detailed Component Interaction

```mermaid
graph TB
    subgraph "MangAI Application"
        subgraph "Frontend Layer"
            UI[app.py - Streamlit UI]
            CONFIG[config.py - Configuration]
            ENV[.env - Environment Variables]
        end
        
        subgraph "Processing Pipeline"
            FD[Frame Detector<br/>modules/frame_detector.py]
            OCR[OCR Processor<br/>modules/ocr_processor.py]
            LLM_VIS[LLM Vision Processor<br/>modules/llm_processor.py]
            LLM_NAR[LLM Narrator Module<br/>modules/llm_processor.py]
            TTS[Multi-Voice TTS Generator<br/>modules/tts_generator.py]
        end
        
        subgraph "External Dependencies"
            YOLO[YOLO Models<br/>ultralytics]
            PADDLE[PaddleOCR<br/>paddlepaddle]
            OPENAI[OpenAI API<br/>GPT-4 Vision & Text]
            ELEVEN[ElevenLabs API<br/>Multi-Voice TTS]
        end
        
        subgraph "File System"
            MODELS[models/<br/>*.pt YOLO files]
            IMAGES[images/<br/>test images]
            PROCESSED[processed_<timestamp>/<br/>├── frames/<br/>├── ocr/<br/>└── audio/]
        end
    end
    
    UI --> CONFIG
    CONFIG --> ENV
    UI --> FD
    FD --> OCR
    OCR --> LLM_VIS
    LLM_VIS --> LLM_NAR
    LLM_NAR --> TTS
    
    FD --> YOLO
    FD --> MODELS
    OCR --> PADDLE
    LLM_VIS --> OPENAI
    LLM_NAR --> OPENAI
    TTS --> ELEVEN
    TTS --> PROCESSED
    
    UI -.-> IMAGES
    
    style UI fill:#e3f2fd
    style CONFIG fill:#f3e5f5
    style ENV fill:#e8f5e8
    style FD fill:#fff3e0
    style OCR fill:#fce4ec
    style LLM_VIS fill:#f1f8e9
    style LLM_NAR fill:#fff8e1
    style TTS fill:#e1f5fe
```

## 3. Data Flow Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Streamlit UI
    participant FD as Frame Detector
    participant OCR as OCR Processor
    participant LLM_VIS as LLM Vision
    participant LLM_NAR as LLM Narrator
    participant TTS as Multi-Voice TTS
    participant FS as File System
    
    U->>UI: Upload manga image
    UI->>UI: Validate file & display preview
    U->>UI: Configure settings (model, speech rate)
    UI->>UI: Update configuration
    U->>UI: Click "Generate Audio"
    
    UI->>FS: Create timestamped processing directory
    FS-->>UI: Return directory structure (frames/, ocr/, audio/)
    
    UI->>FD: Process image for frame detection
    FD->>FD: Load YOLO model
    FD->>FD: Detect manga frames
    FD->>FD: Order frames by reading sequence
    FD->>FS: Save extracted frames to frames/
    FD-->>UI: Return ordered frame images & paths
    
    UI->>OCR: Extract text from each frame
    OCR->>OCR: Preprocess frame images
    OCR->>OCR: Run PaddleOCR (English)
    OCR->>OCR: Filter by confidence threshold
    OCR->>FS: Save OCR results to ocr/
    OCR-->>UI: Return text fragments with metadata
    
    UI->>LLM_VIS: Analyze frames for scene description
    LLM_VIS->>LLM_VIS: Process each frame with GPT-4 Vision
    LLM_VIS-->>UI: Return scene descriptions
    
    UI->>LLM_NAR: Generate manga script from scenes & text
    LLM_NAR->>LLM_NAR: Combine descriptions and text into narrative
    LLM_NAR->>LLM_NAR: Structure into narrator/character roles
    LLM_NAR-->>UI: Return structured script data
    
    UI->>TTS: Generate multi-voice audio from script
    TTS->>TTS: Parse narrator vs character segments
    TTS->>TTS: Generate narrator audio with narrator voice
    TTS->>TTS: Generate character audio with character voice
    TTS->>FS: Save separate audio files to audio/
    TTS->>FS: Create transcript file
    TTS-->>UI: Return primary audio file path
    
    UI->>U: Display results & multi-voice audio player
    U->>UI: Play audio, download files, or view transcript
```

## 4. Module Dependencies

```mermaid
graph LR
    subgraph "Core Application"
        APP[app.py] --> CONFIG[config.py]
        APP --> ST[streamlit]
        APP --> PIL[PIL.Image]
        APP --> NP[numpy]
    end
    
    subgraph "Processing Modules"
        APP --> FD[modules.frame_detector]
        APP --> OCR[modules.ocr_processor]
        APP --> LLM[modules.llm_processor]
        APP --> TTS[modules.tts_generator]
    end
    
    subgraph "Frame Detection Stack"
        FD --> YOLO[ultralytics.YOLO]
        FD --> CV2[cv2/OpenCV]
        FD --> PIL2[PIL.Image]
        FD --> NP2[numpy]
    end
    
    subgraph "OCR Processing Stack"
        OCR --> PADDLE[paddleocr.PaddleOCR]
        OCR --> CV2B[cv2/OpenCV]
        OCR --> NP3[numpy]
    end
    
    subgraph "LLM Processing Stack"
        LLM --> OPENAI[openai.OpenAI]
        LLM --> BASE64[base64]
    end
    
    subgraph "TTS Generation Stack"
        TTS --> ELEVEN[elevenlabs.client]
        TTS --> PYDUB[pydub]
    end
    
    subgraph "System Utilities"
        CONFIG --> OS[os.path]
        CONFIG --> DATETIME[datetime]
        CONFIG --> PATHLIB[pathlib]
    end
    
    style APP fill:#e1f5fe
    style FD fill:#f3e5f5
    style OCR fill:#e8f5e8
    style LLM fill:#fff3e0
    style TTS fill:#fce4ec
    style YOLO fill:#ffebee
    style PADDLE fill:#ffebee
    style OPENAI fill:#ffebee
    style ELEVEN fill:#ffebee
```

## 5. User Interaction Flow

```mermaid
stateDiagram-v2
    [*] --> ImageUpload
    ImageUpload --> FileValidation
    FileValidation --> PreviewDisplay
    PreviewDisplay --> ConfigurationSettings
    ConfigurationSettings --> ReadyToProcess
    ReadyToProcess --> ProcessingPipeline
    
    state ProcessingPipeline {
        [*] --> DirectoryCreation
        DirectoryCreation --> FrameDetection
        FrameDetection --> OCRExtraction
        OCRExtraction --> LLMVisionAnalysis
        LLMVisionAnalysis --> LLMNarrativeGeneration
        LLMNarrativeGeneration --> MultiVoiceTTSGeneration
        MultiVoiceTTSGeneration --> [*]
    }
    
    ProcessingPipeline --> ResultsDisplay
    ResultsDisplay --> MultiVoiceAudioPlayback
    MultiVoiceAudioPlayback --> FileDownload
    FileDownload --> [*]
    
    ImageUpload : User uploads manga image
    FileValidation : System validates file type/size
    PreviewDisplay : Display image preview
    ConfigurationSettings : User configures YOLO model & speech settings
    ReadyToProcess : Ready state with Generate button
    DirectoryCreation : Create timestamped processing directory
    FrameDetection : YOLO detects & extracts manga frames
    OCRExtraction : PaddleOCR extracts English text
    LLMVisionAnalysis : GPT-4 Vision analyzes frame scenes
    LLMNarrativeGeneration : GPT-4 Text creates structured script
    MultiVoiceTTSGeneration : ElevenLabs generates narrator & character audio
    ResultsDisplay : Show statistics & multi-voice breakdown
    MultiVoiceAudioPlayback : Play narrator/character audio separately or combined
    FileDownload : Download audio files & transcript
```

## 6. File System Organization

```mermaid
graph TD
    ROOT[mangAI/] --> APP[app.py]
    ROOT --> CONFIG[config.py]
    ROOT --> REQ[requirements.txt]
    ROOT --> START[start.sh]
    ROOT --> README[README.md]
    ROOT --> ARCH[architecture_diagram.md]
    
    ROOT --> MODULES[modules/]
    MODULES --> INIT[__init__.py]
    MODULES --> FD_PY[frame_detector.py]
    MODULES --> OCR_PY[ocr_processor.py]
    MODULES --> LLM_PY[llm_processor.py]
    MODULES --> TTS_PY[tts_generator.py]
    
    ROOT --> MODELS[models/]
    MODELS --> YOLO8L[yolo8l_50epochs/]
    MODELS --> YOLO8L_FRAME[yolo8l_50epochs_frame/]
    MODELS --> YOLO8S[yolo8s_50epochs/]
    YOLO8L --> BEST1[best.pt]
    YOLO8L_FRAME --> BEST2[best.pt]
    YOLO8S --> BEST3[best.pt]
    
    ROOT --> IMAGES[images/]
    IMAGES --> TEST1[test1.jpg]
    IMAGES --> TEST2[test2.jpg]
    IMAGES --> TEST3[test3.jpg]
    IMAGES --> TEST4[test4.jpg]
    
    ROOT --> AUDIO[audio_output/]
    AUDIO --> PROC1[processed_20240101_120000/]
    AUDIO --> PROC2[processed_20240101_135000/]
    AUDIO --> PROCN[processed_YYYYMMDD_HHMMSS/]
    
    PROC1 --> FRAMES1[frames/]
    PROC1 --> OCR1[ocr/]
    PROC1 --> AUDIO1[audio/]
    
    FRAMES1 --> FRAME1[frame_0.jpg]
    FRAMES1 --> FRAME2[frame_1.jpg]
    FRAMES1 --> FRAMEX[frame_n.jpg]
    
    OCR1 --> OCR_JSON[ocr_results.json]
    OCR1 --> OCR_TXT[combined_text.txt]
    
    AUDIO1 --> NARRATOR[narrator_audio.wav]
    AUDIO1 --> CHARACTER[character_audio.wav]
    AUDIO1 --> COMBINED[combined_audio.wav]
    AUDIO1 --> TRANSCRIPT[transcript.txt]
    
    ROOT --> LOGS[logs/]
    ROOT --> VENV[virtualenv/]
    
    style ROOT fill:#e1f5fe
    style MODULES fill:#f3e5f5
    style MODELS fill:#e8f5e8
    style AUDIO fill:#fff3e0
    style PROC1 fill:#fce4ec
```

## 7. Processing Pipeline Detail

```mermaid
flowchart TD
    START([User Uploads Image]) --> VALIDATE{Validate File}
    VALIDATE -->|Valid| PREVIEW[Display Preview]
    VALIDATE -->|Invalid| ERROR[Show Error Message]
    ERROR --> START
    
    PREVIEW --> SETTINGS[Configure Settings]
    SETTINGS --> GENERATE[Click Generate Button]
    
    GENERATE --> INIT_FD[Initialize Frame Detector]
    INIT_FD --> LOAD_MODEL[Load YOLO Model]
    LOAD_MODEL --> DETECT[Detect Manga Frames]
    DETECT --> ORDER[Order Frames by Reading Sequence]
    ORDER --> CROP[Crop Individual Frames]
    
    CROP --> INIT_OCR[Initialize OCR Processor]
    INIT_OCR --> PREPROCESS[Preprocess Frame Images]
    PREPROCESS --> OCR_EXTRACT[Extract Text with Tesseract]
    OCR_EXTRACT --> FILTER[Filter by Confidence Threshold]
    
    FILTER --> COMBINE[Combine Text Fragments]
    COMBINE --> CLEAN[Clean Text for Speech]
    CLEAN --> OPTIMIZE[Optimize for TTS]
    
    OPTIMIZE --> INIT_TTS[Initialize TTS Generator]
    INIT_TTS --> CONFIG_VOICE[Configure Voice Settings]
    CONFIG_VOICE --> GENERATE_AUDIO[Generate Audio]
    GENERATE_AUDIO --> SAVE_FILE[Save Audio File]
    
    SAVE_FILE --> DISPLAY[Display Results]
    DISPLAY --> PLAY[Audio Player]
    DISPLAY --> DOWNLOAD[Download Link]
    PLAY --> FINISH([Process Complete])
    DOWNLOAD --> FINISH
    
    style START fill:#e8f5e8
    style FINISH fill:#e8f5e8
    style VALIDATE fill:#fff3e0
    style ERROR fill:#ffebee
```

## Key Features Highlighted

- **🔄 Modular Architecture**: Each processing step is isolated in its own module
- **📊 Sequential Pipeline**: Clear data flow from image to audio
- **⚙️ Configuration Management**: Centralized settings and environment variables
- **🎯 English-Only Focus**: Simplified processing for English manga
- **🐳 Docker Support**: Containerized deployment option
- **📝 Error Handling**: Validation at each processing step
- **🎵 Audio Generation**: Local TTS without external API dependencies

## Migration Notes

The original `yolov8Model.py` functionality has been successfully integrated into the modular architecture while maintaining backward compatibility and adding enhanced features for better maintainability.
