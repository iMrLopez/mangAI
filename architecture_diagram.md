# MangAI Architecture Diagrams

This document contains comprehensive Mermaid diagrams showing the architecture and data flow of the MangAI application.

## 1. High-Level System Architecture

```mermaid
graph TD
    A[User Interface - Streamlit] --> B[Frame Detector Module]
    B --> C[OCR Processor Module]
    C --> D[Text Processor]
    D --> E[TTS Generator Module]
    E --> F[Audio Output]
    
    G[Config Management] --> A
    G --> B
    G --> C
    G --> D
    G --> E
    
    H[YOLO Models] --> B
    I[Tesseract OCR] --> C
    J[pyttsx3 Engine] --> E
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#f1f8e9
    style G fill:#fff8e1
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
            TXT[Text Processor<br/>Built into app.py]
            TTS[TTS Generator<br/>modules/tts_generator.py]
        end
        
        subgraph "External Dependencies"
            YOLO[YOLO Models<br/>ultralytics]
            TESS[Tesseract OCR<br/>pytesseract]
            PYTS[pyttsx3<br/>TTS Engine]
        end
        
        subgraph "File System"
            MODELS[models/<br/>*.pt files]
            IMAGES[images/<br/>test images]
            AUDIO[audio_output/<br/>generated audio]
        end
    end
    
    UI --> CONFIG
    CONFIG --> ENV
    UI --> FD
    FD --> OCR
    OCR --> TXT
    TXT --> TTS
    
    FD --> YOLO
    FD --> MODELS
    OCR --> TESS
    TTS --> PYTS
    TTS --> AUDIO
    
    UI -.-> IMAGES
    
    style UI fill:#e3f2fd
    style CONFIG fill:#f3e5f5
    style ENV fill:#e8f5e8
    style FD fill:#fff3e0
    style OCR fill:#fce4ec
    style TXT fill:#f1f8e9
    style TTS fill:#fff8e1
```

## 3. Data Flow Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Streamlit UI
    participant FD as Frame Detector
    participant OCR as OCR Processor
    participant TXT as Text Processor
    participant TTS as TTS Generator
    participant FS as File System
    
    U->>UI: Upload manga image
    UI->>UI: Validate file & display preview
    U->>UI: Configure settings (model, speech rate)
    UI->>UI: Update configuration
    U->>UI: Click "Generate Audio"
    
    UI->>FD: Process image for frame detection
    FD->>FD: Load YOLO model
    FD->>FD: Detect manga frames
    FD->>FD: Order frames by reading sequence
    FD-->>UI: Return ordered frame coordinates
    
    UI->>OCR: Extract text from each frame
    OCR->>OCR: Preprocess frame images
    OCR->>OCR: Run Tesseract OCR (English)
    OCR->>OCR: Filter by confidence threshold
    OCR-->>UI: Return text fragments
    
    UI->>TXT: Combine text fragments
    TXT->>TXT: Clean text for speech
    TXT->>TXT: Optimize for TTS
    TXT-->>UI: Return processed text
    
    UI->>TTS: Generate audio from text
    TTS->>TTS: Configure pyttsx3 engine
    TTS->>TTS: Generate speech audio
    TTS->>FS: Save audio file
    TTS-->>UI: Return audio file path
    
    UI->>U: Display results & audio player
    U->>UI: Play audio or download file
```

## 4. Module Dependencies

```mermaid
graph LR
    subgraph "app.py Dependencies"
        APP[app.py]
        APP --> ST[streamlit]
        APP --> PIL[PIL.Image]
        APP --> CONFIG2[config.py]
        APP --> FD2[modules.frame_detector]
        APP --> OCR2[modules.ocr_processor]
        APP --> TTS2[modules.tts_generator]
    end
    
    subgraph "Frame Detector Dependencies"
        FD2 --> CV2[cv2 - OpenCV]
        FD2 --> NP[numpy]
        FD2 --> YOLO2[ultralytics.YOLO]
        FD2 --> CONFIG3[config.Config]
    end
    
    subgraph "OCR Processor Dependencies"
        OCR2 --> CV2
        OCR2 --> NP
        OCR2 --> PYTESS[pytesseract]
        OCR2 --> TEMP[tempfile]
        OCR2 --> OS[os]
    end
    
    subgraph "TTS Generator Dependencies"
        TTS2 --> OS
        TTS2 --> TEMP
        TTS2 --> UUID[uuid]
        TTS2 --> PYTTSX[pyttsx3]
    end
    
    style APP fill:#e1f5fe
    style FD2 fill:#f3e5f5
    style OCR2 fill:#e8f5e8
    style TTS2 fill:#fce4ec
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
        [*] --> FrameDetection
        FrameDetection --> OCRExtraction
        OCRExtraction --> TextProcessing
        TextProcessing --> TTSGeneration
        TTSGeneration --> [*]
    }
    
    ProcessingPipeline --> ResultsDisplay
    ResultsDisplay --> AudioPlayback
    AudioPlayback --> FileDownload
    FileDownload --> [*]
    
    ImageUpload : User uploads manga image
    FileValidation : System validates file type/size
    PreviewDisplay : Display image preview
    ConfigurationSettings : User configures model & speech rate
    ReadyToProcess : Ready state with Generate button
    FrameDetection : YOLO detects manga frames
    OCRExtraction : Tesseract extracts English text
    TextProcessing : Clean & combine text
    TTSGeneration : Generate audio from text
    ResultsDisplay : Show statistics & preview
    AudioPlayback : Play generated audio
    FileDownload : Download audio file
```

## 6. File System Organization

```mermaid
graph TD
    ROOT[mangAI/] --> APP[app.py]
    ROOT --> CONFIG[config.py]
    ROOT --> REQ[requirements.txt]
    ROOT --> DOCKER[Dockerfile]
    ROOT --> COMPOSE[docker-compose.yml]
    
    ROOT --> MODULES[modules/]
    MODULES --> FD_PY[frame_detector.py]
    MODULES --> OCR_PY[ocr_processor.py]
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
    AUDIO --> GENERATED[manga_audio_*.wav]
    
    ROOT --> DATASET[dataset/]
    DATASET --> MANGA109[Manga109s_released_2023_12_07.zip]
    
    style ROOT fill:#e1f5fe
    style MODULES fill:#f3e5f5
    style MODELS fill:#e8f5e8
    style IMAGES fill:#fff3e0
    style AUDIO fill:#fce4ec
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
