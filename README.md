# 🛡️ Acoustic Shield - AI-Powered Vehicle Emergency Sound Detection

> **Every second counts in emergency situations.** An intelligent system that can hear danger before it's too late.

[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-FF9900?logo=amazon-aws)](https://aws.amazon.com/sagemaker/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)](https://pytorch.org/)

## � Project Links

- **🎨 Frontend Repository**: [ACM_Hack_FE](https://github.com/thesid42/ACM_Hack_FE)
- **⚙️ Backend Repository**: [acousticshield-aws-backend](https://github.com/akrai37/acousticshield-aws-backend) (You are here)

## �📹 Demo Video

https://github.com/user-attachments/assets/0a2d00ee-a6e2-4b76-9ff2-7738aa5ece70

## 💡 Inspiration

Traditional collision detection systems rely on cameras and sensors, but what if we could leverage the **acoustic signature of dangerous driving scenarios**? The sounds produced during vehicle emergencies—tire screeches, collision impacts, emergency braking—contain critical information that could save lives.

**Acoustic Shield** is an AI-powered audio classification system that can identify and categorize vehicle-related emergency sounds in real-time, hearing danger before it's too late.

## 🏗️ What We Built

Acoustic Shield is a complete **end-to-end machine learning pipeline** deployed on AWS that:

1. **Generates synthetic training data** representing different vehicle emergency scenarios
2. **Trains a deep learning model** (wav2vec2) to classify audio into 4 categories:
   - 🟢 **Normal** - Regular driving conditions
   - 🟡 **TireSkid** - Sudden tire skidding sounds
   - 🟠 **EmergencyBraking** - Hard braking events
   - 🔴 **CollisionImminent** - Sounds indicating imminent collision
3. **Deploys a production-ready REST API** on AWS SageMaker for real-time inference
4. **Processes audio streams** at 16 kHz sampling rate with sub-second latency

### Technical Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Audio Input    │─────▶│  SageMaker       │─────▶│  Classification │
│  (WAV/Stream)   │      │  Endpoint        │      │  Results        │
│  16 kHz         │      │  wav2vec2        │      │  + Confidence   │
└─────────────────┘      └──────────────────┘      └─────────────────┘
         ▲                        ▲                         │
         │                        │                         │
         │                 ┌──────┴──────┐                 ▼
         │                 │  S3 Bucket  │         ┌───────────────┐
         │                 │  - Training │         │  Alert System │
         └─────────────────│  - Models   │         │  (Future)     │
                           └─────────────┘         └───────────────┘
```

## 🚀 Quick Start

### Prerequisites
- AWS Account with SageMaker access
- Python 3.10+
- AWS CLI configured

### Installation

```bash
# Clone the repository
git clone https://github.com/akrai37/acousticshield-aws-backend.git
cd acousticshield-aws-backend

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
aws configure
```

### Usage

#### 1. Generate Training Data
Open and run `notebooks/01_build_training_data.ipynb` to:
- Extract crash hotspots from GeoJSON data
- Enrich with weather data (Open-Meteo API)
- Generate synthetic risk events
- Create audio recipes
- Launch SageMaker processing job to generate WAV files

#### 2. Train and Deploy Model
Open and run `notebooks/02_train_and_deploy.ipynb` to:
- Split data into train/validation sets (80/20)
- Fine-tune wav2vec2 model on AWS SageMaker
- Deploy real-time inference endpoint
- Test with sample audio files

## 📊 Results & Performance

| Metric | Value |
|--------|-------|
| **Training Time** | ~15-20 minutes (1 epoch on ml.g4dn.xlarge) |
| **Inference Latency** | <500ms per audio clip |
| **Model Accuracy** | 60-75% (sufficient for demo) |
| **Training Cost** | ~$0.25 per run |
| **Inference Cost** | $0.23/hour (with auto-scaling) |

## 🛠️ How We Built It

### 1. Data Pipeline Engineering

We created a sophisticated **synthetic audio generation system**:

- **Recipe-based synthesis**: Flexible recipe system with configurable parameters (frequency, duration, amplitude)
- **Variation engine**: Built-in randomization (±5%) ensures model robustness
- **Scalability**: Generated 1000+ audio files across 4 classes
- **Weather enrichment**: Integrated weather data API for contextual information

**Variation Formula:**
```
f_actual = f_base × (1 + U(-0.05, 0.05))
```
Where U(a, b) represents uniform random distribution between a and b.

### 2. Machine Learning Model

- **Base Model**: Facebook's wav2vec2-base (300M parameters)
- **Fine-tuning Strategy**:
  - Learning rate: α = 5 × 10⁻⁵
  - Batch size: 16 (optimized for GPU memory)
  - Training epochs: 1 (hackathon speed optimization)
  - Warmup steps: 50
  - Data Split: 80/20 train-validation with stratified sampling

- **Evaluation Metrics**:
  - Accuracy
  - F1-score (macro-averaged)
  - Per-class precision/recall

### 3. AWS Infrastructure

**Training Pipeline:**
- AWS SageMaker Training Jobs with GPU instances (ml.g4dn.xlarge)
- Custom training script using HuggingFace Transformers
- Automatic hyperparameter tuning and model checkpointing
- CloudWatch integration for real-time monitoring

**Inference Pipeline:**
- SageMaker Real-time Endpoints with auto-scaling
- Custom inference handler supporting audio/wav content type
- Sub-second latency (<500ms for typical 1-3 second audio clips)
- JSON response format for easy API integration

### 4. Storage & Organization

```
s3://acousticshield-ml/
├── train/              # Original training data
├── train_split/        # 80% training set
├── val/                # 20% validation set
└── models/             # Trained model artifacts

s3://acousticshield-raw/
├── crash_hotspots/     # GeoJSON crash data
├── risk_events/        # Synthetic event data
└── prompts/            # Audio generation recipes
```

## 🎓 What We Learned

### Technical Learnings

1. **Audio Processing at Scale**
   - Importance of consistent sampling rates (16 kHz)
   - Audio resampling significantly impacts model accuracy
   - Trade-offs between audio quality and processing speed

2. **AWS SageMaker Deep Dive**
   - Mastered SageMaker's HuggingFace container ecosystem
   - Learned about instance quotas and ResourceLimitExceeded errors
   - Importance of custom inference code for production deployment

3. **Model Optimization**
   - 1 epoch can be sufficient for demo-quality models in hackathons
   - Batch size impact on GPU utilization (8 → 16 = 2x faster)
   - Trade-off between model accuracy and training time

4. **Data Engineering**
   - Handle S3 pagination for large datasets (>1000 files)
   - Importance of data validation and stratified splitting
   - Value of synthetic data when real-world data is limited

### Hackathon-Specific Lessons

> **"In hackathons, a working demo beats a perfect solution every time."**

- ⚡ **Iterate quickly**: Our 1-epoch model strategy saved hours
- 📝 **Document thoroughly**: Future us (and others) will thank us
- 🛡️ **Handle errors gracefully**: AWS quotas will surprise you
- 🎯 **Optimize for demo**: Focus on end-to-end functionality first
- 📚 **Learn from failures**: Every error taught us something valuable

## 🚧 Challenges We Faced

### 1. AWS Quota Limitations
**Challenge**: Hit instance quota limits on multiple GPU instance types.

**Solution**:
- Created "Emergency Stop" cell to clean up stuck training jobs
- Documented 5+ alternative instance types with availability
- Switched to ml.g4dn.xlarge (most reliable for new accounts)

### 2. Training Job Interruption
**Challenge**: Accidentally interrupted training at 1.67/4 epochs, losing progress.

**Solution**:
- Learned that SageMaker training jobs continue after notebook interruption
- Implemented proper job monitoring and graceful stopping
- Optimized to 1-epoch training (10-15 min) for hackathon speed

### 3. Audio Format Compatibility
**Challenge**: Pre-trained models don't support audio/wav content type without custom code.

**Solution**:
- Wrote custom `inference.py` handler supporting direct audio/wav input
- Implemented automatic audio resampling to 16 kHz
- Added comprehensive error handling for various audio formats

### 4. Version Compatibility Issues
**Challenge**: Transformers 4.44 not supported by SageMaker, PyTorch 2.3 incompatible.

**Solution**:
- Documented compatible versions: Transformers 4.28 + PyTorch 2.0
- Created detailed version matrix in configuration comments
- Fixed deprecated parameter names in training script

### 5. Data at Scale
**Challenge**: S3 `list_objects_v2` has 1000-file limit.

**Solution**:
- Implemented S3 paginator for unlimited file handling
- Added progress tracking for large dataset operations
- Ensured proper stratification across all files

## 🌟 Real-World Applications

- 🏙️ **Smart City Safety**: Deploy in urban areas to detect accidents in real-time
- 🚛 **Fleet Management**: Monitor commercial vehicles for emergency events
- 💼 **Insurance**: Automated accident detection and reporting
- 🚑 **Emergency Response**: Alert first responders before 911 calls

## 🔮 What's Next

- [ ] **Real-World Data Collection**: Partner with fleet operators for actual vehicle sound data
- [ ] **Multi-Modal Integration**: Combine audio with video and sensor data
- [ ] **Edge Deployment**: Optimize model for on-device inference (TensorFlow Lite/ONNX)
- [ ] **Temporal Analysis**: Detect sequences of events (skid → brake → collision)
- [ ] **Alert System**: Real-time notifications to emergency services


## 📁 Repository Structure

```
acousticshield-aws-backend/
├── data_pipeline/              # Audio generation & processing
│   ├── __init__.py
│   ├── hotspot_extractor.py   # Extract top crash locations
│   ├── weather_enricher.py    # Fetch weather data (Open-Meteo)
│   ├── risk_event_synth.py    # Create synthetic risk events
│   ├── recipe_builder.py      # Build audio generation specs
│   └── s3_utils.py            # S3 operations (region-agnostic)
├── processing/                 # SageMaker processing scripts
│   └── augment.py             # Audio generation script
├── training/                   # ML training & inference code
│   ├── train.py               # SageMaker training script
│   ├── inference.py           # Custom inference handler
│   ├── README.md
│   └── QUICKREF.md
├── notebooks/                  # Jupyter notebooks
│   ├── 01_build_training_data.ipynb   # Data pipeline orchestration
│   └── 02_train_and_deploy.ipynb     # Model training & deployment
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── [Various documentation files]
```

## 🔧 Technical Details

### Risk Types & Classification


- **Normal**: Standard driving conditions
- **TireSkid**: Slippery conditions, potential loss of traction
- **EmergencyBraking**: Sudden stop required
- **CollisionImminent**: Immediate crash danger

### Audio Generation Parameters

Each audio file contains:
- **Ambient noise**: Pink noise based on weather conditions
- **Engine sound**: Low-frequency rumble (40-80 Hz)
- **Tire noise**: Road friction sounds (intensifies for skids)
- **Alert sounds**: Beeping warnings (800-1200 Hz)

Parameters are dynamically adjusted based on:
- Risk type (Normal → CollisionImminent)
- Weather conditions (rain, wind, temperature)
- Time of day (morning, afternoon, evening, night)

### SageMaker Processing Job

Uses PyTorch CPU container:
- **Instance**: `ml.m5.xlarge`
- **Image**: `pytorch-training:2.0.1-cpu-py310`
- **Script**: `processing/augment.py`

The processing job:
1. Reads recipe JSON from S3
2. Generates synthetic audio (5 seconds per event)
3. Applies audio parameters (engine, tire noise, alerts)
4. Writes WAV files to S3

### Region Handling

All components are **region-agnostic**:
- Bucket regions auto-detected from S3
- SageMaker sessions use detected region
- No hard-coded regional endpoints


## ⚙️ Configuration

All parameters are region-agnostic and configurable:

```python
RAW_BUCKET = 'acousticshield-raw'
ML_BUCKET = 'acousticshield-ml'
CRASH_FILE_KEY = 'crash_hotspots/sanjose_crashes.geojson'
SAGEMAKER_ROLE = 'role-sagemaker-processing'
```

### S3 Structure

```
s3://acousticshield-raw/
├── crash_hotspots/
│   └── sanjose_crashes.geojson
├── risk_events/
│   └── risk_events.json
└── prompts/
    └── audio_recipes.json

s3://acousticshield-ml/
├── train/              # Original training data
│   ├── evt_00001_normal.wav
│   ├── evt_00002_tireskid.wav
│   └── ...
├── train_split/        # 80% training set
├── val/                # 20% validation set
└── models/             # Trained model artifacts
```

## 📋 Output Formats

### Risk Event JSON
```json
{
  "event_id": "evt_00001",
  "risk_type": "TireSkid",
  "risk_score": 65.5,
  "road_name": "Main St",
  "weather_risk": "high",
  "time_category": "evening"
}
```

### Audio Recipe JSON
```json
{
  "recipe_id": "recipe_evt_00001",
  "risk_type": "TireSkid",
  "audio_parameters": {
    "ambient_level": 0.4,
    "engine_intensity": 0.7,
    "tire_noise": 0.9,
    "alert_level": 0.4,
    "duration_seconds": 5.0,
    "sample_rate": 22050
  },
  "output": {
    "filename": "evt_00001_tireskid.wav"
  }
}
```


## 📦 Dependencies

Core dependencies:
- `boto3` - AWS SDK for Python
- `sagemaker` - SageMaker Python SDK
- `transformers` - HuggingFace Transformers library
- `torch` - PyTorch deep learning framework
- `datasets` - HuggingFace datasets library
- `numpy` - Numerical operations
- `scipy` - Audio signal processing
- `librosa` - Audio analysis library
- `soundfile` - Audio file I/O
- `requests` - HTTP client (Open-Meteo API)

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🔐 IAM Roles & Permissions

### Required IAM Roles

**SageMaker Execution Role** (`role-sagemaker-processing`):
- S3 read/write access to `acousticshield-raw` and `acousticshield-ml` buckets
- SageMaker full access
- CloudWatch Logs write access

**Trust Policies**:
- `trust-sagemaker.json` - SageMaker service trust policy
- `trust-lambda.json` - Lambda service trust policy (if using Lambda triggers)

### S3 Bucket Policy

See `policy-s3-acousticshield.json` for S3 bucket policy configuration.

## 🔑 API Keys

**No API keys required!** 🎉

- Open-Meteo API is free and doesn't require authentication
- AWS credentials handled via AWS CLI or IAM roles


## 🐛 Troubleshooting

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### S3 Access Errors
Check AWS credentials and IAM permissions:
```bash
aws sts get-caller-identity
aws s3 ls s3://acousticshield-ml/
```

### SageMaker Job Failures
Check CloudWatch logs in AWS Console:
1. Navigate to **SageMaker → Processing Jobs** (or **Training Jobs**)
2. Click on failed job
3. View logs in CloudWatch

### GPU Instance Quota Issues
If you encounter `ResourceLimitExceeded` errors:
1. Check your AWS quota limits in Service Quotas console
2. Try alternative instance types:
   - `ml.g4dn.xlarge` (most reliable for new accounts)
   - `ml.m5.xlarge` (CPU fallback)
   - `ml.p3.2xlarge` (if quota available)
3. Request quota increase for your preferred instance type

### Audio Format Issues
Ensure audio files are:
- WAV format
- 16 kHz sampling rate
- Mono channel
- 16-bit PCM encoding

Convert if needed:
```python
import librosa
import soundfile as sf

# Load and resample
audio, _ = librosa.load('input.wav', sr=16000, mono=True)
sf.write('output.wav', audio, 16000)
```

### Version Compatibility
Stick to tested versions:
- Python: 3.10
- PyTorch: 2.0.1
- Transformers: 4.28
- SageMaker: Latest stable

## 🙏 Acknowledgments

- **AWS SageMaker**: For providing powerful ML infrastructure
- **HuggingFace**: For wav2vec2 and the Transformers library
- **Open-Source Community**: For countless tutorials and documentation
- **Hackathon Organizers**: For creating this amazing learning opportunity

## 📄 Additional Documentation

- [`QUICKSTART.md`](QUICKSTART.md) - Quick start guide
- [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md) - Detailed deployment instructions
- [`PROJECT_INFO.md`](PROJECT_INFO.md) - Project information
- [`PROJECT_STORY.md`](PROJECT_STORY.md) - Development journey
- [`TECHNOLOGIES.md`](TECHNOLOGIES.md) - Technology stack details
- [`SAGEMAKER_SUMMARY.md`](SAGEMAKER_SUMMARY.md) - SageMaker configuration
- [`training/README.md`](training/README.md) - Training script documentation
- [`training/QUICKREF.md`](training/QUICKREF.md) - Quick reference guide

## 👥 Team

Built with ❤️ by the **Acoustic Shield Team** for the AWS Hackathon 2025.

## 📜 License

Copyright © 2025 Acoustic Shield Team. All rights reserved.

---

**⭐ Star this repo if you found it helpful!**

**🐛 Found a bug?** Open an issue!

**💡 Have an idea?** We'd love to hear it!

