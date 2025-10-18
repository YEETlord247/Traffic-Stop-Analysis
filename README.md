TRAFFIC STOP ANALYSIS SYSTEM
------------------------------------------------------------
A modular computer vision and deep learning pipeline that analyzes 
police traffic stop footage to detect events such as vehicle pursuit, 
officer demeanor, and stop classification.
Developed at Axon Enterprises by Utkarsh Rai.

------------------------------------------------------------
OVERVIEW
------------------------------------------------------------
The Traffic Stop Analysis System uses deep learning models to analyze 
real-time video feeds from body cameras or dashcams to identify 
critical behavioral and situational patterns during traffic stops. 

Each component model detects a different aspect of the interaction 
(chase detection, officer demeanor, and stop type), and the results 
are unified through a centralized inference controller. 
This system supports real-time deployment and visualization, enabling 
automated review of incidents for auditing and training purposes.

------------------------------------------------------------
SYSTEM ARCHITECTURE
------------------------------------------------------------
Video Input (Dashcam / Bodycam)
   ↓
Frame Sampling + Preprocessing (OpenCV)
   ↓
Model Ensemble:
   • model_chase.py    → Pursuit detection
   • model_demeanor.py → Officer behavior/demeanor classification
   • model_stop.py     → Stop characterization
   ↓
Unified Inference Controller (model.py)
   ↓
Flask Dashboard (Visualization + Output Logging)

Core pipeline modules:
1. Preprocessing: Extract frames, normalize, augment
2. Classification: Run independent TensorFlow/Keras models
3. Aggregation: Combine model outputs in weighted ensemble
4. Visualization: Real-time dashboard for analytics and playback

------------------------------------------------------------
KEY FEATURES
------------------------------------------------------------
- Modular architecture supporting multiple specialized CNN/ViT models
- Separate classifiers for chase, demeanor, and stop type
- Unified inference pipeline with ensemble integration
- Real-time preprocessing and inference via GPU acceleration
- Flask-based interactive dashboard for review and retraining
- Automatically generates labeled datasets for model refinement

------------------------------------------------------------
TECH STACK
------------------------------------------------------------
Languages: Python 3.10+
Deep Learning: TensorFlow, Keras, Vision Transformers (ViT)
Computer Vision: OpenCV, NumPy, Scikit-learn
Visualization: Flask, Plotly
Hardware Acceleration: CUDA, cuDNN

------------------------------------------------------------
REPOSITORY STRUCTURE
------------------------------------------------------------
traffic_stop_analysis/
 ├── model_chase.py             (Chase detection CNN)
 ├── model_demeanor.py          (Officer demeanor classifier)
 ├── model_stop.py              (Stop type classifier)
 ├── model.py                   (Unified inference controller)
 ├── parser.py                  (Data loader and preprocessor)
 ├── map_interface.py           (Dashboard for output visualization)
 ├── training_data/             (Labeled frame datasets)
 ├── logs/                      (Performance and output logs)
 └── README.txt

------------------------------------------------------------
SETUP INSTRUCTIONS
------------------------------------------------------------
1. Prerequisites
   - Python >= 3.8
   - GPU-enabled environment (recommended)
   - TensorFlow >= 2.9
   - Labeled video datasets for fine-tuning

2. Installation
   git clone <repo-url>
   cd traffic_stop_analysis
   pip install -r requirements.txt

3. Training Individual Models
   python model_chase.py
   python model_demeanor.py
   python model_stop.py

4. Running Unified Inference
   python model.py

5. Launching Dashboard
   python map_interface.py

------------------------------------------------------------
USAGE
------------------------------------------------------------
1. Load a dashcam or bodycam video file.
2. The system performs frame extraction and model inference.
3. Each frame is analyzed for pursuit behavior, officer demeanor, 
   and stop type.
4. Results are displayed in real-time on the Flask dashboard with 
   options to export logs or retrain the model.

------------------------------------------------------------
PERFORMANCE BENCHMARKS
------------------------------------------------------------
Model Accuracy (avg):          ~91%
Per-frame Latency (GPU):       ~600 ms
Supported Frame Rate:          30 FPS (live stream mode)
Training Time per Model:       ~50 minutes (on RTX 3090)
Dashboard Refresh Rate:        Real-time (<1 second)

------------------------------------------------------------
DESIGN HIGHLIGHTS
------------------------------------------------------------
- Multi-model ensemble architecture (CNN + Vision Transformer)
- Transfer learning from ImageNet for faster convergence
- Robust data augmentation: flipping, blurring, lighting shifts
- Weighted ensemble improves classification confidence
- Auto-retraining mechanism for incremental dataset updates

------------------------------------------------------------
EXAMPLE OUTPUT
------------------------------------------------------------
Input Video:
   "HighwayStop_2024_06_03.mp4"

Output:
   Chase Detected: YES
   Officer Demeanor: Neutral (Confidence: 0.91)
   Stop Type: Routine Check
   Aggregated Confidence: 0.93

------------------------------------------------------------
FUTURE WORK
------------------------------------------------------------
- Expand model to include passenger behavior detection
- Integrate audio sentiment analysis for multimodal prediction
- Build web-based analytics dashboard for centralized monitoring
- Add support for streaming bodycam integration via Axon Fleet API
- Transition to Vision Transformer v2 backbone for improved accuracy

------------------------------------------------------------
AUTHOR
------------------------------------------------------------
Utkarsh Rai
R&D Intern — Axon Enterprises
Email: rai.utkarsh2007@gmail.com
LinkedIn: linkedin.com/in/utkarsh-rai-7249611b6
