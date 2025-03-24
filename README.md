# Facial Emotion Detection System

A deep learning-based facial emotion detection system that can recognize emotions in real-time and recommend music based on the detected emotion.

## Features

- Real-time facial emotion detection using webcam
- Emotion classification into 5 categories: Angry, Fear, Happy, Sad, Neutral
- Music recommendation system based on detected emotion
- User-friendly Gradio web interface
- Support for diverse music recommendations (English, K-pop, and Burmese songs)

## Project Structure

```plaintext
Facial_Emotion_Detection_2/
├── data/
│   └── FER2013/           # Dataset directory
│       ├── train/         # Training dataset
│       └── test/          # Testing dataset
├── haarcascades/
│   └── haarcascade_frontalface_default.xml  # Face detection model
├── model/
│   ├── emotion_cnn.py     # CNN model architecture
│   └── emotion_cnn_best.pth  # Trained model weights
├── gradio_app.py          # Gradio web interface
├── real_time_detection.py # Real-time detection script
├── train.py              # Model training script
└── requirements.txt      # Project dependencies
```

## Technologies Used

- Python 3.x
- PyTorch for deep learning
- OpenCV for image processing
- Gradio for web interface
- Matplotlib for visualization

## Model Architecture

The emotion detection model uses a CNN architecture with residual blocks:

- Initial convolution layer
- 3 Residual blocks with increasing channels (64, 128, 256)
- Dropout layers for regularization
- Batch normalization for stable training
- Fully connected layers for final classification

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

Run the Gradio web application:

```bash
python gradio_app.py
```

### Real-time Detection

For real-time webcam detection:

```bash
python real_time_detection.py
```

### Training

To train the model:

```bash
python train.py
```

## Music Recommendation System

The system includes a curated list of songs for each emotion:

- Diverse selection of English, K-pop, and Burmese songs
- Emotion-appropriate playlists
- Direct YouTube links for easy access

## Dataset

The project uses the FER2013 dataset for training and testing, which includes facial expressions categorized into different emotions.

## Requirements

Major dependencies include:

- torch==2.6.0
- opencv-python==4.11.0.86
- numpy==2.2.3
- pandas==2.2.3
- matplotlib==3.10.1

For a complete list of dependencies, see `requirements.txt`.

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

This project is open-source and available under the MIT License.