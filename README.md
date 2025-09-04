# Deep Learning Specialization - Self Project

This repository contains the implementation and documentation of a self-directed project completed as part of the Deep Learning Specialization. The project was undertaken between April 25, 2025, and July 25, 2025, and showcases the application of various deep learning techniques to real-world problems.

## Project Overview

The project focuses on three main tasks, leveraging advanced neural network architectures and techniques:

- **Semantic Image Classification**: Implemented a U-Net architecture to perform semantic image classification on a self-driving car dataset, achieving an accuracy of 97.52%. This task involves classifying pixels in images to assist in autonomous driving decision-making.
- **Trigger Word Detection**: Developed a Long Short-Term Memory (LSTM) based algorithm for trigger word detection, enabling the recognition of a specific trigger word with an accuracy of 92.38%. This application is useful in voice-activated systems.
- **LSTM Translation Model with Attention**: Built an LSTM model incorporating an attention mechanism to convert human-readable dates into machine-readable formats, achieving a high level of accuracy. This demonstrates the power of sequence-to-sequence models in natural language processing tasks.

## Getting Started

### Prerequisites

To run the code in this repository, ensure you have the following dependencies installed:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aabujvb/Deep-Learning-Specialization.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Deep-Learning-Specialization
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

- **Semantic Image Classification**: Run the `semantic_image_classification.py` script with the appropriate dataset path.
- **Trigger Word Detection**: Execute the `trigger_word_detection.py` script, providing the audio dataset and trigger word.
- **LSTM Translation Model**: Use the `lstm_translation.py` script with the date conversion dataset.

## File Structure

- `semantic_image_classification.py`: Contains the U-Net implementation for image classification.
- `trigger_word_detection.py`: Implements the LSTM-based trigger word detection algorithm.
- `lstm_translation.py`: Includes the LSTM with attention mechanism for date conversion.
- `requirements.txt`: Lists all dependencies.
- `data/`: Directory for storing datasets (to be added by the user).
- `models/`: Directory for saving trained models (will be created during execution).

## Results

- **Semantic Image Classification**: Achieved 97.52% accuracy on the self-driving car dataset, demonstrating robust performance in identifying key features.
- **Trigger Word Detection**: Reached 92.38% accuracy, indicating effective recognition of the trigger word in various audio contexts.
- **LSTM Translation Model**: Successfully converted human-readable dates to machine-readable formats with high precision, leveraging the attention mechanism for improved performance.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit pull requests with your enhancements. Ensure to update the documentation and tests accordingly.

## Acknowledgments

- DeepLearning.AI for providing the Deep Learning Specialization course.
- The open-source community for tools and libraries like TensorFlow and Keras.
