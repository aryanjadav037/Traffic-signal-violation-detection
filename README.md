
# Traffic Signal Violation Detection Using Machine Learning

## Overview

This project implements an automated **Traffic Signal Violation Detection System** using machine learning techniques. The system leverages computer vision algorithms, specifically the YOLO (You Only Look Once) framework, for real-time detection of vehicles violating traffic signals. 

![My Banner](demo.jpg)


## Key Features

- **Data Collection:** Collected traffic data from road intersections and traffic cameras for building the dataset.
- **Preprocessing:** Data preprocessing steps to clean and prepare the dataset for model training.
- **Machine Learning Model:** Developed a YOLO-based model for real-time detection of vehicles that violate traffic signals.
- **Violation Detection:** Integrated license plate recognition using Tesseract OCR to identify vehicles and automatically issue fines.
- **Web Deployment:** Created a web interface for monitoring violations and viewing detection results.
- **Dashboard:** Provided a dashboard for visualizing traffic violation trends and system performance.

## Technologies Used

- **Machine Learning:** Python, TensorFlow, YOLOv5
- **Object Detection:** YOLO for real-time vehicle and signal detection
- **Web Development:** HTML, CSS, JavaScript, Flask

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aryanjadav037/traffic-signal-violation-detection.git
   cd traffic-signal-violation-detection
   ```


2. **Run the Flask app:**
   ```bash
   python app.py
   ```

## Contributions

Feel free to open issues or pull requests if you find any bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
