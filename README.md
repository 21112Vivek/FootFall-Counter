People Footfall Counter using YOLO, OpenCV & SORT

This project is a real-time People Footfall Counter that detects, tracks, and counts the number of people entering and exiting an area (such as a shop, mall, event space, office, etc.) using a CCTV or video feed.

The system uses:

✅ YOLO for person detection
✅ OpenCV for video processing
✅ SORT (Simple Online Realtime Tracking) for object tracking
✅ Custom line-crossing logic to count IN and OUT movement

🚀 Features

Real-time person detection and tracking

Counts IN and OUT crossings using a virtual line

Works on live camera feed or any video file

Uses lightweight YOLO models for fast performance

Tracks each individual using SORT

Generates clean logs of footfall data

Easy to integrate with dashboards or analytics

🧠 Tech Stack

Python

OpenCV

YOLO (Ultralytics)

NumPy

SORT Tracker

📂 Project Structure
.
├── footfall_counter.py
├── sort.py
├── requirements.txt
├── models/
│   └── yolov8n.pt
├── videos/
│   └── sample_video.mp4
└── README.md

⚙️ Setup & Installation
1. Create virtual environment
python -m venv myenv

2. Activate environment

Windows

myenv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

▶️ How to Run
Using a video file:
python footfall_counter.py --video videos/sample_video.mp4

Using a webcam:
python footfall_counter.py --webcam 0

📊 How Counting Works

A virtual line is defined in the frame

SORT assigns a unique ID to every detected person

If a person crosses the line, their direction is checked

Counter updates IN or OUT accordingly

📈 Output Example
IN: 12
OUT: 9
Current Footfall: 3

📦 Future Improvements

Dashboard (Power BI / Streamlit)

Hourly & daily analytics

Heatmap visualization

Integration with CCTV/NVR systems

📝 License — Apache License 2.0
Copyright 2025 Vivek Anand

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

🤝 Contributions

Contributions, issues, and feature requests are welcome.

⭐ Support

If you found this project useful, please consider giving it a star ⭐ on GitHub!
