# Python-OpenCV-Emotion-Detection

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

python -m pip install deepface==0.0.104

python -m pip install onnxruntime

python -m pip uninstall opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless -y

python -m pip install "opencv-contrib-python<5"