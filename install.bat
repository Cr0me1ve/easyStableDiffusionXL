python -m venv .\venv
call .\venv\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python install.py