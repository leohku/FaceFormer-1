sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt install python3.9
sudo apt install python3.9-distutils
virtualenv --python=/usr/bin/python3.9 venv
source venv/bin/activate
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install --upgrade pyopengl==3.1.5