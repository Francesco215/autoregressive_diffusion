apt update
apt install btop tmux unzip ffmpeguv  -y

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

pip install swig uv nvitop 
git config --global user.email "f.sacco@protonmail.com"
git config --global user.name "Francesco215"
alias auv='uv venv; . .venv/bin/activate'
uv venv 
. .venv/bin/activate
uv pip install -e .

uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
