sudo apt update
sudo apt install btop tmux ffmpeg pip -y

pip install swig uv nvitop 
alias auv='uv venv; . .venv/bin/activate'

uv venv 
. .venv/bin/activate
uv pip install -e .
