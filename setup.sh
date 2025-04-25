apt update
apt install btop tmux unzip ffmpeg pip -y

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
. ./aws/install
rm -rf awscliv2.zip aws/

# i want to get rid of this two lines
# git config --global user.email "f.sacco@protonmail.com"
# git config --global user.name "Francesco215"

pip install swig uv nvitop 
alias auv='uv venv; . .venv/bin/activate'

uv venv 
. .venv/bin/activate
uv pip install -e .
