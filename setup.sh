apt update
apt install btop tmux unzip ffmpeg  -y

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf awscliv2.zip aws/

git config --global user.email "f.sacco@protonmail.com"
git config --global user.name "Francesco215"

pip install swig uv nvitop 
alias auv='uv venv; . .venv/bin/activate'

uv venv 
. .venv/bin/activate
uv pip install -e .
