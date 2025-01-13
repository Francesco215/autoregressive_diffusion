apt update
apt install btop tmux -y
pip install swig uv nvitop 
git config --global user.email "f.sacco@protonmail.com"
git config --global user.name "Francesco215"
alias auv='uv venv; . .venv/bin/activate'
uv venv 
. .venv/bin/activate
uv pip install -e .


extensions=(
    "ms-python.python"
    "github.copilot"
    "ms-toolsai.jupyter"
    "tamasfe.even-better-toml"
)

# Install each extension
for extension in "${extensions[@]}"; do
    code --install-extension "$extension"
done