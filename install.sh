#!/bin/bash
# Run from anywhere — workspace is derived automatically from the script location.
# Expected layout: <your_ws>/src/drl-sfm/install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd)"
WS_SRC=$WORKSPACE/src

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── 1. ROS 2 Humble + dev tools ────────────────────────────────────────────
if ! dpkg -l ros-humble-desktop 2>/dev/null | grep -q "^ii"; then
    info "ros-humble-desktop not found — setting up ROS 2 Humble..."

    # Enable Ubuntu Universe repository
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y universe

    # Add ROS 2 apt source
    sudo apt update && sudo apt install -y curl
    ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}')
    curl -L -o /tmp/ros2-apt-source.deb \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
    sudo dpkg -i /tmp/ros2-apt-source.deb

    sudo apt update
    sudo apt upgrade -y

    sudo apt install -y ros-humble-desktop
fi

if ! dpkg -l ros-dev-tools 2>/dev/null | grep -q "^ii"; then
    info "ros-dev-tools not found — installing..."
    sudo apt install -y ros-dev-tools
fi

# ── 2. ROS 2 apt dependencies ──────────────────────────────────────────────
info "Installing ROS 2 apt packages..."
sudo apt update -qq
sudo apt install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-turtlebot3-gazebo \
    "ros-humble-rtabmap*" \
    ros-humble-tf-transformations

# ── 3. Python pip dependencies ─────────────────────────────────────────────
if ! python3 -m pip --version &>/dev/null; then
    info "pip not found — installing python3-pip..."
    sudo apt install -y python3-pip
fi
info "Installing Python pip packages..."
pip3 install stable-baselines3 sb3-contrib tensorboard

warn "numpy<2 will be installed. This downgrades numpy if a newer version is present,"
warn "which may affect other Python packages on your system."
read -rp "Press Enter to continue or Ctrl+C to abort..."
pip3 install "numpy<2"

# PyTorch: default CPU/CUDA auto-detect build.
# For a specific CUDA version use e.g.:
#   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install torch torchvision torchaudio

# ── 4. lightsfm ────────────────────────────────────────────────────────────
info "Building and installing lightsfm..."
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
git clone https://github.com/robotics-upo/lightsfm.git "$TMP_DIR/lightsfm"
pushd "$TMP_DIR/lightsfm" > /dev/null
make
sudo make install
popd > /dev/null

# ── 5. Clone workspace dependencies ────────────────────────────────────────
info "Cloning workspace dependencies into $WS_SRC..."
mkdir -p "$WS_SRC"

clone_if_missing() {
    local url=$1 dest=$2 branch=${3:-}
    if [ -d "$dest" ]; then
        warn "Already exists, skipping: $(basename "$dest")"
    elif [ -n "$branch" ]; then
        git clone --branch "$branch" "$url" "$dest"
    else
        git clone "$url" "$dest"
    fi
}

clone_if_missing https://github.com/wg-perception/people.git \
    "$WS_SRC/people" ros2

clone_if_missing https://github.com/aws-robotics/aws-robomaker-hospital-world.git \
    "$WS_SRC/aws-robomaker-hospital-world" ros2

clone_if_missing https://github.com/aws-robotics/aws-robomaker-small-house-world.git \
    "$WS_SRC/aws-robomaker-small-house-world" ros2

clone_if_missing https://github.com/Kalemat96/hunav_gazebo_wrapper.git \
    "$WS_SRC/hunav_gazebo_wrapper"

clone_if_missing https://github.com/Kalemat96/hunav_sim.git \
    "$WS_SRC/hunav_sim"

# ── 6. Build workspace ──────────────────────────────────────────────────────
info "Building workspace..."
source /opt/ros/humble/setup.bash
cd "$WORKSPACE"
colcon build
source "$WORKSPACE/install/setup.bash"

# ── 7. Environment setup in ~/.bashrc ───────────────────────────────────────
info "Setting up environment variables in ~/.bashrc..."

append_if_missing() {
    grep -qxF "$1" ~/.bashrc || echo "$1" >> ~/.bashrc
}

append_if_missing 'source /opt/ros/humble/setup.bash'
append_if_missing "source $WORKSPACE/install/setup.bash"
append_if_missing 'source /usr/share/gazebo/setup.sh'
append_if_missing 'export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models'
append_if_missing "export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:$WORKSPACE/src/aws-robomaker-hospital-world/models"
append_if_missing "export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:$WORKSPACE/src/aws-robomaker-hospital-world/fuel_models"
append_if_missing "export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:$WORKSPACE/src/aws-robomaker-small-house-world/models"
append_if_missing "export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:$WORKSPACE/src/drl-sfm/hunav_rl/models"

info "Done! Restart your terminal or run:  source ~/.bashrc"
