#!/bin/bash

# === Unified Cluster Node Setup Script ===
# Usage:
#   ./setup.sh ray head
#   ./setup.sh ray worker <HEAD_IP>
#   ./setup.sh dask head
#   ./setup.sh dask worker <HEAD_IP>
#   ./setup.sh spark head
#   ./setup.sh spark worker <MASTER_IP>
#
# "head"   ~ Ray head / Dask scheduler / Spark master
# "worker" ~ Ray worker / Dask worker / Spark worker

set -e

# --- Parameters ---
FRAMEWORK="$1"   # "ray" | "dask" | "spark"
ROLE="$2"        # "head" | "worker"
HEAD_IP="$3"     # required for worker (head / scheduler / master IP)

if [[ -z "$FRAMEWORK" || -z "$ROLE" ]]; then
  echo "Usage: $0 <ray|dask|spark> <head|worker> [HEAD_IP]"
  exit 1
fi

if [[ "$ROLE" == "worker" && -z "$HEAD_IP" ]]; then
  echo "❌ ERROR: For 'worker' role you must provide the head/scheduler/master IP."
  echo "Example: $0 ray worker 10.0.0.5"
  exit 1
fi

if [[ "$FRAMEWORK" != "ray" && "$FRAMEWORK" != "dask" && "$FRAMEWORK" != "spark" ]]; then
  echo "❌ ERROR: FRAMEWORK must be 'ray', 'dask', or 'spark'."
  exit 1
fi

# --- Names & paths ---
ENV_DIR="$HOME/${FRAMEWORK}-env"
SPARK_VERSION="3.5.7"
SPARK_PACKAGE="spark-${SPARK_VERSION}-bin-hadoop3"
SPARK_DIR="$HOME/$SPARK_PACKAGE"

echo "🔧 Setting up $FRAMEWORK $ROLE node..."
echo "📂 Virtualenv directory: $ENV_DIR"

# --- Install system dependencies ---
echo "🔧 Installing base system dependencies..."
sudo apt update -y
sudo apt install -y python3-pip python3-venv tmux

# Spark-specific system deps
if [[ "$FRAMEWORK" == "spark" ]]; then
  echo "🔧 Installing Java for Spark..."
  sudo apt install -y openjdk-17-jdk wget
fi

# --- Create and activate virtual environment ---
if [[ ! -d "$ENV_DIR" ]]; then
  echo "📦 Creating Python virtual environment..."
  python3 -m venv "$ENV_DIR"
fi

echo "📦 Activating virtual environment..."
source "$ENV_DIR/bin/activate"

# --- Install common Python dependencies ---
echo "⬆️  Upgrading pip and installing common Python packages..."
pip install --upgrade pip
pip install notebook pandas scikit-learn google-cloud-storage

# --- Framework-specific installation ---
if [[ "$FRAMEWORK" == "ray" ]]; then
  echo "⬇️  Installing Ray..."
  pip install "ray[default]"

elif [[ "$FRAMEWORK" == "dask" ]]; then
  echo "⬇️  Installing Dask and Dask-ML..."
  pip install "bokeh>=3.1.0"
  pip install "dask[distributed]" dask-ml
  pip install "pyarrow>=10.0.1"

elif [[ "$FRAMEWORK" == "spark" ]]; then
  echo "⬇️  Ensuring Spark ${SPARK_VERSION} is installed..."
  if [[ ! -d "$SPARK_DIR" ]]; then
    cd "$HOME"
    wget "https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/${SPARK_PACKAGE}.tgz"
    tar zxvf "${SPARK_PACKAGE}.tgz"
    rm "${SPARK_PACKAGE}.tgz"
  fi

  pip install "pyspark==3.5.1"
  pip install psutil

  # Make Spark available in normal shells too
  {
    echo "export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64"
    echo "export SPARK_HOME=$SPARK_DIR"
    echo 'export PATH="$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin"'
  } >> "$HOME/.bashrc"
fi


# --- Start processes in tmux ---
echo "🚀 Starting $FRAMEWORK $ROLE node (in tmux)..."

if [[ "$FRAMEWORK" == "ray" ]]; then

  if [[ "$ROLE" == "head" ]]; then
    tmux new-session -d -s ray-head "
      source '$ENV_DIR/bin/activate' && \
      ray start --head --port=6379 --dashboard-host=0.0.0.0 --block --include-dashboard=false
    "
    echo "🌐 Ray head started on port 6379 (dashboard usually on 8265)."

  elif [[ "$ROLE" == "worker" ]]; then
    tmux new-session -d -s ray-worker "
      source '$ENV_DIR/bin/activate' && \
      ray start --address='$HEAD_IP:6379' --block
    "
    echo "🔗 Ray worker connected to head at $HEAD_IP:6379."
  fi

elif [[ "$FRAMEWORK" == "dask" ]]; then

  export DASK_PYARROW_STRINGS=0

  if [[ "$ROLE" == "head" ]]; then
    tmux new-session -d -s dask-head "
      source '$ENV_DIR/bin/activate' && \
      export DASK_PYARROW_STRINGS=0 && \
      dask-scheduler --host 0.0.0.0
    "
    echo "🌐 Dask scheduler started (port 8786, dashboard on 8787)."

  elif [[ "$ROLE" == "worker" ]]; then
    tmux new-session -d -s dask-worker "
      source '$ENV_DIR/bin/activate' && \
      export DASK_PYARROW_STRINGS=0 && \
      dask-worker tcp://$HEAD_IP:8786
    "
    echo "🔗 Dask worker connected to scheduler at tcp://$HEAD_IP:8786."
  fi

elif [[ "$FRAMEWORK" == "spark" ]]; then

  if [[ "$ROLE" == "head" ]]; then
    tmux new-session -d -s spark-head "
      export JAVA_HOME=\"/usr/lib/jvm/java-17-openjdk-amd64\" && \
      export SPARK_HOME=\"$SPARK_DIR\" && \
      export PATH=\"\$PATH:\$SPARK_HOME/bin:\$SPARK_HOME/sbin\" && \
      cd \"$SPARK_DIR\" && \
      start-master.sh --host 0.0.0.0 &&\
      sleep infinity
    "
    echo "🌐 Spark master started (default UI on port 8080, master port 7077)."

  elif [[ "$ROLE" == "worker" ]]; then
    tmux new-session -d -s spark-worker "
      export JAVA_HOME=\"/usr/lib/jvm/java-17-openjdk-amd64\" && \
      export SPARK_HOME=\"$SPARK_DIR\" && \
      export PATH=\"\$PATH:\$SPARK_HOME/bin:\$SPARK_HOME/sbin\" && \
      cd \"$SPARK_DIR\" && \
      start-worker.sh spark://$HEAD_IP:7077 &&\
      sleep infinity
    "
    echo "🔗 Spark worker connected to master at spark://$HEAD_IP:7077."
  fi
fi

echo "✅ $FRAMEWORK $ROLE node setup complete (running in tmux)."
echo "   Use 'tmux ls' to see sessions, 'tmux attach -t <name>' to inspect."
