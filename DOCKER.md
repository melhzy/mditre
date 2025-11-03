# MDITRE Docker Guide

This directory contains Docker configurations for reproducible MDITRE environments. Docker resolves version conflicts by providing consistent, isolated environments.

## 🐳 Quick Start

### Prerequisites
- Docker Engine 20.10+ 
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support)

### Option 1: Python-Only Environment

```bash
# Build and start Python environment
docker-compose up -d mditre-python

# Enter container
docker exec -it mditre-python bash

# Run tests
cd /workspace/Python
pytest tests/test_all.py -v
```

### Option 2: Full Environment (Python + R)

```bash
# Build and start full environment
docker-compose up -d mditre-full

# Enter container
docker exec -it mditre-full bash

# Run Python tests
cd /workspace/Python
pytest tests/test_all.py -v

# Run R tests
cd /workspace/R
Rscript run_comprehensive_tests.R
```

### Option 3: Jupyter Lab Environment

```bash
# Start Jupyter Lab
docker-compose up -d mditre-jupyter

# Access at http://localhost:8888
# No token required (development mode)
```

## 🏗️ Building Images

### Build Python-only image
```bash
docker build --target base -t mditre:python .
```

### Build full image with R
```bash
docker build --target with-r -t mditre:full .
```

### Build with custom tags
```bash
docker build --target base -t mditre:python-1.0.1 .
docker build --target with-r -t mditre:full-1.0.1 .
```

## 📦 Image Details

### Base Image (`mditre:python`)
- **Base**: nvidia/cuda:12.4.0-runtime-ubuntu24.04
- **Python**: 3.12.3
- **PyTorch**: 2.5.1 with CUDA 12.4
- **Size**: ~5GB
- **Use case**: Python-only development and testing

### Full Image (`mditre:full`)
- **Base**: mditre:python
- **R**: 4.5.2
- **Additional**: R packages (phyloseq, ggtree, reticulate, etc.)
- **Size**: ~7GB
- **Use case**: Full R+Python development

## 🔧 Common Operations

### Run Python tests
```bash
docker-compose run --rm mditre-python pytest Python/tests/test_all.py -v
```

### Run cross-platform verification
```bash
docker-compose run --rm mditre-python python scripts/verify_cross_platform.py
```

### Interactive Python session
```bash
docker-compose run --rm mditre-python python
```

### Interactive R session
```bash
docker-compose run --rm mditre-full R
```

### Run specific Python script
```bash
docker-compose run --rm mditre-python python Python/mditre/examples/data_loader_example.py
```

### Mount custom data directory
```bash
docker-compose run --rm -v /path/to/data:/data mditre-python bash
```

## 🖥️ GPU Support

GPU support is enabled by default via NVIDIA Docker runtime.

### Check GPU availability in container
```bash
docker-compose run --rm mditre-python python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### Specify which GPUs to use
```bash
# Use GPU 0 only
CUDA_VISIBLE_DEVICES=0 docker-compose up mditre-python

# Use GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 docker-compose up mditre-python
```

### CPU-only mode
```bash
CUDA_VISIBLE_DEVICES=-1 docker-compose up mditre-python
```

## 📁 Volume Mounts

### Default volumes
- `.:/workspace` - Source code (live reload)
- `mditre-data:/data` - Persistent data storage

### Access data volume
```bash
docker volume inspect mditre_mditre-data
```

### Backup data volume
```bash
docker run --rm -v mditre_mditre-data:/data -v $(pwd):/backup ubuntu tar czf /backup/mditre-data-backup.tar.gz /data
```

## 🔄 Development Workflow

### 1. Start development environment
```bash
docker-compose up -d mditre-full
docker exec -it mditre-full bash
```

### 2. Make code changes (on host machine)
Changes are immediately reflected in the container due to volume mount.

### 3. Run tests in container
```bash
cd /workspace/Python
pytest tests/test_all.py -v
```

### 4. Rebuild image after dependency changes
```bash
docker-compose build mditre-full
```

## 🧹 Cleanup

### Stop all containers
```bash
docker-compose down
```

### Remove images
```bash
docker rmi mditre:python mditre:full mditre:jupyter
```

### Remove volumes (⚠️ deletes data)
```bash
docker-compose down -v
```

### Full cleanup
```bash
docker-compose down -v --rmi all
```

## 🐛 Troubleshooting

### Issue: "docker: command not found"
**Solution**: Install Docker from https://docs.docker.com/get-docker/

### Issue: "permission denied while trying to connect to the Docker daemon"
**Solution**: 
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Issue: "nvidia-container-cli: initialization error"
**Solution**: Install NVIDIA Container Toolkit:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Issue: "Cannot connect to Jupyter Lab"
**Solution**: Check port mapping:
```bash
docker-compose ps
# Ensure 8888:8888 is mapped
# Try accessing http://127.0.0.1:8888
```

### Issue: Container runs out of memory
**Solution**: Increase Docker memory limit in Docker Desktop settings or Docker daemon config.

### Issue: R packages fail to install
**Solution**: Check R installation logs:
```bash
docker-compose run --rm mditre-full Rscript R/install_dependencies.R
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [MDITRE GitHub Repository](https://github.com/melhzy/mditre)

## 🔐 Security Notes

⚠️ **Development Configuration**: The Jupyter Lab instance is configured without authentication for development convenience. **Do not expose to public networks.**

For production use:
```bash
# Set password
docker-compose run --rm mditre-jupyter jupyter lab password

# Update docker-compose.yml to remove --NotebookApp.token='' and --NotebookApp.password=''
```

## 📝 Version Information

- **Docker Image Version**: 1.0.1
- **Python**: 3.12.3
- **R**: 4.5.2
- **PyTorch**: 2.5.1
- **CUDA**: 12.4.0
- **Ubuntu**: 24.04 LTS

Last updated: November 3, 2025
