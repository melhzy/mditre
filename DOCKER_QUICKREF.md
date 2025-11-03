# MDITRE Docker Quick Reference

## 🚀 Quick Start (3 Commands)

```bash
git clone https://github.com/melhzy/mditre.git && cd mditre
docker-compose up -d mditre-python
docker exec -it mditre-python bash
```

## 📦 Docker Images

| Image | Base | Size | Python | R | Use Case |
|-------|------|------|--------|---|----------|
| `mditre:python` | CUDA 12.4 + Ubuntu 24.04 | ~5GB | 3.12.3 | ❌ | Python development/testing |
| `mditre:full` | mditre:python | ~7GB | 3.12.3 | 4.5.2 | Full R+Python development |
| `mditre:jupyter` | mditre:full | ~7GB | 3.12.3 | 4.5.2 | Interactive analysis |

## 🔧 Common Commands

### Build
```bash
docker-compose build mditre-python    # Build Python-only
docker-compose build mditre-full      # Build with R
docker-compose build                  # Build all
```

### Run
```bash
# Interactive shell
docker-compose run --rm mditre-python bash
docker-compose run --rm mditre-full bash

# Start Jupyter (http://localhost:8888)
docker-compose up -d mditre-jupyter

# Run tests
docker-compose run --rm mditre-python pytest Python/tests/test_all.py -v
```

### Manage
```bash
docker-compose ps              # Show running containers
docker-compose down            # Stop all
docker-compose down -v         # Stop and remove volumes
docker-compose logs -f         # View logs
```

## 🎯 Quick Tasks

### Run Python Tests
```bash
docker-compose run --rm mditre-python pytest Python/tests/test_all.py -v
```

### Check GPU
```bash
docker-compose run --rm mditre-python python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Interactive Python
```bash
docker-compose run --rm mditre-python python
```

### Interactive R
```bash
docker-compose run --rm mditre-full R
```

### Run Script
```bash
docker-compose run --rm mditre-python python your_script.py
```

### Mount Data
```bash
docker-compose run --rm -v /path/to/data:/data mditre-python bash
```

## 🐛 Troubleshooting

### Docker not found
```bash
# Install Docker: https://docs.docker.com/get-docker/
```

### GPU not available
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Port 8888 already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8889:8888"  # Use port 8889 instead
```

### Permission denied
```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 📖 Full Documentation

See [DOCKER.md](DOCKER.md) for complete documentation including:
- Advanced configuration
- Volume management
- Network settings
- Security considerations
- Performance optimization

## 🔗 Links

- **Repository**: https://github.com/melhzy/mditre
- **Documentation**: [DOCKER.md](DOCKER.md)
- **Issues**: https://github.com/melhzy/mditre/issues
