# TD-FUN

![image (1)](https://github.com/user-attachments/assets/046b208c-dee6-4384-ad47-2ce8794ebde1)
![image](https://github.com/user-attachments/assets/ad38fb20-b6f6-45f2-8063-ff8b6354961d)

# Uncertainty Analysis Tool

A Streamlit-based web application for uncertainty sentence extraction in text using BERT models.

## Environment Setup

### Option 1: Using Conda (Recommended)

1. Create a new conda environment:
```bash
# Create environment
conda create -n uncertainty python=3.9

# Activate environment
conda activate uncertainty
```

2. Install dependencies:
```bash
# Install PyTorch and related packages
conda install -c pytorch pytorch torchvision torchaudio

# Install other dependencies
pip install streamlit transformers datasets pandas numpy scikit-learn plotly
pip install fugashi ipadic
```

### Option 2: Using venv

1. Create a new virtual environment:
```bash
# Create environment
python -m venv venv

# Activate environment (MacOS/Linux)
source venv/bin/activate

# Activate environment (Windows)
.\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

The main dependencies are listed in `requirements.txt`:
```
conda install -c conda-forge streamlit
pip install torch torchvision torchaudio
pip install transformers datasets pandas numpy scikit-learn plotly
pip install fugashi ipadic
```

## Running the Application

1. Ensure your virtual environment is activated:
```bash
# If using conda
conda activate uncertainty

# If using venv
source venv/bin/activate  # MacOS/Linux
# or
.\venv\Scripts\activate  # Windows
```

2. Run the Streamlit app:
```bash
streamlit run uncertainty.py
```

## Managing the Virtual Environment

### Using Conda

```bash
# Create new environment
conda create -n uncertainty python=3.9

# Activate environment
conda activate uncertainty

# Deactivate environment
conda deactivate

# Delete environment
conda remove --name uncertainty --all

# Export environment
conda env export > environment.yml

# Create environment from file
conda env create -f environment.yml
```

### Using venv

```bash
# Create new environment
python -m venv venv

# Activate environment
source venv/bin/activate  # MacOS/Linux
.\venv\Scripts\activate   # Windows

# Deactivate environment
deactivate

# Delete environment
rm -rf venv  # MacOS/Linux
rmdir /s /q venv  # Windows

# Export requirements
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

1. CUDA/GPU Issues:
```bash
# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

2. Memory Issues:
- Ensure you have enough RAM and VRAM
- Consider using a smaller batch size

3. Module Not Found Errors:
- Verify your virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

4. Streamlit Port Issues:
```bash
# Run on a different port
streamlit run uncertainty.py --server.port 8502
```

### For Apple Silicon (M1/M2) Users

Special installation steps for Apple Silicon Macs:
```bash
# Create conda environment with specific platform
CONDA_SUBDIR=osx-arm64 conda create -n uncertainty python=3.9

# Activate and install dependencies
conda activate uncertainty
conda config --env --set subdir osx-arm64
```

## Development

1. Format code before committing:
```bash
# Install formatter
pip install black

# Format code
black .
```

2. Run tests:
```bash
# Install test dependencies
pip install pytest

# Run tests
pytest
```

## License

MIT License
