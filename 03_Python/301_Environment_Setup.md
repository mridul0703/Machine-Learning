# 301. Python for Machine Learning - Environment Setup

Python is the primary programming language for machine learning. Setting up a proper environment ensures smooth workflow, reproducibility, and scalability.

---

## 🔹 1. Why Python for ML?

- Extensive libraries for ML/DL: `NumPy`, `Pandas`, `Scikit-learn`, `TensorFlow`, `PyTorch`
- Easy to learn and widely used in industry and academia
- Active community and support
- Cross-platform support (Windows, macOS, Linux)

---

## 🔹 2. Anaconda

Anaconda is a Python distribution that simplifies package management and environment setup.

### 2.1 Installation
1. Download Anaconda from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
2. Choose Python 3.x version suitable for your OS
3. Follow installation instructions for Windows/macOS/Linux

### 2.2 Creating a Virtual Environment
```bash
# Create environment named 'ml_env' with Python 3.10
conda create -n ml_env python=3.10

# Activate environment
conda activate ml_env

# Deactivate environment
conda deactivate
```

### 2.3 Installing Packages

```bash
# Install numpy, pandas, matplotlib, scikit-learn, jupyter
conda install numpy pandas matplotlib scikit-learn jupyter

# Install tensorflow and pytorch (optional)
conda install tensorflow
conda install pytorch torchvision torchaudio -c pytorch
```

## 🔹 3. Jupyter Notebook

Jupyter Notebook is an interactive web-based tool for coding, visualization, and documentation.

### 3.1 Installation

```bash
# If using Anaconda, Jupyter is pre-installed
# Otherwise, install via pip
pip install notebook
```
### 3.2 Launching Notebook

```bash
# Start Jupyter Notebook server
jupyter notebook
Opens browser interface at http://localhost:8888

Create new Python notebooks and run code interactively
```

### 3.3 Useful Extensions

- nbextensions for extra features: pip install jupyter_contrib_nbextensions
- Enable table of contents, code folding, spell check, etc.

## 🔹 4. VS Code
VS Code is a lightweight, powerful code editor with rich Python support.

### 4.1 Installation

- Download VS Code: https://code.visualstudio.com/
- Install Python extension from Microsoft Marketplace

### 4.2 Configuring Python Environment

- Open Command Palette (Ctrl+Shift+P) → Python: Select Interpreter → choose your conda env (ml_env)
- Install recommended extensions: Pylance, Jupyter

### 4.3 Running Notebooks in VS Code

- Open .ipynb file → VS Code provides interactive interface similar to Jupyter
- Run cells, visualize outputs, and integrate version control

## 🔹 5. Best Practices

- Always use virtual environments for each project
- Keep packages updated but maintain version compatibility
- Use Git for version control
- Document experiments and code in notebooks

## ✅ Summary

- Set up Python environment using Anaconda or VS Code
- Use Jupyter or VS Code notebooks for interactive ML workflow
- Maintain virtual environments for reproducibility
- Install essential ML libraries: NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow, PyTorch
