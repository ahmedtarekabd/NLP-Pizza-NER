# Setup

## Create a Virtual Enviroment

```shell
    # First time only
    python -m venv env
    # Activate venv
    .\env\Scripts\activate
```

## Installing Dependencies

### CUDA support (~ 3GB!)

```shell
    # Activate venv
    .\env\Scripts\activate

    # Install dependencies
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    # pip install numpy pandas gensim scipy scikit-learn sklearn_crfsuite nltk matplotlib seaborn tqdm transformers datasets
    pip install -r requirements.txt
```

### CPU Only

Install torch without `--index-url` flag

```shell
    pip3 install torch torchvision torchaudio
```

## Create `data/` directory

1. Create `data/` directory in the root of the project.
2. Download the following zip files
   1. Dataset: https://drive.google.com/file/d/1HAoroMJstkCyyQ4k6s0I2pvmM0MjYYQC/view
   2. Sample (Optional): https://drive.google.com/file/d/1N7YuUSWrDjBUVQdIk0Zm00xZIdul4gZz/view
3. Extract both zip files inside `data/`

---
