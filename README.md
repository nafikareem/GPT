# **GPT Architecture**


![alt text](img/demo.gif)


This Repository  inspired by the [*LLMs-from-scratch*](https://github.com/rasbt/LLMs-from-scratch) repository, designed to understanding and implement a GPT-2-like large language model (LLM) from the ground up. This repository provides a complete implementation of a GPT-2 model (355M parameters) with a conversational chatbot interface built using Streamlit. It includes scripts and Jupyter notebooks for model training, fine-tuning, and data processing, making it an educational resource for learning about LLMs.

The goal of this project is to provide a hands-on, code-centric approach to building and interacting with a GPT-2 model, from architecture implementation to deployment as a chatbot.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Chatbot](#running-the-chatbot)
  - [Training the Model](#training-the-model)
  - [Fine-Tuning the Model](#fine-tuning-the-model)
  - [Testing and Notebooks](#testing-and-notebooks)
- [Datasets](#datasets)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
This project implements a GPT-2 model from scratch, including the transformer architecture, attention mechanism, finetuning for classification and instruction and training pipeline. The model is fine-tuned for text generation and integrated into a Streamlit-based chatbot interface (`app.py`) that allows users to interact with the model via a conversational UI. The repository includes:

- A custom GPT-2 achitecture implementation (`src/gpt.py`) with multi-head attention (`src/utils/attention_mechanism.py`).
- Scripts for data loading, training, and fine-tuning (`src/utils/`, `src/training/`, `src/finetuning/`).
- Datasets for raw text, classification, and instruction-based tasks (`data/`).
- Jupyter notebooks for testing (`notebooks/`).
- A pre-trained model checkpoint (`models/gpt2-medium355M-sft-standalone.pth`, not included in the repo).

This project is heavily inspired by the [*LLMs-from-scratch*](https://github.com/rasbt/LLMs-from-scratch) repository by Sebastian Raschka, adapted to include a user-friendly chatbot interface and custom datasets.

## Project Structure
```
GPT/
├── app.py                           
├── data/
│   ├── raw/
│   │   └── pride_and_prejudice.txt  
│   ├── classification/              
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── validation.csv           
│   └── instruction/                 
│       ├── instruction-data-with-response-standalone.json
│       └── instruction-data.json    
├── src/
│   ├── gpt.py                       
│   ├── training/
│   │   ├── load_weights.ipynb
│   │   └── train.ipynb              
│   ├── finetuning/
│   │   ├── instruction.py
│   │   └── classification.py        
│   └── utils/          
│       ├── attention_mechanism.py
│       ├── data_loader.py
│       ├── dataset_classification.py
│       ├── gpt_download.py
│       ├── train.py                
├── models/
│   ├── classification_model.pth
│   ├── classification_train_metrics.json
│   ├── gpt2-medium355M-sft-standalone.pth  
│   ├── model_and_optimizer.pth
│   └── model.pth                   
├── notebooks/
│   ├── test_instruction.ipynb      
│   └── test_train.ipynb            
└── README.md                       
```

## Installation
To set up the project, follow these steps. We recommend using a virtual environment or Conda for dependency management.

### Prerequisites
- Python 3.8 or higher
- A CUDA-compatible GPU (optional but recommended for faster training/inference)
- Git (for cloning the repository)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/GPT.git
   cd GPT
   ```

2. **Set Up a Conda Environment** (recommended):
   ```bash
   conda create -n GPT python=3.8
   conda activate GPT
   ```

   Alternatively, use a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Create a `requirements.txt` file with the following content:
   ```
   streamlit>=1.10.0
   torch>=1.10.0
   tiktoken
   pandas
   numpy
   jupyter
   ```
   Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   For GPU support, install the appropriate PyTorch version:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Obtain the Model Checkpoint**:
   - The `gpt2-medium355M-sft-standalone.pth` file is required for the chatbot and is not included in the repository due to its size.
   - Download the model (if available) and place it in `models/`.
   - Alternatively, check `src/utils/gpt_download.py` for downloading instructions or train your own model using `src/training/train.ipynb`.

5. **Verify Setup**:
   - Confirm datasets are in `data/` and the model checkpoint is in `models/`.

## Usage
This repository offers multiple ways to interact with the GPT-2 model, from running the chatbot to training and fine-tuning.

### Running the Chatbot
The primary interface is a Streamlit-based chatbot (`app.py`) that lets you interact with the fine-tuned GPT-2 model.

1. Start the Streamlit app:
   ```bash
   cd GPT
   streamlit run app.py
   ```

2. Open your browser at `http://localhost:8501`.
3. Type a prompt in the chat input field and press Enter to receive a response from the model.

**Example**:
- **Prompt**: "What is the plural form of 'mouse'?"
- **Response**: (A 35-token response generated by the model, displayed in the chat interface)

### Training the Model
To train the GPT-2 model from scratch or continue training:

1. Open `src/training/train.ipynb` in Jupyter Notebook:
   ```bash
   jupyter notebook src/training/train.ipynb
   ```
2. Follow the notebook’s instructions to load data (e.g., `data/raw/pride_and_prejudice.txt`), configure hyperparameters, and train the model.
3. Save the trained checkpoint to `models/`.

Alternatively, use `src/utils/train.py` for command-line training (check the script for usage).

### Fine-Tuning the Model
To fine-tune the model on specific tasks:

1. For instruction-based fine-tuning, use `src/finetuning/instruction.py`:
   ```bash
   python src/finetuning/instruction.py
   ```
   - Requires `data/instruction/instruction-data.json` or `instruction-data-with-response-standalone.json`.
2. For classification tasks, use `src/finetuning/classification.py`:
   ```bash
   python src/finetuning/classification.py
   ```
   - Uses `data/classification/train.csv`, `test.csv`, and `validation.csv`.

Check the scripts for configuration details and adjust hyperparameters as needed.

### Testing and Notebooks
Explore the model and data using the provided notebooks:

- `notebooks/test_instruction.ipynb`: Test instruction-based generation.
- `notebooks/test_train.ipynb`: Experiment with training configurations.

Run the notebooks:
```bash
jupyter notebook notebooks/
```

## Datasets
The `data/` directory contains datasets for various tasks:

- **Raw Text** (`data/raw/pride_and_prejudice.txt`): Used for pre-training the model on a large text corpus (e.g., [*Pride and Prejudice*](https://www.gutenberg.org/ebooks/1342)).
- **Classification** (`data/classification/`): CSV files (`train.csv`, `test.csv`, `validation.csv`) for classification tasks, such as sentiment analysis or text categorization.
- **Instruction** (`data/instruction/`): JSON files (`instruction-data.json`, `instruction-data-with-response-standalone.json`) for instruction-based fine-tuning, containing prompts and responses.

You can add your own datasets to these directories, following the same format.

## FAQ
**Q: Why is the model checkpoint not included in the repository?**  
A: The `gpt2-medium355M-sft-standalone.pth` file is large and not included to keep the repository lightweight. Download it separately or train your own model using `src/training/train.ipynb`.

**Q: I get a `ModuleNotFoundError` when running `app.py`.**  
A:  Verify that `src/gpt.py` and `src/utils/train.py` use relative imports (e.g., `from .utils.attention_mechanism import MultiHeadAttention`). Run the app from the `GPT/` directory.

**Q: How can I improve response time?**  
A: Use a CUDA-compatible GPU for faster inference. Alternatively, reduce `max_new_tokens` in `app.py` (default: 35) for quicker responses.

**Q: Can I use my own dataset?**  
A: Yes! Place raw text files in `data/raw/`, CSVs in `data/classification/`, or JSONs in `data/instruction/`, and update the training or fine-tuning scripts accordingly.

## Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

