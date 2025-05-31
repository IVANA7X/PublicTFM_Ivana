# Cell Image-Text Matching with CLIP Fine-Tuning

> **Note**: This is a public version of the repository. The `data` folder containing microscopy images and dataset files has been excluded due to privacy concerns. The code and model implementations are fully available for review and adaptation.

## Project Overview

This repository contains the implementation of a fine-tuned CLIP (Contrastive Language-Image Pretraining) model for matching cell microscopy images with textual descriptions. The project focuses on improving the semantic understanding between cellular images and their corresponding textual descriptions through various fine-tuning approaches.

## Dataset

The dataset consists of microscopy images of different cell types paired with textual descriptions of their morphological features. The data is organized as follows:

- **Images**: Located in `data/dataset_finetuning/`, containing microscopy images of various cell types (e.g., lymphocytes, monocytes, blasts, etc.)
- **CSV Files**: 
  - `train_split.csv`: Training dataset with image paths, captions, and similarity scores
  - `val_split.csv`: Validation dataset
  - `test_split.csv`: Test dataset for final evaluation

Each data entry contains:
- Image path
- Textual caption describing cellular features
- Similarity score (1 for positive matches, 0 for non-matches)

## Model Variants

Several model variants have been implemented and evaluated:

1. **MNRL-ViT-B16**: CLIP with ViT-B/16 backbone fine-tuned using Multiple Negatives Ranking Loss
2. **MNRL-ViT-B32**: CLIP with ViT-B/32 backbone fine-tuned using Multiple Negatives Ranking Loss
3. **MNRL-ViT-L-14**: CLIP with ViT-L/14 backbone fine-tuned using Multiple Negatives Ranking Loss
4. **Contrastive Loss Models**: Fine-tuned using contrastive loss functions
5. **Cosine Similarity Loss Models**: Fine-tuned using cosine similarity loss

## Implementation Details

### Fine-Tuning Approaches

The repository includes several fine-tuning approaches:

1. **Multiple Negatives Ranking Loss (MNRL)**: Improves the model by considering multiple negative examples in a batch
2. **Contrastive Loss**: Traditional approach for learning discriminative features
3. **Cosine Similarity Loss**: Optimizes the cosine similarity between matching pairs

### Evaluation Metrics

Models are evaluated using:
- Embedding similarity scores
- Recall metrics (both specific and non-specific)
- Information retrieval performance

## Files and Directories

- `data/`: Contains dataset files and splits
  - `dataset_finetuning/`: Directory with cell microscopy images
  - `train_split.csv`, `val_split.csv`, `test_split.csv`: Data splits
- `other/`: Contains Jupyter notebooks and Python scripts
  - `Dataset_prep.ipynb`: Notebook for dataset preparation
  - `Fine_tuning2.ipynb`: Notebook for model fine-tuning
  - `fine_tuning_sentence_transformer.py`: Script for fine-tuning using Sentence Transformers
- Model implementation files:
  - `MNRL-VIT-B16`: Implementation of CLIP ViT-B/16 with MNRL
  - `MNRL-ViT-B32`: Implementation of CLIP ViT-B/32 with MNRL
  - `MNRL-ViT-L-14`: Implementation of CLIP ViT-L/14 with MNRL
  - `Recall_especifico_ContrastiveLoss`: Specific recall evaluation with contrastive loss
  - `Recall_especifico_CosineSimLoss`: Specific recall evaluation with cosine similarity loss
  - `Recall_especifico_MNRL`: Specific recall evaluation with MNRL
  - `Recall_no_especifico_CosineSimLoss`: Non-specific recall evaluation with cosine similarity loss

## Requirements

The project requires the following main dependencies:
- Python 3.7+
- PyTorch
- Sentence Transformers
- Hugging Face Transformers
- Pandas
- PIL (Pillow)
- tqdm
- Datasets (Hugging Face)
- Wandb (for experiment tracking)

## Usage

### Data Preparation

1. Ensure your data is organized with the correct directory structure
2. Use `Dataset_prep.ipynb` to prepare your dataset if needed

### Fine-Tuning

For fine-tuning using the Sentence Transformers approach:

```python
# Example command to run the fine-tuning script
python other/fine_tuning_sentence_transformer.py
```

For notebook-based fine-tuning, open and run:
- `Fine_tuning2.ipynb`

### Model Evaluation

The repository includes various evaluation scripts to assess model performance:
- Specific recall evaluation
- Non-specific recall evaluation
- Embedding similarity evaluation

## Project Structure

```
TFM_Ivana/
├── data/
│   ├── dataset_finetuning/  # Cell microscopy images
│   ├── dataset_finetuning.zip
│   ├── test_split.csv
│   ├── train_split.csv
│   └── val_split.csv
├── other/
│   ├── Dataset_prep.ipynb
│   ├── Fine_tuning2.ipynb
│   └── fine_tuning_sentence_transformer.py
├── MNRL-VIT-B16  # Model implementation files
├── MNRL-ViT-B32
├── MNRL-ViT-L-14
├── Recall_especifico_ContrastiveLoss
├── Recall_especifico_CosineSimLoss
├── Recall_especifico_MNRL
└── Recall_no_especifico_CosineSimLoss
```

## Future Work

Potential areas for improvement and extension:
- Experimenting with additional CLIP backbones
- Implementing more sophisticated loss functions
- Expanding the dataset with more cell types and features
- Exploring zero-shot and few-shot learning capabilities
- Developing a web interface for interactive image-text matching

## License

This project is part of a Master's Thesis (TFM) and may be subject to specific licensing requirements.
