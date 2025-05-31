#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para realizar fine-tuning de un modelo CLIP con Sentence Transformers
para un dataset de pares de imágenes y texto.
"""

import os
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from datasets import DatasetDict, Dataset
from sentence_transformers import SentenceTransformer, models, losses, InputExample, evaluation
from torch.utils.data import DataLoader

# Configuración
BATCH_SIZE = 16
NUM_EPOCHS = 4
MODEL_NAME = "clip-ViT-B-32"  # Modelo CLIP de Sentence Transformers
OUTPUT_PATH = "./sentence-transformer-finetuned-model"

def replace_path(df, column_name='image_path'):
    """
    Reemplaza las rutas '/content/drive/MyDrive/TFM' por 'data' en la columna especificada.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        El dataframe que contiene la columna a modificar
    column_name : str, opcional
        El nombre de la columna que contiene las rutas (por defecto 'image_path')
    
    Retorna:
    --------
    pandas.DataFrame
        El dataframe con las rutas modificadas
    """
    # Crear una copia para no modificar el original
    df_copy = df.copy()
    
    # Verificar que la columna existe
    if column_name in df_copy.columns:
        # Reemplazar la ruta
        df_copy[column_name] = df_copy[column_name].str.replace('/content/drive/MyDrive/TFM', 'data')
    else:
        print(f"La columna '{column_name}' no existe en el dataframe")
    
    return df_copy

def load_image(path):
    """
    Carga una imagen desde la ruta especificada.
    
    Parámetros:
    -----------
    path : str
        Ruta a la imagen
        
    Retorna:
    --------
    PIL.Image
        La imagen cargada
    """
    try:
        return Image.open(path).convert('RGB')
    except Exception as e:
        print(f"Error al cargar la imagen {path}: {e}")
        # Devolver una imagen negra como fallback
        return Image.new('RGB', (224, 224), color=0)

def load_datasets(train_csv, val_csv, test_csv):
    """
    Carga los datasets desde archivos CSV.
    
    Parámetros:
    -----------
    train_csv : str
        Ruta al archivo CSV de entrenamiento
    val_csv : str
        Ruta al archivo CSV de validación
    test_csv : str
        Ruta al archivo CSV de prueba
        
    Retorna:
    --------
    DatasetDict
        Un diccionario con los datasets de entrenamiento, validación y prueba
    """
    print("Cargando datasets...")
    
    # Cargar los dataframes
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)
    
    # Reemplazar las rutas
    df_train = replace_path(df_train)
    df_val = replace_path(df_val)
    df_test = replace_path(df_test)
    
    # Convertir a datasets de Hugging Face
    train_ds = Dataset.from_pandas(df_train)
    val_ds = Dataset.from_pandas(df_val)
    test_ds = Dataset.from_pandas(df_test)
    
    # Crear el diccionario de datasets
    dataset_dict = DatasetDict({
        'train': train_ds,
        'valid': val_ds,
        'test': test_ds
    })
    
    print(f"Dataset de entrenamiento: {len(dataset_dict['train'])} ejemplos")
    print(f"Dataset de validación: {len(dataset_dict['valid'])} ejemplos")
    print(f"Dataset de prueba: {len(dataset_dict['test'])} ejemplos")
    
    return dataset_dict

def create_training_examples(dataset):
    """
    Crea ejemplos para el entrenamiento a partir de un dataset.
    """
    examples = []
    for item in tqdm(dataset, desc="Creando ejemplos"):
        image_path = item['image_path']
        caption = item['caption']
        similarity = float(item['similarity'])
        
        # Verificar que la imagen existe
        if not os.path.exists(image_path):
            print(f"Advertencia: La imagen {image_path} no existe. Saltando ejemplo.")
            continue
            
        # En esta versión de Sentence Transformers, InputExample solo acepta texts y label
        # Para manejar imágenes, pasamos tanto el texto como la ruta de la imagen como elementos de texts
        examples.append(InputExample(texts=[caption, image_path], label=similarity))
    
    return examples

def train_model(model, train_examples, val_examples):
    """
    Entrena el modelo CLIP con Sentence Transformers.
    
    Parámetros:
    -----------
    model : SentenceTransformer
        Modelo a entrenar
    train_examples : list
        Lista de ejemplos de entrenamiento
    val_examples : list
        Lista de ejemplos de validación
        
    Retorna:
    --------
    SentenceTransformer
        Modelo entrenado
    """
    print("Configurando entrenamiento...")
    
    # Definir la función de pérdida para CLIP
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Crear evaluador
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1=[example.texts[0] for example in val_examples],  # Texto
        sentences2=[example.texts[1] for example in val_examples],  # Ruta de la imagen
        scores=[example.label for example in val_examples],
        name='val'
    )
    
    # Parámetros de entrenamiento
    warmup_steps = int(len(train_examples) * NUM_EPOCHS * 0.1)  # 10% de los pasos totales
    
    # Crear el dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    
    print(f"Iniciando entrenamiento por {NUM_EPOCHS} épocas...")
    
    # Entrenar el modelo
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_PATH,
        show_progress_bar=True
    )
    
    print(f"Entrenamiento completado. Modelo guardado en {OUTPUT_PATH}")
    
    return model

def evaluate_model(model, test_examples):
    """
    Evalúa el modelo en el conjunto de prueba.
    
    Parámetros:
    -----------
    model : SentenceTransformer
        Modelo a evaluar
    test_examples : list
        Lista de ejemplos de prueba
        
    Retorna:
    --------
    dict
        Diccionario con las puntuaciones de evaluación
    """
    print("Evaluando modelo en el conjunto de prueba...")
    
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1=[example.texts[0] for example in test_examples],  # Texto
        sentences2=[example.texts[1] for example in test_examples],  # Ruta de la imagen
        scores=[example.label for example in test_examples],
        name='test'
    )
    
    test_score = evaluator(model)
    
    # El evaluador puede devolver un diccionario o un valor único, manejamos ambos casos
    if isinstance(test_score, dict):
        for metric_name, metric_value in test_score.items():
            print(f"{metric_name}: {metric_value:.4f}")
    else:
        print(f"Puntuación en el conjunto de prueba: {test_score:.4f}")
    
    return test_score

def calculate_similarity(model, image_path, text):
    """
    Calcula la similitud entre una imagen y un texto.
    
    Parámetros:
    -----------
    model : SentenceTransformer
        Modelo entrenado
    image_path : str
        Ruta a la imagen
    text : str
        Texto
        
    Retorna:
    --------
    float
        Similitud entre la imagen y el texto
    """
    # Para esta versión de Sentence Transformers, codificamos texto e imagen como textos
    # Codificar el texto y la ruta de la imagen
    embeddings = model.encode([text, image_path], convert_to_tensor=True)
    
    # Obtener los embeddings individuales
    text_embedding = embeddings[0].reshape(1, -1)
    image_embedding = embeddings[1].reshape(1, -1)
    
    # Calcular similitud de coseno
    from sentence_transformers import util
    similarity = util.pytorch_cos_sim(text_embedding, image_embedding)[0][0].item()
    
    return similarity

def create_sentence_transformer_model():
    """
    Carga el modelo CLIP de Sentence Transformers.
    
    Retorna:
    --------
    SentenceTransformer
        Modelo CLIP cargado
    """
    print(f"Cargando modelo CLIP: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    return model

def main():
    """
    Función principal.
    """
    # Rutas a los archivos CSV
    train_csv = 'data/train_split.csv'
    val_csv = 'data/val_split.csv'
    test_csv = 'data/test_split.csv'
    
    # Crear directorio de salida si no existe
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Cargar datasets
    dataset_dict = load_datasets(train_csv, val_csv, test_csv)
    
    # Crear ejemplos para el entrenamiento
    print("Preparando datos para el entrenamiento...")
    train_examples = create_training_examples(dataset_dict['train'])
    val_examples = create_training_examples(dataset_dict['valid'])
    test_examples = create_training_examples(dataset_dict['test'])
    
    print(f"Ejemplos de entrenamiento: {len(train_examples)}")
    print(f"Ejemplos de validación: {len(val_examples)}")
    print(f"Ejemplos de prueba: {len(test_examples)}")
    
    # Cargar modelo CLIP de Sentence Transformers
    model = create_sentence_transformer_model()
    
    # Entrenar modelo
    model = train_model(model, train_examples, val_examples)
    
    # Evaluar modelo
    test_score = evaluate_model(model, test_examples)
    
    # Ejemplo de uso
    example_image_path = dataset_dict['test'][0]['image_path']
    example_text = "Una célula con núcleo redondo"
    
    # Calcular similitud
    similarity = calculate_similarity(model, example_image_path, example_text)
    print(f"Similitud entre la imagen y el texto de ejemplo: {similarity:.4f}")
    
    return model

if __name__ == "__main__":
    # Verificar si CUDA está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device}")
    
    # Ejecutar función principal
    model = main()
