#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_ensemble_high_dim.py
Prediction script for an ensemble high-dimensional feature model - output all prediction results
"""

import os
import sys
import math
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def resolve_device(use_gpu=True, gpu_id=0):
    """Resolve the torch device from external command-line settings.

    Parameters:
    use_gpu: whether to use CUDA if available.
    gpu_id: CUDA device index, e.g. 0 for cuda:0.
    """
    if not use_gpu:
        return torch.device("cpu")

    if not torch.cuda.is_available():
        print("Warning: GPU was requested, but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")

    if gpu_id is None:
        gpu_id = 0

    try:
        gpu_id = int(gpu_id)
    except (TypeError, ValueError):
        print(f"Warning: Invalid gpu_id={gpu_id}. Falling back to GPU 0.")
        gpu_id = 0

    cuda_count = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= cuda_count:
        print(f"Warning: gpu_id={gpu_id} is out of range. Available GPU IDs: 0-{cuda_count - 1}. Falling back to GPU 0.")
        gpu_id = 0

    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")



# ==================== ESM Imports =====================
try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import *
    from esm.tokenization import EsmSequenceTokenizer
    ESM_AVAILABLE = True
except ImportError:
    print("Warning: Failed to import the ESM module. Please make sure ESM is installed correctly.")
    ESM_AVAILABLE = False

# ================= Model definitions (same as training) =================
class HighDimTransformer(nn.Module):
    """Transformer module specialized for high-dimensional features."""
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x_proj = self.input_projection(x)
        x_seq = x_proj.unsqueeze(1)
        x_seq = x_seq + self.pos_encoding
        encoded = self.transformer(x_seq)
        pooled = encoded.mean(dim=1)
        features = self.feature_extractor(pooled)
        return features

class HighDimCNN(nn.Module):
    """CNN module specialized for high-dimensional features."""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.2):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.conv_output_size = 256
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x_conv = x.unsqueeze(1)
        conv_features = self.conv_layers(x_conv)
        conv_features = conv_features.view(batch_size, -1)
        features = self.fc_layers(conv_features)
        return features

class HighDimLSTM(nn.Module):
    """LSTM module specialized for high-dimensional features."""
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.sequence_creator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        sequence = self.sequence_creator(x)
        sequence = sequence.view(batch_size, 4, self.hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(sequence)
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_out, dim=1)
        features = self.feature_extractor(context_vector)
        return features

class HighDimEnsembleModel(nn.Module):
    """High-dimensional feature ensemble model combining Transformer, CNN, and LSTM branches."""
    def __init__(self, input_dim=1152, transformer_params=None, cnn_params=None, 
                 lstm_params=None, fusion_dims=[512, 256], dropout=0.2, max_value=0.7):
        super().__init__()
        
        if transformer_params is None:
            transformer_params = {'d_model': 256, 'nhead': 8, 'num_layers': 3}
        if cnn_params is None:
            cnn_params = {'hidden_dims': [512, 256, 128]}
        if lstm_params is None:
            lstm_params = {'hidden_dim': 256, 'num_layers': 2}
        
        self.transformer_branch = HighDimTransformer(input_dim, **transformer_params, dropout=dropout)
        self.cnn_branch = HighDimCNN(input_dim, **cnn_params, dropout=dropout)
        self.lstm_branch = HighDimLSTM(input_dim, **lstm_params, dropout=dropout)
        
        self.branch_weights = nn.Parameter(torch.ones(3))
        
        fusion_layers = []
        input_fusion_dim = 128 * 3
        
        for hidden_dim in fusion_dims:
            fusion_layers.extend([
                nn.Linear(input_fusion_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            input_fusion_dim = hidden_dim
        
        self.fusion_network = nn.Sequential(*fusion_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(input_fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.max_value = max_value
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        
        transformer_features = self.transformer_branch(x)
        cnn_features = self.cnn_branch(x)
        lstm_features = self.lstm_branch(x)
        
        weights = torch.softmax(self.branch_weights, dim=0)
        weighted_transformer = transformer_features * weights[0]
        weighted_cnn = cnn_features * weights[1]
        weighted_lstm = lstm_features * weights[2]
        
        combined_features = torch.cat([weighted_transformer, weighted_cnn, weighted_lstm], dim=1)
        fused_features = self.fusion_network(combined_features)
        output = self.output_layer(fused_features) * self.max_value
        
        branch_info = {
            'transformer': transformer_features,
            'cnn': cnn_features,
            'lstm': lstm_features,
            'weights': weights
        }
        
        return output, branch_info

# ================= Predictor class =================
class EnsembleHighDimPredictor:
    def __init__(self, model_path, scaler_path, target_dim=1152, device=None):
        self.device = device if device is not None else resolve_device(use_gpu=True, gpu_id=0)
        self.target_dim = target_dim
        
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Build the model
        self.model = HighDimEnsembleModel(input_dim=target_dim)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Ensemble high-dimensional model loaded successfully: {model_path}")
        print(f"Target feature dimension: {target_dim}")
        print(f"Device in use: {self.device}")
    
    def prepare_flat_data(self, embeddings_dict):
        """Prepare flattened data and convert all embeddings to the same dimension."""
        sequences = []
        sequence_ids = []
        
        for seq_id, data in embeddings_dict.items():
            if 'embedding' not in data:
                continue
                
            emb = data['embedding']
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            
            # Flatten directly to preserve all information
            emb_flat = np.array(emb).flatten()
            
            # Standardize feature dimensions
            if len(emb_flat) > self.target_dim:
                emb_flat = emb_flat[:self.target_dim]
            elif len(emb_flat) < self.target_dim:
                pad_length = self.target_dim - len(emb_flat)
                emb_flat = np.pad(emb_flat, (0, pad_length), mode='constant')
            
            sequences.append(emb_flat)
            sequence_ids.append(seq_id)
        
        return np.array(sequences, dtype=np.float32), sequence_ids
    
    def predict(self, embeddings_dict, batch_size=16):
        """Run prediction."""
        # Prepare data
        X, sequence_ids = self.prepare_flat_data(embeddings_dict)
        
        if len(X) == 0:
            print("Error: No valid embedding data found.")
            return {}
        
        # Standardize features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_scaled))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = {}
        with torch.no_grad():
            for batch_idx, (batch_x,) in enumerate(dataloader):
                batch_x = batch_x.to(self.device)
                outputs, branch_info = self.model(batch_x)
                
                # Process batch results
                batch_start = batch_idx * batch_size
                batch_end = batch_start + len(batch_x)
                
                for i, seq_idx in enumerate(range(batch_start, batch_end)):
                    if seq_idx < len(sequence_ids):
                        seq_id = sequence_ids[seq_idx]
                        pred = outputs[i].cpu().numpy()[0]
                        weights = branch_info['weights'].cpu().numpy()
                        
                        predictions[seq_id] = {
                            'efficiency': pred,
                            'is_high_efficiency': pred > 0.3,
                            'transformer_weight': weights[0],
                            'cnn_weight': weights[1],
                            'lstm_weight': weights[2]
                        }
        
        return predictions

# ==================== ESM Embedding Generation =====================
def setup_esm_model(esm_weights_path=None, device=None):
    """Initialize the ESM model on the selected device.

    Parameters:
    esm_weights_path: path to the local ESM-C weight file. If not provided,
        the script will look for ./esmc_600m_2024_12_v0.pth.
    device: torch.device selected from command-line settings.
    """
    if not ESM_AVAILABLE:
        print("Error: The ESM module is not available.")
        return None, torch.device("cpu")

    expected_dir_weights = "data/weights/esmc_600m_2024_12_v0.pth"

    if esm_weights_path is None:
        esm_weights_path = "./esmc_600m_2024_12_v0.pth"

    if os.path.exists(esm_weights_path):
        if not os.path.exists(expected_dir_weights):
            print(f"Detected ESM-C weight file: {esm_weights_path}")
            print("Creating a symbolic link under data/weights/...")
            os.makedirs(os.path.dirname(expected_dir_weights), exist_ok=True)
            os.symlink(os.path.abspath(esm_weights_path), expected_dir_weights)
            print("Symbolic link created successfully.")
        else:
            print(f"ESM-C weight file already exists at: {expected_dir_weights}")
    else:
        print(f"Warning: ESM-C weight file was not found at: {esm_weights_path}")
        print("The script will still try ESMC.from_pretrained('esmc_600m').")

    # Use the externally selected device
    if device is None:
        device = resolve_device(use_gpu=True, gpu_id=0)
    print(f"ESM device in use: {device}")

    # Initialize the model
    os.environ["INFRA_PROVIDER"] = "True"
    client = ESMC.from_pretrained("esmc_600m", device=device)
    return client, device

def clean_sequence(sequence):
    """Clean the sequence and keep only valid amino acid characters."""
    sequence = sequence.upper()
    valid_chars = "ACDEFGHIKLMNPQRSTVWYX"
    return ''.join(c if c in valid_chars else 'X' for c in sequence)

def get_esm_embedding(client, device, sequence):
    """Get the ESM embedding for a sequence."""
    if client is None:
        print("ESM client is not initialized.")
        return None
        
    # Clean the sequence
    sequence = clean_sequence(sequence)
    
    try:
        # Encode the sequence
        tokenizer = EsmSequenceTokenizer()
        token_ids = tokenizer.encode(sequence)
        protein_tensor = ESMProteinTensor(sequence=torch.tensor(token_ids).to(device))
        
        # Get embeddings
        logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        esm_embedding = logits_output.embeddings
        return esm_embedding
    except Exception as e:
        print(f"Sequence encoding error: {e}")
        print(f"Problematic sequence: {sequence[:50]}... (length: {len(sequence)})")
        return None

def read_txt_file(filepath):
    """
    Read a TXT file in the format: ID\tsequence
    Return a sequence dictionary
    """
    sequences_dict = {}
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            parts = line.split('\t')
            
            if len(parts) < 2:
                print(f"Warning: Line {line_num} has an invalid format and will be skipped.")
                continue
                
            seq_id = parts[0].strip()
            sequence = parts[1].strip()
            
            sequences_dict[seq_id] = sequence
    
    print(f"Read {len(sequences_dict)} sequences from {filepath}.")
    return sequences_dict

def process_sequences_for_prediction(sequences_dict, output_pkl=None, esm_weights_path=None, device=None):
    """
    Process new sequences and generate embeddings.
    sequences_dict: dictionary in the format {sequence_id: sequence}.
    esm_weights_path: path to the local ESM-C weight file.
    device: torch.device selected from command-line settings.
    """
    client, device = setup_esm_model(esm_weights_path=esm_weights_path, device=device)
    
    if client is None:
        print("Failed to initialize the ESM model.")
        return {}
    
    embeddings_dict = {}
    skipped_sequences = 0
    
    for seq_id, sequence in sequences_dict.items():
        embedding = get_esm_embedding(client, device, sequence)
        
        if embedding is not None:
            embeddings_dict[seq_id] = {
                'sequence': sequence,
                'embedding': embedding.cpu()
            }
            print(f"Processed sequence: {seq_id} (length: {len(sequence)})")
        else:
            skipped_sequences += 1
            print(f"Skipped sequence: {seq_id}")
    
    # Optional: save embeddings
    if output_pkl:
        with open(output_pkl, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print(f"Embeddings saved to: {output_pkl}")
    
    print(f"Successfully processed {len(embeddings_dict)} sequences; skipped {skipped_sequences} sequences.")
    return embeddings_dict

# ==================== Main Prediction Function =====================
def predict_from_txt(
    txt_file_path,
    model_path,
    scaler_path,
    output_csv=None,
    batch_size=16,
    save_embeddings=None,
    target_dim=1152,
    esm_weights_path=None,
    use_gpu=True,
    gpu_id=0
):
    """
    Predict protein efficiency from a TXT file.
    
    Parameters:
    txt_file_path: input TXT file path (format: ID\tsequence)
    model_path: path to the trained model
    scaler_path: path to the scaler file
    output_csv: output CSV file path (optional)
    batch_size: batch size
    save_embeddings: path to save embeddings as a pkl file (optional)
    target_dim: target feature dimension
    esm_weights_path: path to the local ESM-C weight file
    use_gpu: whether to use GPU if CUDA is available
    gpu_id: CUDA GPU ID to use when use_gpu is True
    """
    # Resolve the device once and use it for both ESM embedding and prediction
    device = resolve_device(use_gpu=use_gpu, gpu_id=gpu_id)
    print(f"Selected device: {device}")

    # Initialize predictor
    predictor = EnsembleHighDimPredictor(model_path, scaler_path, target_dim=target_dim, device=device)
    
    # Read the TXT file
    print(f"Reading TXT file: {txt_file_path}")
    sequences_dict = read_txt_file(txt_file_path)
    
    if not sequences_dict:
        print("Error: No sequences were read.")
        return None
    
    # Process sequences and generate embeddings
    print("Processing sequences and generating ESM embeddings...")
    embeddings_dict = process_sequences_for_prediction(
        sequences_dict,
        output_pkl=save_embeddings,
        esm_weights_path=esm_weights_path,
        device=device
    )
    
    if not embeddings_dict:
        print("Error: No embeddings were generated successfully.")
        return None
    
    # Run prediction
    print("Starting prediction...")
    predictions = predictor.predict(embeddings_dict, batch_size=batch_size)
    
    # Process results
    results = []
    for seq_id, pred_data in predictions.items():
        original_sequence = sequences_dict[seq_id]
        results.append({
            'sequence_id': seq_id,
            'predicted_efficiency': pred_data['efficiency'],
            'is_high_efficiency': pred_data['is_high_efficiency'],
            'transformer_weight': pred_data['transformer_weight'],
            'cnn_weight': pred_data['cnn_weight'],
            'lstm_weight': pred_data['lstm_weight'],
            'sequence_length': len(original_sequence),
            'sequence_preview': original_sequence[:50] + '...' if len(original_sequence) > 50 else original_sequence,
            'full_sequence': original_sequence
        })
    
    # Create the results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by predicted efficiency
    results_df = results_df.sort_values('predicted_efficiency', ascending=False)
    
    # Display results
    print("\n" + "="*80)
    print("Prediction Summary:")
    print("="*80)
    print(f"Total number of sequences: {len(results_df)}")
    print(f"Number of high-efficiency proteins (efficiency > 0.3): {results_df['is_high_efficiency'].sum()}")
    print(f"Maximum efficiency: {results_df['predicted_efficiency'].max():.4f}")
    print(f"Minimum efficiency: {results_df['predicted_efficiency'].min():.4f}")
    print(f"Mean efficiency: {results_df['predicted_efficiency'].mean():.4f}")
    print(f"Efficiency standard deviation: {results_df['predicted_efficiency'].std():.4f}")
    
    # Calculate average branch weights
    avg_transformer = results_df['transformer_weight'].mean()
    avg_cnn = results_df['cnn_weight'].mean()
    avg_lstm = results_df['lstm_weight'].mean()
    print(f"Average branch weights - Transformer: {avg_transformer:.3f}, CNN: {avg_cnn:.3f}, LSTM: {avg_lstm:.3f}")
    
    print("\nPrediction results for all sequences:")
    print("-" * 120)
    print(f"{'No.':<4} {'ID':<15} {'Predicted':<10} {'Class':<8} {'Transformer':<12} {'CNN':<8} {'LSTM':<8} {'Seq_len':<10} {'Seq_preview':<30}")
    print("-" * 120)
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        classification = 'High' if row['is_high_efficiency'] else 'Low'
        print(f"{i+1:<4} {row['sequence_id']:<15} {row['predicted_efficiency']:.4f}    {classification:<8} "
              f"{row['transformer_weight']:.3f}       {row['cnn_weight']:.3f}    {row['lstm_weight']:.3f}    "
              f"{row['sequence_length']:<10} {row['sequence_preview']:<30}")
    
    # Save results - save all prediction results
    if output_csv:
        # Save full results
        final_df = results_df[['sequence_id', 'predicted_efficiency', 'full_sequence']]
        final_df.to_csv(output_csv, sep='\t', index=False)
        
        print(f"\nResults saved to: {output_csv}")
        print(f"Saved {len(final_df)} sequences in total.")
    
    return results_df

# ==================== Command-line Interface =====================
def main():
    import argparse

    def str2bool(value):
        """Parse common true/false strings from the command line."""
        if isinstance(value, bool):
            return value
        value = value.lower()
        if value in ('yes', 'true', 't', 'y', '1'):
            return True
        if value in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected: true or false.')
    
    parser = argparse.ArgumentParser(description='Predict protein efficiency from a TXT file (ensemble high-dimensional model) - output all prediction results')
    parser.add_argument('--input', '-i', required=True, help='Input TXT file path')
    parser.add_argument('--model', '-m', required=True, help='Path to the trained model')
    parser.add_argument('--scaler', '-s', required=True, help='Path to the scaler file')
    parser.add_argument('--output', '-o', default='all_predictions.txt', help='Output CSV file path (contains all prediction results)')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--save_embeddings', '-e', help='Path to save embeddings as a pkl file (optional)')
    parser.add_argument('--target_dim', '-t', type=int, default=1152, help='Target feature dimension')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='Whether to use GPU if CUDA is available: true or false')
    parser.add_argument('--gpu_id', type=int, default=0, help='CUDA GPU ID to use when --use_gpu true')
    parser.add_argument(
        '--esm_weights',
        default='./esmc_600m_2024_12_v0.pth',
        help='Path to the local ESM-C weight file'
    )
    
    args = parser.parse_args()
    
    # Print CUDA information
    print("="*60)
    print("Protein Efficiency Prediction Tool (Ensemble High-Dimensional Model)")
    print("Output: all predicted protein results")
    print("="*60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.get_device_name(0)}")
    print(f"Input file: {args.input}")
    print(f"Model file: {args.model}")
    print(f"Scaler file: {args.scaler}")
    print(f"Output file: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print(f"Target feature dimension: {args.target_dim}")
    print(f"Use GPU: {args.use_gpu}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"ESM-C weight file: {args.esm_weights}")
    if args.save_embeddings:
        print(f"Save embeddings to: {args.save_embeddings}")
    print("="*60)
    
    results = predict_from_txt(
        txt_file_path=args.input,
        model_path=args.model,
        scaler_path=args.scaler,
        output_csv=args.output,
        batch_size=args.batch_size,
        save_embeddings=args.save_embeddings,
        target_dim=args.target_dim,
        esm_weights_path=args.esm_weights,
        use_gpu=args.use_gpu,
        gpu_id=args.gpu_id
    )
    
    if results is not None:
        print(f"\nPrediction completed! Processed {len(results)} sequences in total.")
        print(f"High-efficiency proteins (efficiency > 0.3): {results['is_high_efficiency'].sum()}")
        print(f"Non-high-efficiency proteins: {len(results) - results['is_high_efficiency'].sum()}")
        print(f"All prediction results saved to: {args.output}")
        
        # Display efficiency distribution
        efficiency_ranges = {
            '0.0-0.1': len(results[(results['predicted_efficiency'] >= 0.0) & (results['predicted_efficiency'] < 0.1)]),
            '0.1-0.2': len(results[(results['predicted_efficiency'] >= 0.1) & (results['predicted_efficiency'] < 0.2)]),
            '0.2-0.3': len(results[(results['predicted_efficiency'] >= 0.2) & (results['predicted_efficiency'] < 0.3)]),
            '0.3-0.4': len(results[(results['predicted_efficiency'] >= 0.3) & (results['predicted_efficiency'] < 0.4)]),
            '0.4-0.5': len(results[(results['predicted_efficiency'] >= 0.4) & (results['predicted_efficiency'] < 0.5)]),
            '0.5-0.6': len(results[(results['predicted_efficiency'] >= 0.5) & (results['predicted_efficiency'] < 0.6)]),
            '0.6-0.7': len(results[(results['predicted_efficiency'] >= 0.6) & (results['predicted_efficiency'] <= 0.7)])
        }
        
        print("\nEfficiency distribution:")
        for range_name, count in efficiency_ranges.items():
            percentage = (count / len(results)) * 100
            print(f"  {range_name}: {count} sequences ({percentage:.1f}%)")
            
    else:
        print("\nPrediction failed. Please check the input file and model paths.")

if __name__ == "__main__":
    main()
