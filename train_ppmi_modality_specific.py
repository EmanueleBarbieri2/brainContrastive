
"""
Train Modality-Specific GNN-CLIP on PPMI dataset.

This model uses different encoder architectures for FC and SC:
- FC: Deep GAT (3 layers, attention, multi-scale)
- SC: Shallow GNN (2 layers, BatchNorm, local structure)

Like CLIP uses different encoders for text (Transformer) and images (CNN/ViT),
we use specialized encoders optimized for each brain connectivity modality.
"""

import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from modality_specific_gnn import create_modality_specific_gnn_clip
from data_loading import create_dataloaders


def contrastive_loss(fc_embed, sc_embed, temperature=0.07):
    """Symmetric contrastive loss (InfoNCE)."""
    fc_embed = fc_embed / fc_embed.norm(dim=-1, keepdim=True)
    sc_embed = sc_embed / sc_embed.norm(dim=-1, keepdim=True)
    
    logits = (fc_embed @ sc_embed.T) / temperature
    
    batch_size = fc_embed.shape[0]
    labels = torch.arange(batch_size, device=fc_embed.device)
    
    loss_fc = nn.functional.cross_entropy(logits, labels)
    loss_sc = nn.functional.cross_entropy(logits.T, labels)
    loss = (loss_fc + loss_sc) / 2
    
    return loss


def compute_metrics(fc_embed, sc_embed):
    """Compute alignment metrics."""
    fc_norm = fc_embed / fc_embed.norm(dim=-1, keepdim=True)
    sc_norm = sc_embed / sc_embed.norm(dim=-1, keepdim=True)
    
    # Paired similarity
    pair_sims = (fc_norm * sc_norm).sum(dim=-1)
    paired_sim = pair_sims.mean().item()
    
    # Unpaired similarity
    sim_matrix = fc_norm @ sc_norm.T
    mask = ~torch.eye(fc_norm.shape[0], dtype=torch.bool, device=fc_norm.device)
    unpaired_sim = sim_matrix[mask].mean().item()
    
    # Alignment gap (THIS IS WHAT WE WANT TO MAXIMIZE!)
    alignment_gap = paired_sim - unpaired_sim
    
    # Within-modality similarity
    fc_within = (fc_norm @ fc_norm.T)[mask].mean().item()
    sc_within = (sc_norm @ sc_norm.T)[mask].mean().item()
    
    return {
        'pair_sim': paired_sim,
        'unpaired_sim': unpaired_sim,
        'alignment_gap': alignment_gap,
        'fc_within': fc_within,
        'sc_within': sc_within
    }


def train_epoch(model, train_loader, optimizer, device, temperature):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_metrics = {k: 0 for k in ['pair_sim', 'unpaired_sim', 'alignment_gap', 'fc_within', 'sc_within']}
    
    pbar = tqdm(train_loader, desc="Training")
    for fc_matrices, sc_matrices in pbar:
        fc_matrices = fc_matrices.to(device)
        sc_matrices = sc_matrices.to(device)
        
        fc_embed, sc_embed, _ = model(fc_matrices, sc_matrices)
        
        loss = contrastive_loss(fc_embed, sc_embed, temperature)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            metrics = compute_metrics(fc_embed, sc_embed)
        
        total_loss += loss.item()
        for k, v in metrics.items():
            all_metrics[k] += v
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'gap': f'{metrics["alignment_gap"]:.4f}'
        })
    
    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        **{k: v / n_batches for k, v in all_metrics.items()}
    }


def validate(model, val_loader, device, temperature):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_metrics = {k: 0 for k in ['pair_sim', 'unpaired_sim', 'alignment_gap', 'fc_within', 'sc_within']}
    
    with torch.no_grad():
        for fc_matrices, sc_matrices in val_loader:
            fc_matrices = fc_matrices.to(device)
            sc_matrices = sc_matrices.to(device)
            
            fc_embed, sc_embed, _ = model(fc_matrices, sc_matrices)
            
            loss = contrastive_loss(fc_embed, sc_embed, temperature)
            metrics = compute_metrics(fc_embed, sc_embed)
            
            total_loss += loss.item()
            for k, v in metrics.items():
                all_metrics[k] += v
    
    n_batches = len(val_loader)
    return {
        'loss': total_loss / n_batches,
        **{k: v / n_batches for k, v in all_metrics.items()}
    }


def main():
    config = {
        'data_dir': 'data_ppmi',
        'fc_dim': 90,
        'sc_dim': 90,
        'num_nodes': 90,
        'embed_dim': 256,
        'fc_hidden_dims': [128, 128, 128],  # Deep for FC
        'sc_hidden_dims': [96, 96],         # Shallow for SC
        'num_heads': 4,                     # For GAT
        'dropout': 0.2,
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'temperature': 0.07,
        'checkpoint_dir': 'checkpoints_ppmi_modality_specific',
        'patience': 15,
        'lr_decay_patience': 5,
        'lr_decay_factor': 0.5,
        'save_every': 10,
    }
    
    print("="*70)
    print("Training Modality-Specific GNN-CLIP")
    print("="*70)
    print(f"\nArchitecture Philosophy:")
    print(f"  Just like CLIP uses different encoders for text and images,")
    print(f"  we use specialized encoders for FC and SC modalities.")
    print(f"\nFC Encoder (Functional Connectivity):")
    print(f"  • Type: Graph Attention Network (GAT)")
    print(f"  • Depth: 3 layers (captures distributed patterns)")
    print(f"  • Hidden dims: {config['fc_hidden_dims']}")
    print(f"  • Attention heads: {config['num_heads']}")
    print(f"  • Normalization: LayerNorm")
    print(f"  • Pooling: Attention-weighted (learns important nodes)")
    print(f"  • Rationale: FC has dense, distributed connectivity patterns")
    print(f"\nSC Encoder (Structural Connectivity):")
    print(f"  • Type: Standard GNN")
    print(f"  • Depth: 2 layers (focuses on local structure)")
    print(f"  • Hidden dims: {config['sc_hidden_dims']}")
    print(f"  • Normalization: BatchNorm")
    print(f"  • Pooling: Attention-weighted")
    print(f"  • Rationale: SC has sparse, local structural pathways")
    print(f"\nShared Embedding Space:")
    print(f"  • Dimension: {config['embed_dim']}")
    print(f"  • Both modalities project to same space for alignment")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Save config
    with open(f"{config['checkpoint_dir']}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create model
    print("Creating modality-specific model...")
    model = create_modality_specific_gnn_clip(
        fc_dim=config['fc_dim'],
        sc_dim=config['sc_dim'],
        embed_dim=config['embed_dim'],
        num_nodes=config['num_nodes'],
        dropout=config['dropout']
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    fc_params = sum(p.numel() for p in model.fc_encoder.parameters())
    sc_params = sum(p.numel() for p in model.sc_encoder.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  FC encoder: {fc_params:,} ({fc_params/total_params*100:.1f}%)")
    print(f"  SC encoder: {sc_params:,} ({sc_params/total_params*100:.1f}%)")
    print(f"  Asymmetry ratio: {fc_params/sc_params:.2f}x (FC has more params)")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_dir=f"{config['data_dir']}/train",
        val_dir=f"{config['data_dir']}/val",
        batch_size=config['batch_size'],
        file_extension='.csv',
        expected_shape1=(config['fc_dim'], config['fc_dim']),
        expected_shape2=(config['sc_dim'], config['sc_dim']),
    )
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config['lr_decay_factor'],
        patience=config['lr_decay_patience']
    )
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    history = []
    best_alignment_gap = -float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-"*70)
        
        train_metrics = train_epoch(model, train_loader, optimizer, device, config['temperature'])
        val_metrics = validate(model, val_loader, device, config['temperature'])
        
        print(f"\nTrain: loss={train_metrics['loss']:.4f}, "
              f"gap={train_metrics['alignment_gap']:.4f}, "
              f"pair={train_metrics['pair_sim']:.4f}, "
              f"unpaired={train_metrics['unpaired_sim']:.4f}")
        print(f"Val:   loss={val_metrics['loss']:.4f}, "
              f"gap={val_metrics['alignment_gap']:.4f}, "
              f"pair={val_metrics['pair_sim']:.4f}, "
              f"unpaired={val_metrics['unpaired_sim']:.4f}")
        
        # Additional diagnostics
        print(f"       fc_within={val_metrics['fc_within']:.4f}, "
              f"sc_within={val_metrics['sc_within']:.4f}")
        
        # Check for feature collapse
        if val_metrics['sc_within'] > 0.5:
            print(f"  ⚠️  WARNING: High SC within-similarity ({val_metrics['sc_within']:.4f})")
            print(f"       May indicate feature collapse!")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # LR scheduler step
        scheduler.step(val_metrics['alignment_gap'])
        
        # Check for improvement
        if val_metrics['alignment_gap'] > best_alignment_gap:
            best_alignment_gap = val_metrics['alignment_gap']
            epochs_without_improvement = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'metrics': val_metrics
            }
            torch.save(checkpoint, f"{config['checkpoint_dir']}/best_model.pt")
            print(f"✓ NEW BEST! Alignment gap: {val_metrics['alignment_gap']:.4f}")
            
            # Compare to baselines
            if val_metrics['alignment_gap'] > 0.0212:
                improvement = (val_metrics['alignment_gap'] - 0.0212) / 0.0212 * 100
                print(f"  🎉 Better than proper baseline by {improvement:.1f}%!")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs "
                  f"(best gap: {best_alignment_gap:.4f})")
        
        # Early stopping
        if epochs_without_improvement >= config['patience']:
            print(f"\n{'='*70}")
            print(f"Early stopping triggered after {epoch} epochs")
            print(f"Best alignment gap: {best_alignment_gap:.4f}")
            print(f"{'='*70}")
            break
        
        # Save periodic checkpoints
        if epoch % config['save_every'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'metrics': val_metrics
            }
            torch.save(checkpoint, f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch}.pt")
    
    # Save final model
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'metrics': val_metrics
    }
    torch.save(final_checkpoint, f"{config['checkpoint_dir']}/final_model.pt")
    
    # Save history
    with open(f"{config['checkpoint_dir']}/history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nBest validation alignment gap: {best_alignment_gap:.4f}")
    print(f"Stopped at epoch: {epoch}")
    print(f"\nComparison to baselines:")
    print(f"  Baseline (no improvements):    gap = 0.0180")
    print(f"  Proper (early stopping):       gap = 0.0212")
    print(f"  Modality-specific (this run):  gap = {best_alignment_gap:.4f}")
    
    if best_alignment_gap > 0.0212:
        improvement = (best_alignment_gap - 0.0212) / 0.0212 * 100
        print(f"\n  ✓ Improvement over proper: +{improvement:.1f}%")
    else:
        decline = (0.0212 - best_alignment_gap) / 0.0212 * 100
        print(f"\n  ✗ Decline from proper: -{decline:.1f}%")


if __name__ == '__main__':
    main()
