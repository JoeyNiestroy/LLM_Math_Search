"""
This is the core training script for the FFN style model, designed to be called directly with command line args. See below for the specifics. Designed to train a single model at a time on one GPU.
Can be called concurrently without issues for 'distributed training' but is pretty fast on A100 with models < 500M params.

This script works with the LastHiddenFullDataset, so extract_last_hidden MUST be built before you can use this. Will not add functionality for ShardedHiddenStateDataset (see comments in optimized_dataset)

RUN WITH use_residual, the other model was for debugging earlier. MAY NOT WORK with the older model arch. #TODO maybe fix this if time allows

I would avoid running with num_workers > 0, huge memory hit and does not speed up anything

"""

import argparse
import json
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader , random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings('ignore', '.*DataLoader worker.*')

# 
from optimized_dataset import LastHiddenFullDataset

from FFN_Model import FFNBinaryClassifier, ResidualFFNBinaryClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)





#Provides training analytics 
def compute_metrics(probs, labels, threshold=0.5):
    """Compute accuracy, precision, recall, F1."""
    preds = (probs > threshold).long()
    labels = labels.long()
    
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


#Core function for running one epoch, I would recommend running with amp and accumulation
def train_epoch(model, dataloader, optimizer, device, grad_clip=1.0, use_amp=False, scaler=None, accumulation_steps=2):
    """Train for one epoch with gradient accumulation support."""
    model.train()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):

        hidden = batch["hidden"].to(device)

        labels = batch["label"].to(device)
        
        if use_amp:
            with autocast(dtype=torch.float16):
                outputs = model(hidden, labels=labels)
                loss = outputs["loss"] / accumulation_steps 
            

            #After adding stablizing factor to ffn this is warining is deprc
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss at batch {batch_idx}! Skipping.")
                continue
            
            scaler.scale(loss).backward()
            
            # only update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(hidden, labels=labels)
            loss = outputs["loss"] / accumulation_steps  # Scale loss
            
            #see above comment
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss at batch {batch_idx}! Skipping.")
                continue
            
            loss.backward()
            
            # Accumuation logic
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps  # unscale for logging
        all_probs.append(outputs["probs"].detach().cpu())
        all_labels.append(labels.cpu())
        
        pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_probs, all_labels)
    
    return avg_loss, metrics



#Simple eval function, wrapped in no grad. Provides metrics

@torch.no_grad()
def evaluate(model, dataloader, device, use_amp=False):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        hidden = batch["hidden"].to(device)

        labels = batch["label"].to(device)
        
        if use_amp:
            with autocast(dtype=torch.float16):
                outputs = model(hidden, labels=labels)
                loss = outputs["loss"]
        else:
            outputs = model(hidden, labels=labels)
            loss = outputs["loss"]
        
        total_loss += loss.item()
        all_probs.append(outputs["probs"].cpu())
        all_labels.append(labels.cpu())
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_probs, all_labels)
    
    return avg_loss, metrics





#Main function, loads model and data and traings moodel according to params
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load full dataset
    logger.info("Loading dataset...")
    full_dataset = LastHiddenFullDataset(root=args.train_data)
    logger.info(f"Total samples: {len(full_dataset):,}")
    
    # Split into train/val
    if args.val_split > 0:
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        
        logger.info(f"Splitting dataset: {args.val_split*100:.1f}% validation")
        
        # Use generator for reproducibility
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=generator
        )
        
        logger.info(f"Train samples: {len(train_dataset):,}")
        logger.info(f"Val samples: {len(val_dataset):,}")
    else:
        train_dataset = full_dataset
        val_dataset = None
        logger.info(f"Train samples: {len(train_dataset):,}")
        logger.info("No validation split")
    
    # Get hidden dimension, won't change with model but good check if the indexing got fucked up somewhere
    sample = full_dataset[0]
    hidden_dim = sample["hidden"].shape[-1]
    logger.info(f"Hidden dimension: {hidden_dim}")
    
    # Create dataloaders, see the num_worked warning at the top
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == "cuda" else False,
            persistent_workers=True if args.num_workers > 0 else False,
        )
    
    logger.info(f"Train batches: {len(train_loader):,}")
    if val_loader:
        logger.info(f"Val batches: {len(val_loader):,}")
    
    # Create model
    logger.info("Initializing FFN model...")
    if args.use_residual:
        model = ResidualFFNBinaryClassifier(
            in_dim=hidden_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_layer_norm=args.use_layer_norm,
        )
    else:
        hidden_dims = [int(d) for d in args.hidden_dims.split(',')]
        model = FFNBinaryClassifier(
            in_dim=hidden_dim,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
            use_batch_norm=args.use_batch_norm,
        )
    
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Optimizer & scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.1,
    )
    
    scaler = None
    if args.use_amp and device.type == "cuda":
        scaler = GradScaler()
        logger.info("Using mixed precision training (FP16)")
    
    # Training loop, best model is decided based on F1. #TODO make this param
    best_val_f1 = 0.0
    history = []
    
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, device, args.grad_clip,
            use_amp=args.use_amp, scaler=scaler, 
            accumulation_steps=args.accumulation_steps
        )
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Train Metrics: Acc={train_metrics['accuracy']:.4f}, "
                   f"P={train_metrics['precision']:.4f}, "
                   f"R={train_metrics['recall']:.4f}, "
                   f"F1={train_metrics['f1']:.4f}")
        
        val_loss = None
        val_metrics = None
        if val_loader is not None:
            val_loss, val_metrics = evaluate(model, val_loader, device, 
                                            use_amp=args.use_amp)
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Metrics: Acc={val_metrics['accuracy']:.4f}, "
                       f"P={val_metrics['precision']:.4f}, "
                       f"R={val_metrics['recall']:.4f}, "
                       f"F1={val_metrics['f1']:.4f}")
        
        scheduler.step()
        
        epoch_history = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_metrics": train_metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        if val_loss is not None:
            epoch_history["val_loss"] = val_loss
            epoch_history["val_metrics"] = val_metrics
        history.append(epoch_history)
        
        # Save best model
        current_f1 = val_metrics["f1"] if val_metrics is not None else train_metrics["f1"]
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            logger.info(f"New best F1: {best_val_f1:.4f} - Saving checkpoint...")
            
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_val_f1,
                "args": vars(args),
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
        
        # Save last checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(checkpoint, output_dir / "last_model.pt")
    
    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\nTraining complete! Best F1: {best_val_f1:.4f}")
    logger.info(f"Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train FFN binary classifier on sharded hidden states",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data args
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to sharded training data directory")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Fraction of data to use for validation (e.g., 0.1 = 10%%)")
    parser.add_argument("--output_dir", type=str, default="checkpoints_ffn")
    parser.add_argument("--stride_k", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--preload_shards", action="store_true",
                       help="Preload all shards into RAM (faster but uses more memory)")
    
    # Model args
    parser.add_argument("--use_residual", action="store_true",
                       help="Use residual FFN instead of simple FFN")
    parser.add_argument("--hidden_dims", type=str, default="512,256",
                       help="Comma-separated hidden dims for simple FFN (e.g., '512,256')")
    parser.add_argument("--hidden_dim", type=int, default=512,
                       help="Hidden dim for residual FFN")
    parser.add_argument("--num_layers", type=int, default=3,
                       help="Number of layers for residual FFN")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_batch_norm", action="store_true",
                       help="Use batch norm in simple FFN")
    parser.add_argument("--use_layer_norm", action="store_true",
                       help="Use layer norm in residual FFN")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--accumulation_steps", type=int, default=64)
    
    # System args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    main(args)