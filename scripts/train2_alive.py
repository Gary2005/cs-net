import yaml
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from models.model2 import Model2
from dataset.tick_dataset2 import TickDataset2
from tqdm import tqdm
import wandb
import os
import shutil

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_lr_scheduler(optimizer, config, num_training_steps):
    from torch.optim.lr_scheduler import LambdaLR
    import math

    warmup_steps = config['training']['warmup_steps']
    min_lr_ratio = config['training'].get('min_lr_ratio', 0.1)  # 最低降到 10%

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # 加 min_lr
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)

def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    correct_predictions = 0
    total_predictions = 0
    
    total_val = 32

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating", leave=False):
            batch, nxt_kill, nxt_death, ct_win, alive_info, player_idx = batch_data

            alive_info = alive_info.to(device).float()  # shape: (batch, )
            player_idx = player_idx.to(device).unsqueeze(1).long()  # shape: (batch, 1)
            batch = batch.to(device)
            
            # Forward pass - predict next ticks
            logits = model(batch, player_idx)  # (batch, 1)
            logits = logits.squeeze(-1)  # (batch,)
            
            
            # Compute loss
            loss = nn.functional.binary_cross_entropy_with_logits(logits, alive_info)
            
            total_loss += loss.item()
            num_batches += 1

            # compute accuracy
            predictions = (logits > 0).float()  # (batch,)
            correct_predictions += (predictions == alive_info).sum().item()
            total_predictions += alive_info.size(0)

            total_val -= 1
            if total_val <= 0:
                break

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return total_loss / num_batches, accuracy

def main():
    parser = argparse.ArgumentParser(description="Pretraining script")
    parser.add_argument('--config', type=str, default='config/model2_alive.yaml', help='Path to the configuration file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cuda" or "cpu")')
    args = parser.parse_args()
    config = load_config(args.config)

    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)

    # clear checkpoint dir if resume is not specified
    if args.resume is None:
        for filename in os.listdir(config['training']['checkpoint_dir']):
            file_path = os.path.join(config['training']['checkpoint_dir'], filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    
    # copy config file to checkpoint dir for reference
    config_save_path = os.path.join(
        config['training']['checkpoint_dir'],
        os.path.basename(args.config)
    )

    shutil.copy(args.config, config_save_path)

    print(f"Config file copied to {config_save_path}")

    # copy 'demoparser_utils/tokenizer.yaml' to checkpoint dir for reference
    tokenizer_config_path = 'demoparser_utils/tokenizer.yaml'
    tokenizer_config_save_path = os.path.join(
        config['training']['checkpoint_dir'],
        os.path.basename(tokenizer_config_path)
    )

    shutil.copy(tokenizer_config_path, tokenizer_config_save_path)

    print(f"Tokenizer config file copied to {tokenizer_config_save_path}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    print("Loaded configuration:")
    print("=" * 30)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 30)

    train_dataset = TickDataset2(args.config, split="train", shuffle_shards=True, skip_no_death=False, skip_no_alive=True)
    val_dataset = TickDataset2(args.config, split="val", shuffle_shards=False, skip_no_death=False, skip_no_alive=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    len_train = len(train_loader)
    len_val = len(val_loader)
    print(f"Number of training batches: {len_train}")
    print(f"Number of validation batches: {len_val}")

    model = Model2(config).to(device)

    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=config['training']['weight_decay'], lr=config['training']['lr'])

    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Number of total parameters: {sum(p.numel() for p in model.parameters())}")



    grad_accum_steps = config["training"].get("grad_accum_steps", 1)
    num_training_steps = ((len_train + grad_accum_steps - 1) // grad_accum_steps) * config['training']['num_epochs']

    print(f"Total training steps: {num_training_steps}")

    scheduler = get_lr_scheduler(optimizer, config, num_training_steps)

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']

    wandb.init(project=config['logging']['project_name'], config=config)

    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")

        model.train()
        total_loss = 0
        total_accu = 0
        num_batches = 0

        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", total=len_train)

        for batch_idx, batch_data in enumerate(progress_bar):
            batch, nxt_kill, nxt_death, ct_win, alive_info, player_idx = batch_data

            alive_info = alive_info.to(device).float()  # shape: (batch, )
            player_idx = player_idx.to(device).unsqueeze(1).long()  # shape: (batch, 1)
            batch = batch.to(device)
            
            # Forward pass - predict next ticks
            logits = model(batch, player_idx)  # (batch, 1)
            logits = logits.squeeze(-1)  # (batch,)
            
            
            # Compute loss
            loss = nn.functional.binary_cross_entropy_with_logits(logits, alive_info)
            
            loss /= grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len_train:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accum_steps
            num_batches += 1

            train_accu = ((logits > 0).float() == alive_info).float().mean().item()
            total_accu += train_accu

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}',
                'avg_accu': f'{total_accu / num_batches:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

            wandb.log({
                'train_loss': loss.item(),
                'avg_train_loss': total_loss / num_batches,
                'avg_train_accu': total_accu / num_batches,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

            if batch_idx % config['logging']['test'] == 0:
                val_loss, val_accuracy = validate(model, val_loader, device)
                print(f"\nEpoch {epoch+1} Batch {batch_idx}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
                wandb.log({
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                })
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(
                        config['training']['checkpoint_dir'],
                        f'model_epoch{epoch+1}_batch{batch_idx}_valloss{val_loss:.4f}.pt'
                    )
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss
                    }, checkpoint_path)
                    print(f"New best model saved to {checkpoint_path}")
                
                model.train()

        print(f"Epoch {epoch+1} completed. Average Training Loss: {total_loss / num_batches:.4f}, Average Training Accuracy: {total_accu / num_batches:.4f}")

        wandb.log({
            'epoch_avg_train_loss': total_loss / num_batches,
            'epoch_avg_train_accuracy': total_accu / num_batches
        })
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss
        }, os.path.join(config['training']['checkpoint_dir'], f'latest_model.pt'))

if __name__ == "__main__":
    main()