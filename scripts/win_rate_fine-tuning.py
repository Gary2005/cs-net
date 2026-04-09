import yaml
import argparse
from models.tfm_model import TickTransformerModel
from models.tfm_model_rope import TickTransformerModelRope
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset.streaming_dataset_win_rate import TickStreamingWinRateDataset
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
    """
    Create learning rate scheduler with warmup using HuggingFace Transformers.
    """
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
    
    warmup_steps = config['training']['warmup_steps']
    scheduler_type = config['training']['scheduler']

    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return scheduler

class Win_Rate_Prediction_Head(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Output layer for binary classification (win/loss)
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x).squeeze(-1)  # Return scalar logits, shape: (batch_size,)

class Win_Rate_Prediction_Model(nn.Module):
    def __init__(self, base_model, prediction_head):
        super().__init__()
        self.base_model = base_model
        self.prediction_head = prediction_head

    def forward(self, x):
        # x shape: (batch, ticks, seq_len)
        features = self.base_model.get_intermediate_data(x)  # shape: (batch, ticks, feature_dim)
        # only keep the last ticks' features for prediction
        last_tick_features = features[:, -1, :]  # shape: (batch, feature_dim)
        # cat the mean of all ticks' features to the last tick's features
        mean_features = features.mean(dim=1)  # shape: (batch, feature_dim)
        tick_features = torch.cat([last_tick_features, mean_features], dim=-1)  # shape: (batch, feature_dim * 2)
        logits = self.prediction_head(tick_features)  # shape: (batch,)
        return logits


def validate(model, dataloader, device, config):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating", leave=False):
            batch, target = batch_data
            target = target.to(device).float()  # shape: (batch,)
            batch = batch.to(device)
            
            # Forward pass - predict next ticks
            logits = model(batch)  # (batch, )
            
            
            # Compute loss
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            
            total_loss += loss.item()
            num_batches += 1

            # compute accuracy
            predictions = (logits > 0).float()  # (batch,)
            correct_predictions += (predictions == target).sum().item()
            total_predictions += target.size(0)
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return total_loss / num_batches, accuracy

def main():
    parser = argparse.ArgumentParser(description="Pretraining script")
    parser.add_argument('--config', type=str, default='config/tfm_win_rate_fine-tuning.yaml', help='Path to the configuration file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()

    config = load_config(args.config)

    # 确保 checkpoint_dir 存在
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)

    # 复制 config 文件到 checkpoint_dir
    config_save_path = os.path.join(
        config['training']['checkpoint_dir'],
        os.path.basename(args.config)
    )

    shutil.copy(args.config, config_save_path)

    print(f"Config file copied to {config_save_path}")

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    print("Device:", device)

    # print the configuration for verification
    print("Loaded configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    train_dataset = TickStreamingWinRateDataset(config_path=args.config, split="train", shuffle_shards=True)
    val_dataset = TickStreamingWinRateDataset(config_path=args.config, split="val", shuffle_shards=False)

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
    
    # Load pre-trained model
    if config['model']['model_name'] == "TickTransformerModel":
        base_model = TickTransformerModel(config['model']).to(device)
        if config['training']['from_scratch']:
            print("Training from scratch without loading pre-trained weights.")
        else:
            checkpoint = torch.load(config['training']['base_model_path'], map_location=device)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pre-trained model from {config['training']['base_model_path']}")
    elif config['model']['model_name'] == "TickTransformerModelROPE":
        base_model = TickTransformerModelRope(config['model']).to(device)
        if config['training']['from_scratch']:
            print("Training from scratch without loading pre-trained weights.")
        else:
            checkpoint = torch.load(config['training']['base_model_path'], map_location=device)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pre-trained model from {config['training']['base_model_path']}")
    else:
        raise ValueError(f"Unsupported model name: {config['model']['model_name']}")

    prediction_head = Win_Rate_Prediction_Head(
        input_dim=config['model']['embed_dim'] * 2,
        hidden_dim=config['model']['win_rate_hidden_dim'],
        num_hidden_layers=config['model']['win_rate_hidden_layers']
    ).to(device)

    if config['training']['use_lora']:
        from peft import get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(
            r=config['training']['lora_r'],
            lora_alpha=config['training']['lora_alpha'],
            lora_dropout=config['training']['lora_dropout'],
            target_modules=["linear1", "linear2"],
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        base_model = get_peft_model(base_model, peft_config)
        print("Applied LoRA to the base model for fine-tuning.")
        # print trainable parameters for verification
        base_model.print_trainable_parameters()

    model = Win_Rate_Prediction_Model(base_model, prediction_head).to(device)
    trainable_params = []

    if config['training']['learning_rate_embedder'] == 0:
        for param in model.base_model.embedder.parameters():
            param.requires_grad = False
    else:
        trainable_params.append({
            'params': model.base_model.embedder.parameters(),
            'lr': config['training']['learning_rate_embedder']
        })

    for param in model.base_model.decoder.parameters():
        param.requires_grad = False

    trainable_params.append({
        'params': model.prediction_head.parameters(),
        'lr': config['training']['learning_rate_prediction_head']
    })
    trainable_params.append({
        'params': model.base_model.processor.parameters(),
        'lr': config['training']['learning_rate_processor']
    })

    optimizer = optim.AdamW(
        trainable_params,
        weight_decay=config['training']['weight_decay']
    )

    print(f"Number of trainable parameters: {sum(sum(p.numel() for p in group['params']) for group in trainable_params)}")

    grad_accum_steps = config["training"].get("grad_accum_steps", 1)
    num_training_steps = ((len_train + grad_accum_steps - 1) // grad_accum_steps) * config['training']['num_epochs']

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
    
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")

        '''
        Training loop
        '''

        model.train()
        total_loss = 0
        total_accu = 0
        num_batches = 0

        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", total=len_train)
        
        for batch_idx, batch_data in enumerate(progress_bar):
            batch, target = batch_data
            target = target.to(device).float()  # shape: (batch,)
            batch = batch.to(device)  # (batch, ticks, seq_len)
            
            # Forward pass - predict next ticks
            logits = model(batch)  # (batch, )
            
            # Compute loss
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['max_grad_norm']
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * grad_accum_steps  # scale back to original loss value
            num_batches += 1
            
            train_accu = ((logits > 0).float() == target).float().mean().item()
            total_accu += train_accu

            # Update progress bar
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
            
            # Log periodically
            if batch_idx % config['logging']['test'] == 0:
                # validation loop
                val_loss, val_accuracy = validate(model, val_loader, device, config)
                print(f"Epoch {epoch}, Batch {batch_idx}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                wandb.log({
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                })
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss
                    }, f"{config['training']['checkpoint_dir']}/best_checkpoint.pth")

                model.train()  # switch back to training mode after validation
        
        if len_train % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['max_grad_norm']
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        print(f"Epoch {epoch} completed. Average Training Loss: {total_loss / num_batches:.4f}, Average Training Accuracy: {total_accu / num_batches:.4f}")
        wandb.log({'avg_train_loss_epoch': total_loss / num_batches, 'avg_train_accuracy_epoch': total_accu / num_batches})

        # save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss
        }, f"{config['training']['checkpoint_dir']}/latest_checkpoint.pth")

if __name__ == "__main__":
    main()