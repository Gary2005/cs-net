import yaml
import argparse
from models.tfm_model import TickTransformerModel, TickTransformerModelLearnablePositional
from models.tfm_model_rope import TickTransformerModelRope
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset.ddp_streaming_dataset_all_label import TickStreamingAllLabelsDataset as TickStreamingDataset
from tqdm import tqdm
import wandb
import os
import torch.distributed as dist

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

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

def validate(model, dataloader, device, config):
    """DDP-aware validation with progress bar"""
    model.eval()

    total_loss = 0.0
    num_batches = 0
    correct_predictions = 0
    total_predictions = 0

    max_batches = config['training'].get('validation_batches', 32)

    # 只在 rank 0 显示进度条
    is_main = (dist.get_rank() == 0)
    iterator = dataloader
    if is_main:
        iterator = tqdm(dataloader, total=min(max_batches, len(dataloader)), desc="Validation")

    with torch.no_grad():
        for i, batch_data in enumerate(iterator):
            if i >= max_batches:
                break

            batch, _, _, _, _ = batch_data
            batch = batch.to(device)

            logits = model(batch, teacher_forcing=True)
            targets = batch[:, 1:, :]

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=config['data']['pad_token']
            )

            total_loss += loss.item()
            num_batches += 1

            # accuracy
            predictions = logits.argmax(dim=-1)

            mask = (targets != config['data']['pad_token'])

            correct_predictions += ((predictions == targets) & mask).sum().item()
            total_predictions += mask.sum().item()

    # -------------------------
    # 🔥 DDP 汇总（关键）
    # -------------------------
    total_loss = torch.tensor(total_loss, device=device)
    num_batches = torch.tensor(num_batches, device=device)
    correct_predictions = torch.tensor(correct_predictions, device=device)
    total_predictions = torch.tensor(total_predictions, device=device)

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_predictions, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_predictions, op=dist.ReduceOp.SUM)

    # global metrics
    avg_loss = total_loss.item() / num_batches.item()
    accuracy = (
        correct_predictions.item() / total_predictions.item()
        if total_predictions.item() > 0 else 0
    )

    return avg_loss, accuracy

def main():

    # torch.backends.cuda.enable_flash_sdp(True)

    parser = argparse.ArgumentParser(description="Pretraining script")
    parser.add_argument('--config', type=str, default='config/tfm_pretrain_config.yaml', help='Path to the configuration file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    args = parser.parse_args()

    config = load_config(args.config)
    # device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    print("Device:", device)

    # print the configuration for verification
    if is_main:
        print("Loaded configuration:")
        for key, value in config.items():
            print(f"{key}: {value}")

    train_dataset = TickStreamingDataset(config_path=args.config, split="train", shuffle_shards=True)
    val_dataset = TickStreamingDataset(config_path=args.config, split="val", shuffle_shards=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    if config['model']['model_name'] == "TickTransformerModel":
        from torch.nn.parallel import DistributedDataParallel as DDP
        if is_main:
            print("Using TickTransformerModel")
        model = TickTransformerModel(config['model']).to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    elif config['model']['model_name'] == "TickTransformerModelLP":
        from torch.nn.parallel import DistributedDataParallel as DDP
        if is_main:
            print("Using TickTransformerModelLearnablePositional")
        model = TickTransformerModelLearnablePositional(config['model']).to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    elif config['model']['model_name'] == "TickTransformerModelROPE":
        from torch.nn.parallel import DistributedDataParallel as DDP
        if is_main:
            print("Using TickTransformerModelRope")
        model = TickTransformerModelRope(config['model']).to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    else:
        raise ValueError(f"Unsupported model name: {config['model']['model_name']}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    grad_accum_steps = config["training"].get("grad_accum_steps", 1)
    num_training_steps = config['training']['max_steps']

    scheduler = get_lr_scheduler(optimizer, config, num_training_steps//grad_accum_steps)
    
    start_epoch = 0
    best_val_loss = float('inf')
    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        step = checkpoint.get('step', 0)
        best_val_loss = checkpoint.get('val_loss', float('inf'))

    if is_main:
        wandb.init(project=config['logging']['project_name'], config=config)

    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)

    if is_main:
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    max_steps = config['training']['max_steps']
    log_interval = config['logging']['log_interval']
    val_interval = config['logging']['test']

    total_loss = 0
    num_batches = 0

    optimizer.zero_grad()

    train_iter = iter(train_loader)

    progress_bar = tqdm(total=max_steps, initial=step, disable=not is_main)

    while step < max_steps:
        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_data = next(train_iter)

        batch, _, _, _, _ = batch_data
        batch = batch.to(device)

        # forward
        logits = model(batch, teacher_forcing=True)
        targets = batch[:, 1:, :]

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=config['data']['pad_token']
        )

        loss = loss / grad_accum_steps
        loss.backward()

        # grad step
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['max_grad_norm']
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # -------- DDP loss sync --------
        loss_detached = loss.detach()
        dist.all_reduce(loss_detached, op=dist.ReduceOp.AVG)

        total_loss += loss_detached.item() * grad_accum_steps
        num_batches += 1

        # -------- logging --------
        if is_main and step % log_interval == 0:
            avg_loss = total_loss / num_batches

            progress_bar.set_postfix({
                'step': step,
                'loss': f'{(loss.item() * grad_accum_steps):.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

            wandb.log({
                'train_loss': loss.item() * grad_accum_steps,
                'avg_train_loss': avg_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'step': step
            })

        # -------- validation --------
        if step % val_interval == 0:

            dist.barrier()  # 🔥 所有卡同步进入验证

            val_loss, val_accuracy = validate(model, val_loader, device, config)

            if is_main:
                print(f"[Step {step}] Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}")

                wandb.log({
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'step': step
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'step': step,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss
                    }, f"{config['training']['checkpoint_dir']}/best.pth")
                    print(f"Saved new best checkpoint at step {step} with val loss {val_loss:.4f}")

            dist.barrier()  # 🔥 所有卡同步退出验证

            model.train()

        # -------- save latest --------
        if (step % config['logging']['save_interval'] == 0 or step == max_steps - 1) and step > 0:
            dist.barrier()
            if is_main:
                torch.save({
                    'step': step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, f"{config['training']['checkpoint_dir']}/latest.pth")
                print(f"Saved checkpoint at step {step}")
            dist.barrier()

        step += 1
        if is_main:
            progress_bar.update(1)
    
    dist.barrier()
    if is_main:
        torch.save({
            'step': step,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, f"{config['training']['checkpoint_dir']}/final.pth")
        print(f"Saved final checkpoint at step {step}")
    dist.barrier()

if __name__ == "__main__":
    main()