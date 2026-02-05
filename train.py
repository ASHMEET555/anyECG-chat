import os
import torch
import argparse
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from datasets import load_dataset
import transformers
from anyecg.utils import set_random_seed, setup_logger
# from anyecg.utils import Collate_Fn
from anyecg.dataset import ReportGenCollator as Collate_Fn
from anyecg.ecg_language_modeling import ECG_Language_Model
warnings.filterwarnings("ignore", category=RuntimeWarning)

parser = argparse.ArgumentParser()
# Data
parser.add_argument('--data_path', type=str, default='./data/mimic_report.json', help='Path to the data file')
# Model
parser.add_argument('--unfreeze_ecg_model', action='store_true', help='Unfreeze ECG model')
parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
parser.add_argument('--projection_ckpt', type=str, default=None)
parser.add_argument('--ecg_model_ckpt', type=str, default=None)
parser.add_argument('--lora_ckpt', type=str, default=None)
# Training
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=int, default=0, help='CUDA device ID to use')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs')
parser.add_argument('--warmup', type=float, default=0.1, help='Warmup ratio')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--save_steps', type=int, default=1000, help='Save steps')
parser.add_argument('--output_dir', type=str, default='/mnt/sda1/xxxx/output/anyECG/debug', help='Path to the checkpoint file')
args = parser.parse_args()
def apply_share_recipe(model,use_lora):
    """
    Implements the TinyLLaVA 'Share Recipe'[cite: 825]:
    1. Encoder: Freeze bottom 75%, Train top 25%.
    2. Projector: Always Train.
    3. LLM: Frozen (unless LoRA is active).
    """
    print("\n[Configuring TinyLLava share recipe]")

    # start by freezing the entire model 
    for param in model.parameters():
        param.requires_grad = False
    # unfreze projector
    print(" - Unfreezing projector")
    for param in model.projection.parameters():
        param.requires_grad = True
    
    # handl the ecg encoder partial unfreeze 
    encoder=model.ecg_model
    layers=None
    if hasattr(encoder,'blocks'):
        layers=encoder.blocks
    elif hasattr(encoder,'layers'):
        layers=encoder.layers
    elif hasattr(encoder,'transformer'):
        layers=encoder.transformer.layers
    
    if layers is not None:
        nums_layers=len(layers)
        freeze_until=int(nums_layers*0.75)
        print(f" - ECG Encoder: Total {num_layers} layers.")
        print(f" - Freezing layers 0-{freeze_until-1}")
        print(f" - Training layers {freeze_until}-{num_layers-1}")

        for i in range(freeze_until,nums_layers):
            for param in layers[i].parameters():
                param.requires_grad = True

        if hasattr(encoder, 'norm') and encoder.norm is not None:
            for param in encoder.norm.parameters(): 
                param.requires_grad = True
    else: 
        print("! WARNING: Could not find encoder layers to split . Unfreezign Full encoder")
        for param in encoder.parameters():
            param.requires_grad = True  
    # handle the llm
    if use_lora:
        print("- Enabling lora for llm ")
        model.language_model.print_trainable_parameters()
    total_params=sum(p.numel() for p in model.parameters())
    trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" -> Total Trainable Params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)\n")


# Main execution 

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
logger = setup_logger(os.path.join(args.output_dir, 'logfile.log'))
logger.info(args)
set_random_seed(args.seed)

# Model
print("Initializing hybrid model (phi-2 + anyecg)")
ecg_language_model = ECG_Language_Model(unfreeze_ecg_model=args.unfreeze_ecg_model, use_lora=args.use_lora)
# load checkpoints 
if args.projection_ckpt is not None:
    print(f"Loading Projection from {args.projection_ckpt}")
    res = ecg_language_model.projection.load_state_dict(torch.load(args.projection_ckpt))
    print(f'Load projection: {res}')
if args.ecg_model_ckpt is not None:
    print(f"Loading ECG Encoder form {args.ecg_model_ckpt}")
    res = ecg_language_model.ecg_model.load_state_dict(torch.load(args.ecg_model_ckpt))
    print(f'Load ecg model: {res}')
if args.lora_ckpt is not None:
    print(f"Loading Lora from {args.lora_ckpt}")
    ecg_language_model.language_model.delete_adapter("default")
    ecg_language_model.language_model.load_adapter(args.lora_ckpt)
    print(f'Load lora: {args.lora_ckpt}')

# applyign training recipe 

if args.unfreeze_ecg_model:
    apply_share_recipe(ecg_language_model,args.use_lora)
else:
    print("Using Base recipe (Frozen Encoer)")
    for param in ecg_language_model.parameters():
        param.requires_grad = False
    for param in ecg_language_model.projection.parameters():
        param.requires_grad = True


device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
ecg_language_model = ecg_language_model.to(device)

# Data
print(f"Loading dataset form {args.data_path}")
dataset = load_dataset('json', data_files=args.data_path)['train']
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Collate_Fn(), num_workers=0)

# Optimizer
named_params = list(ecg_language_model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in named_params if p.requires_grad and not any(nd in n for nd in no_decay)], 
        'weight_decay': args.weight_decay
    },
    {
        'params': [p for n, p in named_params if p.requires_grad and any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0
    }
]
num_train_steps = len(train_loader) * args.num_epochs // args.accumulation_steps
num_warmup_steps = int(num_train_steps * args.warmup)
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

# Training
def save_model(ecg_language_model, steps, output_dir, unfreeze_ecg_model, use_lora):
    output_dir = os.path.join(output_dir, f'step_{steps}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Saving checkpoint to {output_dir}...")
    torch.save(ecg_language_model.projection.state_dict(), os.path.join(output_dir, 'projection.pth'))
    if unfreeze_ecg_model:
        torch.save(ecg_language_model.ecg_model.state_dict(), os.path.join(output_dir, 'ecg_model.pth'))
    if use_lora:
        ecg_language_model.language_model.save_pretrained(output_dir)
# training loop 
steps = 0
print(f"Starting training...")
for epoch in trange(args.num_epochs, desc='Epoch'):
    ecg_language_model.train()
    optimizer.zero_grad()
    for i, (ecgs, messages) in enumerate(tqdm(train_loader, desc='Iteration')):
        loss = ecg_language_model(ecgs, messages)
        loss /= args.accumulation_steps
        loss.backward()
        
        if (i + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(ecg_language_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            steps += 1

            if steps % 10 == 0:
                logger.info(f'Epoch {epoch}, Step {steps}, Loss {loss.item()}')
            if steps % args.save_steps == 0:
                save_model(ecg_language_model, steps, args.output_dir, args.unfreeze_ecg_model, args.use_lora)
save_model(ecg_language_model, steps, args.output_dir, args.unfreeze_ecg_model, args.use_lora)
print("Training Complete.")
