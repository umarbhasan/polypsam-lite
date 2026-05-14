!pip install -q transformers peft monai
!pip install -q protobuf==3.20.3

import torch
print(torch.__version__)

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm.notebook import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import SamModel, SamProcessor
from peft import LoraConfig, get_peft_model
from monai.losses import DiceCELoss
from scipy.stats import ttest_rel

# --- 1. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ABLATION_EPOCHS = 3
FINAL_EPOCHS = 5
SEED = 42

CONFIG = {
    "split_root": "/kaggle/input/kvasirseg",
    "image_root": "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/images",
    "mask_root": "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/masks",
    "model_checkpoint": "facebook/sam-vit-base",
    "lr": 1e-4,
    "batch_size": 1,
    "grad_accum": 4
}

# --- 2. UTILS ---
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f">>> Seed Locked: {seed}")

class KvasirSAMDataset(Dataset):
    def __init__(self, split_root, image_root, mask_root, split_file, processor):
        self.image_root = image_root
        self.mask_root = mask_root
        self.processor = processor
        with open(os.path.join(split_root, split_file), 'r') as f:
            self.file_names = [line.strip() for line in f.readlines() if line.strip()]
    def __len__(self): return len(self.file_names)
    def get_bounding_box(self, mask):
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0: return [0, 0, 1, 1]
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = mask.shape
        noise = np.random.randint(0, 10)
        return [max(0, x_min-noise), max(0, y_min-noise), min(W, x_max+noise), min(H, y_max+noise)]
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = Image.open(os.path.join(self.image_root, file_name + ".jpg")).convert("RGB").resize((256, 256))
        mask = Image.open(os.path.join(self.mask_root, file_name + ".jpg")).convert("L").resize((256, 256), resample=Image.NEAREST)
        image_np = np.array(image)
        mask_np = (np.array(mask) > 128).astype(np.uint8)
        inputs = self.processor(image_np, input_boxes=[[self.get_bounding_box(mask_np)]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask_np).float().unsqueeze(0)
        inputs["original_image"] = image_np
        return inputs

from torch.utils.data import ConcatDataset

print(">>> Initializing Data Splits (70% Train / 10% Val / 20% Test)...")

processor = SamProcessor.from_pretrained(CONFIG["model_checkpoint"])

# 1. Load both original splits to pool all 1000 images
ds_part1 = KvasirSAMDataset(CONFIG["split_root"], CONFIG["image_root"], CONFIG["mask_root"], "train.txt", processor)
ds_part2 = KvasirSAMDataset(CONFIG["split_root"], CONFIG["image_root"], CONFIG["mask_root"], "val.txt", processor)

# 2. Combine them into a single dataset
full_ds = ConcatDataset([ds_part1, ds_part2])
total_images = len(full_ds)

# 3. Calculate exact split sizes
num_train = int(0.70 * total_images)
num_val = int(0.10 * total_images)
num_test = total_images - num_train - num_val # Catches any rounding remainders

# 4. Split using fixed SEED
train_ds, val_ds, test_ds = torch.utils.data.random_split(
    full_ds,
    [num_train, num_val, num_test],
    generator=torch.Generator().manual_seed(SEED)
)

# 5. Initialize Loaders
train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

print(f"Total Images Pooled: {total_images}")
print(f"Data Loaded: {len(train_ds)} Train | {len(val_ds)} Val | {len(test_ds)} Test")

def get_model(rank=None, zero_shot=False):
    model = SamModel.from_pretrained(CONFIG["model_checkpoint"])
    if not zero_shot:
        lora_config = LoraConfig(r=rank, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.05, bias="none")
        model = get_peft_model(model, lora_config)
        for name, param in model.named_parameters():
            if "mask_decoder" in name: param.requires_grad = True
    else:
        # For Zero-Shot, we freeze everything
        for param in model.parameters():
            param.requires_grad = False
    return model.to(DEVICE)

import torch
import pandas as pd
import os

# --- 1. UPDATED FUNCTIONS (Aligned with CONFIG & New Splits) ---

def train_engine(model, train_loader, val_loader, epochs, desc, start_epoch=0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    # Cosine Annealing: Drops LR from 1e-4 down to 1e-6 smoothly
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    best_val_dice = 0.0
    history = []

    for epoch in range(start_epoch, start_epoch + epochs):
        # --- TRAINING PHASE ---
        model.train()
        epoch_loss_sum = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"{desc} | Ep {epoch+1}/{start_epoch+epochs} [Train]")

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_boxes = batch["input_boxes"].to(DEVICE)
            gt_masks = batch["ground_truth_mask"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            loss = loss_fn(outputs.pred_masks.squeeze(1), gt_masks)

            # Using CONFIG for Grad Accumulation
            (loss / CONFIG["grad_accum"]).backward()

            if (step + 1) % CONFIG["grad_accum"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            current_loss = loss.item()
            epoch_loss_sum += current_loss
            num_batches += 1
            pbar.set_postfix({"Loss": f"{current_loss:.4f}"})

        avg_train_loss = epoch_loss_sum / num_batches
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step() # Step the learning rate down

        # --- VALIDATION PHASE (On the 10% Split) ---
        model.eval()
        val_dices = []
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE)
                input_boxes = batch["input_boxes"].to(DEVICE)
                gt = batch["ground_truth_mask"].numpy()[0, 0].astype(bool)

                outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
                pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(bool)

                inter = np.logical_and(pred, gt).sum()
                dice = (2. * inter) / (pred.sum() + gt.sum() + 1e-8)
                val_dices.append(dice)

        avg_val_dice = np.mean(val_dices)
        print(f"--> Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {current_lr:.2e}")

        # --- SAVE BEST MODEL ---
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_checkpoint = f"checkpoint_{desc}_best.pth"
            torch.save(model.state_dict(), best_checkpoint)
            print(f"    🌟 Saved NEW BEST model based on Validation Dice ({best_val_dice:.4f})!")

        # Log History
        history.append({"Epoch": epoch + 1, "Train_Loss": avg_train_loss, "Val_Dice": avg_val_dice, "LR": current_lr})
        pd.DataFrame(history).to_csv(f"training_log_{desc.replace(' ', '_')}.csv", index=False)

    return model

# Updated to accept a specific loader (defaulting to test_loader)
def evaluate_metrics(model, loader, desc="Eval"):
    model.eval()
    dice_list = []
    print(f"Evaluating {desc}...")
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_boxes = batch["input_boxes"].to(DEVICE)
            gt = batch["ground_truth_mask"].numpy()[0, 0].astype(bool)
            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(bool)
            dice_list.append((2. * np.logical_and(pred, gt).sum()) / (pred.sum() + gt.sum() + 1e-8))
    return dice_list

# ==============================================================================
# --- 2. THE MAIN EXECUTION ---
# ==============================================================================

WINNER_RANK = 4

# --- A. Re-calculate Zero-Shot Baseline on TEST SET ---
print("\n>>> STEP 1: Re-calculating Zero-Shot Baseline on Final Test Set")
set_seed(SEED)
model_base = get_model(zero_shot=True)
# Explicitly pass test_loader here
baseline_dices = evaluate_metrics(model_base, loader=test_loader, desc="Zero-Shot Baseline")
del model_base
torch.cuda.empty_cache()

# --- B. Train with Validation Tracking (10 Epochs) ---
print(f"\n>>> STEP 2: Starting Optimized Production Run (Rank {WINNER_RANK}, 10 Epochs)")
set_seed(SEED)
model = get_model(rank=WINNER_RANK)

# Pass both train_loader and val_loader
model = train_engine(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    desc="Final_Model_Optimized"
)

# --- C. Final Evaluation on UNSEEN TEST SET ---
print("\n>>> STEP 3: Generating Final Artifacts on Unseen Test Set...")
# CRITICAL: Load the best model, not just the last epoch!
model.load_state_dict(torch.load("checkpoint_Final_Model_Optimized_best.pth", map_location=DEVICE))
model.eval()

poly_dices, iou_scores, precisions, recalls = [], [], [], []

with torch.no_grad():
    for i, batch in enumerate(test_loader): # Use test_loader here!
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_boxes = batch["input_boxes"].to(DEVICE)
        gt = batch["ground_truth_mask"].numpy()[0, 0].astype(bool)

        outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
        pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(bool)

        # Metrics
        inter = np.logical_and(pred, gt).sum()
        dice = (2. * inter) / (pred.sum() + gt.sum() + 1e-8)
        poly_dices.append(dice)
        iou_scores.append(inter / (np.logical_or(pred, gt).sum() + 1e-8))
        precisions.append(inter / (pred.sum() + 1e-8))
        recalls.append(inter / (gt.sum() + 1e-8))

        # Save Images (Changed prefix to not overwrite old images)
        if i < 3:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 3, 1); plt.imshow(batch["original_image"][0]); plt.axis('off'); plt.title("Input")
            plt.subplot(1, 3, 2); plt.imshow(gt, cmap='gray'); plt.axis('off'); plt.title("Ground Truth")
            plt.subplot(1, 3, 3); plt.imshow(pred, cmap='gray'); plt.axis('off'); plt.title("PolySAM-Lite (Optimized)")
            plt.tight_layout()
            plt.savefig(f"Final_Figure_Optimized_{i}.pdf", bbox_inches='tight', dpi=300)
            plt.close()

# T-TEST vs BASELINE
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(poly_dices, baseline_dices)
print(f"\n>>> Paired T-Test Results")
print(f"PolySAM (Optimized) Mean: {np.mean(poly_dices):.4f}")
print(f"Baseline Mean: {np.mean(baseline_dices):.4f}")
print(f"P-Value: {p_value:.4e}")

# SAVE STATS
final_metrics = {
    "Dice": np.mean(poly_dices),
    "IoU": np.mean(iou_scores),
    "Precision": np.mean(precisions),
    "Recall": np.mean(recalls),
    "P_Value": p_value
}
pd.DataFrame([final_metrics]).to_csv("results_table_Optimized.csv", index=False)
pd.DataFrame({"Baseline": baseline_dices, "PolySAM": poly_dices}).to_csv("raw_dices_Optimized.csv", index=False)

# BOX PLOT
plt.figure(figsize=(6, 5), dpi=300)
plt.boxplot([baseline_dices, poly_dices], labels=['Zero-Shot SAM', 'PolySAM-Lite (Optimized)'], patch_artist=True)
plt.title(f'Significance Analysis (p={p_value:.2e})')
plt.ylabel('Dice Score')
plt.grid(True, alpha=0.3)
plt.savefig("fig_boxplot_stats_Optimized.pdf")

print(">>> ALL DONE.")

# ==============================================================================
# --- 2. THE MAIN EXECUTION ---
# ==============================================================================

WINNER_RANK = 4

# --- A. Re-calculate Zero-Shot Baseline on TEST SET ---
print("\n>>> STEP 1: Re-calculating Zero-Shot Baseline on Final Test Set")
set_seed(SEED)
model_base = get_model(zero_shot=True)
# Explicitly pass test_loader here
baseline_dices = evaluate_metrics(model_base, loader=test_loader, desc="Zero-Shot Baseline")
del model_base
torch.cuda.empty_cache()

set_seed(SEED)
# --- 1. THE SMART RESUME ENGINE ---
def train_engine_resume(model, train_loader, val_loader, remaining_epochs, desc, start_epoch, previous_best_dice, total_epochs=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    # 🌟 CRITICAL FIX: Fast-forward the scheduler so the LR doesn't spike
    for _ in range(start_epoch):
        scheduler.step()

    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # 🌟 CRITICAL FIX: Tell the engine what the score to beat is
    best_val_dice = previous_best_dice
    history = []

    print(f"Resuming with LR: {scheduler.get_last_lr()[0]:.2e} and Target to Beat: {best_val_dice:.4f}")

    for epoch in range(start_epoch, start_epoch + remaining_epochs):
        model.train()
        epoch_loss_sum = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"{desc} | Ep {epoch+1}/{total_epochs} [Train]")

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_boxes = batch["input_boxes"].to(DEVICE)
            gt_masks = batch["ground_truth_mask"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            loss = loss_fn(outputs.pred_masks.squeeze(1), gt_masks)

            (loss / CONFIG["grad_accum"]).backward()

            if (step + 1) % CONFIG["grad_accum"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            current_loss = loss.item()
            epoch_loss_sum += current_loss
            num_batches += 1
            pbar.set_postfix({"Loss": f"{current_loss:.4f}"})

        avg_train_loss = epoch_loss_sum / num_batches
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # --- VALIDATION ---
        model.eval()
        val_dices = []
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE)
                input_boxes = batch["input_boxes"].to(DEVICE)
                gt = batch["ground_truth_mask"].numpy()[0, 0].astype(bool)

                outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
                pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(bool)

                inter = np.logical_and(pred, gt).sum()
                dice = (2. * inter) / (pred.sum() + gt.sum() + 1e-8)
                val_dices.append(dice)

        avg_val_dice = np.mean(val_dices)
        print(f"--> Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {current_lr:.2e}")

        # --- SAVE ONLY IF IT BEATS 95.886% ---
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            # Save back to the ORIGINAL file name so we maintain one master "best" file
            best_checkpoint = f"checkpoint_Final_Model_Optimized_best.pth"
            torch.save(model.state_dict(), best_checkpoint)
            print(f"    🌟 Saved NEW BEST model based on Validation Dice ({best_val_dice:.4f})!")

        history.append({"Epoch": epoch + 1, "Train_Loss": avg_train_loss, "Val_Dice": avg_val_dice, "LR": current_lr})
        pd.DataFrame(history).to_csv(f"training_log_Resumed_Part2.csv", index=False)

    return model

# ==============================================================================
# --- 2. EXECUTE THE SMART RESUME ---
# ==============================================================================

WINNER_RANK = 4
LAST_COMPLETED_EPOCH = 6
PREVIOUS_BEST_DICE = 0.9588630739744086
REMAINING_EPOCHS = 10 - LAST_COMPLETED_EPOCH

print(f"\n>>> Loading the true best weights (Epoch {LAST_COMPLETED_EPOCH})...")
model = get_model(rank=WINNER_RANK)

# Load best weights
original_best_weights = "checkpoint_Final_Model_Optimized_best.pth"
model.load_state_dict(torch.load(original_best_weights, map_location=DEVICE))
print("✅ Weights loaded successfully!")

# Run the remaining 4 epochs
model = train_engine_resume(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    remaining_epochs=REMAINING_EPOCHS,
    desc="Final_Model",
    start_epoch=LAST_COMPLETED_EPOCH,
    previous_best_dice=PREVIOUS_BEST_DICE,
    total_epochs=10
)

print(">>> Resume Training Complete. We can now run the Final Evaluation block!")

set_seed(SEED)
# --- C. Final Evaluation on UNSEEN TEST SET ---
print("\n>>> STEP 3: Generating Final Artifacts on Unseen Test Set...")
# CRITICAL: Load the best model, not just the last epoch!
model.load_state_dict(torch.load("checkpoint_Final_Model_Optimized_best.pth", map_location=DEVICE))
model.eval()

poly_dices, iou_scores, precisions, recalls = [], [], [], []

with torch.no_grad():
    for i, batch in enumerate(test_loader): # Use test_loader here!
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_boxes = batch["input_boxes"].to(DEVICE)
        gt = batch["ground_truth_mask"].numpy()[0, 0].astype(bool)

        outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
        pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(bool)

        # Metrics
        inter = np.logical_and(pred, gt).sum()
        dice = (2. * inter) / (pred.sum() + gt.sum() + 1e-8)
        poly_dices.append(dice)
        iou_scores.append(inter / (np.logical_or(pred, gt).sum() + 1e-8))
        precisions.append(inter / (pred.sum() + 1e-8))
        recalls.append(inter / (gt.sum() + 1e-8))

        # Save Images (Changed prefix to not overwrite old images)
        if i < 3:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 3, 1); plt.imshow(batch["original_image"][0]); plt.axis('off'); plt.title("Input")
            plt.subplot(1, 3, 2); plt.imshow(gt, cmap='gray'); plt.axis('off'); plt.title("Ground Truth")
            plt.subplot(1, 3, 3); plt.imshow(pred, cmap='gray'); plt.axis('off'); plt.title("PolypSAM-Lite")
            plt.tight_layout()
            plt.savefig(f"Final_Figure_Optimized_{i}.pdf", bbox_inches='tight', dpi=300)
            plt.close()

# T-TEST vs BASELINE
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(poly_dices, baseline_dices)
print(f"\n>>> Paired T-Test Results")
print(f"PolySAM (Optimized) Mean: {np.mean(poly_dices):.4f}")
print(f"Baseline Mean: {np.mean(baseline_dices):.4f}")
print(f"P-Value: {p_value:.4e}")

# SAVE STATS
final_metrics = {
    "Dice": np.mean(poly_dices),
    "IoU": np.mean(iou_scores),
    "Precision": np.mean(precisions),
    "Recall": np.mean(recalls),
    "P_Value": p_value
}
pd.DataFrame([final_metrics]).to_csv("results_table_Optimized.csv", index=False)
pd.DataFrame({"Baseline": baseline_dices, "PolySAM": poly_dices}).to_csv("raw_dices_Optimized.csv", index=False)

# BOX PLOT
plt.figure(figsize=(6, 5), dpi=300)
plt.boxplot([baseline_dices, poly_dices], labels=['Zero-Shot SAM', 'PolypSAM-Lite'], patch_artist=True)
plt.title(f'Significance Analysis (p={p_value:.2e})')
plt.ylabel('Dice Score')
plt.grid(True, alpha=0.3)
plt.savefig("fig_boxplot_stats_Optimized.pdf")

print(">>> ALL DONE.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(">>> Stitching Training Logs together...")

# 1. Load the two CSV files
try:
    log_part1 = pd.read_csv("training_log_Final_Model_Optimized.csv")
    log_part2 = pd.read_csv("training_log_Resumed_Part2.csv")
except FileNotFoundError as e:
    print(f"❌ Error finding files: {e}")
    print("Please check the exact filenames in the directory.")

# 2. Filter Part 1 to only include Epochs 1 through 6
# (This safely removes any partial/failed data from Epoch 7 before the crash)
log_part1_clean = log_part1[log_part1['Epoch'] <= 6]

# 3. Combine them into a continuous 1-10 sequence
merged_log = pd.concat([log_part1_clean, log_part2], ignore_index=True)

# 4. Save the pristine master log
merged_log.to_csv("training_log_Final_Model_10ep_Master.csv", index=False)
print("✅ Successfully created 'training_log_Final_Model_10ep_Master.csv'")

# ==========================================
# 5. Generate the Publication Figure
# ==========================================
print(">>> Generating Dual-Axis Learning Curve...")

fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

# X-Axis settings
ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_xticks(range(1, 11))
ax1.grid(True, alpha=0.3)

# Primary Y-Axis (Training Loss - Blue)
color1 = 'tab:blue'
ax1.set_ylabel('Training Loss', color=color1, fontweight='bold')
line1, = ax1.plot(merged_log['Epoch'], merged_log['Train_Loss'],
                  marker='o', color=color1, linewidth=2, label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color1)

# Secondary Y-Axis (Validation Dice - Orange)
ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.set_ylabel('Validation Dice Score', color=color2, fontweight='bold')
line2, = ax2.plot(merged_log['Epoch'], merged_log['Val_Dice'],
                  marker='s', color=color2, linewidth=2, linestyle='--', label='Val Dice')
ax2.tick_params(axis='y', labelcolor=color2)

# Combine legends from both axes
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title('PolypSAM-Lite: Training Convergence', fontweight='bold', fontsize=12)
fig.tight_layout()

# Save the figure
plt.savefig('fig_learning_curve_10ep.pdf', bbox_inches='tight')
plt.show()

print("Saved 'fig_learning_curve_10ep.pdf' for the manuscript.")

import time

set_seed(SEED)
# Update Target to reflect the new pipeline
TARGET_EPOCHS = 10

print(">>> Starting Benchmark (1 Epoch)...")
# Using train_loader
speed_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
model = get_model(rank=4)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

torch.cuda.synchronize()
start_time = time.time()

for step, batch in enumerate(tqdm(speed_loader, desc="Timing Epoch 1")):
    pixel_values = batch["pixel_values"].to(DEVICE)
    input_boxes = batch["input_boxes"].to(DEVICE)
    gt_masks = batch["ground_truth_mask"].to(DEVICE)

    outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
    loss = loss_fn(outputs.pred_masks.squeeze(1), gt_masks)

    (loss / CONFIG["grad_accum"]).backward()
    if (step + 1) % CONFIG["grad_accum"] == 0:
        optimizer.step()
        optimizer.zero_grad()

torch.cuda.synchronize()
end_time = time.time()

duration_1_epoch = end_time - start_time
estimated_total = duration_1_epoch * TARGET_EPOCHS

print(f"\n" + "="*40)
print(f"RESULTS")
print(f"="*40)
print(f"Time for 1 Epoch:   {duration_1_epoch/60:.2f} minutes")
print(f"Est. for 10 Epochs: {estimated_total/60:.2f} minutes")
print(f"="*40)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

set_seed(SEED)
# LOAD THE NEW OPTIMIZED DICE SCORES
df = pd.read_csv('raw_dices_Optimized.csv')

# Calculate statistics
mu_baseline = df['Baseline'].mean()
sigma_baseline = df['Baseline'].std()

mu_polysam = df['PolySAM'].mean()
sigma_polysam = df['PolySAM'].std()

# Dynamic Range: mu +/- 4 sigma to capture 99.9% of the curve
min_x = min(mu_baseline - 4*sigma_baseline, mu_polysam - 4*sigma_polysam)
max_x = max(mu_baseline + 4*sigma_baseline, mu_polysam + 4*sigma_polysam)

# Generate points
x = np.linspace(min_x, max_x, 1000)

# Calculate PDF
pdf_baseline = stats.norm.pdf(x, mu_baseline, sigma_baseline)
pdf_polysam = stats.norm.pdf(x, mu_polysam, sigma_polysam)

# Plotting
plt.figure(figsize=(8, 6), dpi=300)

# Plot Baseline (Gray Dashed)
plt.plot(x, pdf_baseline, color='gray', linestyle='--', linewidth=2,
         label=f'Zero-Shot SAM ($\mu={mu_baseline:.4f}$, $\sigma={sigma_baseline:.4f}$)')
plt.fill_between(x, pdf_baseline, color='gray', alpha=0.1)

# Plot PolySAM (Orange Solid)
plt.plot(x, pdf_polysam, color='darkorange', linewidth=2,
         label=f'PolypSAM-Lite ($\mu={mu_polysam:.4f}$, $\sigma={sigma_polysam:.4f}$)')
plt.fill_between(x, pdf_polysam, color='darkorange', alpha=0.1)

plt.title('Distribution of Dice Scores (Normal Fit)')
plt.xlabel('Dice Similarity Coefficient (DSC)')
plt.ylabel('Probability Density')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xlim(min_x, max_x)

plt.savefig('fig_bell_curve_stats_Optimized.pdf', bbox_inches='tight')
print("Figure saved successfully.")

from sklearn.manifold import TSNE

set_seed(SEED)
# 1. SETUP MODEL & LOAD BEST WEIGHTS
print(f"\n>>> Loading Fine-Tuned Model (Optimized 10 Ep)...")
ft_model = get_model(rank=4, zero_shot=False)
ft_model.load_state_dict(torch.load("checkpoint_Final_Model_Optimized_best.pth", map_location=DEVICE))

def get_embeddings(target_model, loader):
    target_model.eval()
    embeddings = []
    print("Extracting features (running inference)...")
    with torch.no_grad():
        for batch in tqdm(loader):
            pixel_values = batch["pixel_values"].to(DEVICE)
            outputs = target_model.vision_encoder(pixel_values, output_hidden_states=True)
            feats = outputs.last_hidden_state
            pooled = feats.mean(dim=[2, 3]).cpu().numpy()
            embeddings.append(pooled)
    return np.concatenate(embeddings, axis=0)

# 2. EXTRACT EMBEDDINGS (Using test_loader!)
print("\n>>> Extracting PolySAM Embeddings on Test Set...")
ft_embeddings = get_embeddings(ft_model, test_loader)
del ft_model
torch.cuda.empty_cache()

print("\n>>> Extracting Zero-Shot Embeddings on Test Set...")
baseline_model = get_model(zero_shot=True)
baseline_embeddings = get_embeddings(baseline_model, test_loader)

# 3. COMPUTE AND PLOT t-SNE
print("\n>>> Computing t-SNE...")
all_feats = np.concatenate([baseline_embeddings, ft_embeddings], axis=0)
labels = np.array([0] * len(baseline_embeddings) + [1] * len(ft_embeddings))

tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init='pca', learning_rate='auto')
vis_dims = tsne.fit_transform(all_feats)

base_vis = vis_dims[labels == 0]
ft_vis = vis_dims[labels == 1]

plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(base_vis[:, 0], base_vis[:, 1], c='gray', alpha=0.5, label='Zero-Shot Baseline')
plt.scatter(ft_vis[:, 0], ft_vis[:, 1], c='red', alpha=0.6, label='PolypSAM-Lite')

plt.title("Effect of LoRA Fine-Tuning on Feature Space")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.grid(True, alpha=0.2)

plt.savefig("tsne_comparison_Optimized.pdf", bbox_inches='tight')
print(">>> SUCCESS: Plot saved.")

from sklearn.metrics import roc_curve, auc

set_seed(SEED)

print("Running Comparative ROC Analysis on Test Set...")
# Ensure we load the best model first
model = get_model(rank=4)
model.load_state_dict(torch.load("checkpoint_Final_Model_Optimized_best.pth", map_location=DEVICE))
model.eval()

all_gts = []
probs_polysam = []
probs_baseline = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="ROC Inference"): # USE test_loader
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_boxes = batch["input_boxes"].to(DEVICE)
        gt = batch["ground_truth_mask"].numpy().flatten().astype(int)
        all_gts.extend(gt)

        # PolySAM-Lite (Adapter ON)
        outputs_poly = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
        prob_poly = torch.sigmoid(outputs_poly.pred_masks[0, 0, 0]).cpu().numpy().flatten()
        probs_polysam.extend(prob_poly)

        # Zero-Shot Baseline (Adapter OFF)
        with model.disable_adapter():
            outputs_base = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            prob_base = torch.sigmoid(outputs_base.pred_masks[0, 0, 0]).cpu().numpy().flatten()
            probs_baseline.extend(prob_base)

# Calculate AUCs
fpr_poly, tpr_poly, _ = roc_curve(all_gts, probs_polysam)
auc_poly = auc(fpr_poly, tpr_poly)

fpr_base, tpr_base, _ = roc_curve(all_gts, probs_baseline)
auc_base = auc(fpr_base, tpr_base)

# Plot
plt.figure(figsize=(6, 6), dpi=300)
plt.plot(fpr_poly, tpr_poly, color='darkorange', lw=2, label=f'PolypSAM-Lite (AUC = {auc_poly:.4f})')
plt.plot(fpr_base, tpr_base, color='navy', lw=2, linestyle=':', label=f'Zero-Shot SAM (AUC = {auc_base:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Chance')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

plt.savefig('fig_roc_combined_Optimized.pdf', bbox_inches='tight')
plt.show()

print(f"PolySAM AUC: {auc_poly:.4f} | Baseline AUC: {auc_base:.4f}")

print("Scanning Test Set for the hardest case (Failure Mode)...")

set_seed(SEED)
model = get_model(rank=4)
model.load_state_dict(torch.load("checkpoint_Final_Model_Optimized_best.pth", map_location=DEVICE))
model.eval()

worst_dice = 1.0
worst_batch, worst_pred, worst_gt = None, None, None

with torch.no_grad():
    for batch in test_loader: # USE test_loader
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_boxes = batch["input_boxes"].to(DEVICE)
        gt = batch["ground_truth_mask"].numpy()[0, 0].astype(bool)

        outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
        pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(bool)

        intersection = np.logical_and(pred, gt).sum()
        dice = (2. * intersection) / (pred.sum() + gt.sum() + 1e-8)

        if dice < worst_dice and gt.sum() > 0:
            worst_dice = dice
            worst_batch = batch
            worst_pred = pred
            worst_gt = gt

print(f"Found Hardest Case! Dice Score: {worst_dice:.4f}")

if worst_batch is not None:
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(worst_batch["original_image"][0])
    plt.axis('off')
    plt.title(f"Input (Dice: {worst_dice:.2f})")

    plt.subplot(1, 3, 2)
    plt.imshow(worst_gt, cmap='gray')
    plt.axis('off')
    plt.title("Ground Truth")

    plt.subplot(1, 3, 3)
    plt.imshow(worst_pred, cmap='gray')
    plt.axis('off')
    plt.title("PolypSAM-Lite")

    plt.tight_layout()
    plt.savefig("Final_Figure_FailureCase_Optimized.pdf", bbox_inches='tight', dpi=300)
    plt.show()

import torch
import time
import numpy as np
from transformers import SamModel
from peft import LoraConfig, get_peft_model

set_seed(SEED)
# 1. Setup Architecture
print("Initializing PolySAM-Lite...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-base")
lora_config = LoraConfig(r=4, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, lora_config)
for name, param in model.named_parameters():
    if "mask_decoder" in name: param.requires_grad = True
model.to(DEVICE)

# 2. Load Weights (Optional for speed test, but good practice)
# We skip loading weights here just to get the speed (architecture determines speed)
model.eval()

# 3. CORRECTED DUMMY INPUT (Must be 1024x1024 for ViT-Base)
# The processor resizes 256 -> 1024 internally, so the model sees 1024.
dummy_input = torch.randn(1, 3, 1024, 1024).to(DEVICE)
dummy_boxes = torch.tensor([[[0, 0, 100, 100]]]).float().to(DEVICE)

# 4. Benchmark
print("\nWarming up GPU...")
for _ in range(10): # Reduced warmup
    with torch.no_grad():
        _ = model(pixel_values=dummy_input, input_boxes=dummy_boxes, multimask_output=False)

iterations = 100 # Reduced iterations to save time (100 is enough for avg)
print(f"Benchmarking {iterations} frames (1024x1024)...")
start_time = time.time()

with torch.no_grad():
    for _ in range(iterations):
        _ = model(pixel_values=dummy_input, input_boxes=dummy_boxes, multimask_output=False)
        torch.cuda.synchronize()

end_time = time.time()
total_time = end_time - start_time
fps = iterations / total_time
latency = (total_time / iterations) * 1000

print(f"\nRESULTS:")
print(f"Latency: {latency:.2f} ms")
print(f"Throughput: {fps:.2f} FPS")

import torch
import time
import numpy as np
from transformers import SamModel
from peft import LoraConfig, get_peft_model

set_seed(SEED)
# 1. Setup Architecture
print("Initializing PolySAM-Lite...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-base")
lora_config = LoraConfig(r=1, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, lora_config)
for name, param in model.named_parameters():
    if "mask_decoder" in name: param.requires_grad = True
model.to(DEVICE)

# 2. Load Weights (Optional for speed test, but good practice)
# We skip loading weights here just to get the speed (architecture determines speed)
model.eval()

# 3. CORRECTED DUMMY INPUT (Must be 1024x1024 for ViT-Base)
# The processor resizes 256 -> 1024 internally, so the model sees 1024.
dummy_input = torch.randn(1, 3, 1024, 1024).to(DEVICE)
dummy_boxes = torch.tensor([[[0, 0, 100, 100]]]).float().to(DEVICE)

# 4. Benchmark
print("\nWarming up GPU...")
for _ in range(10): # Reduced warmup
    with torch.no_grad():
        _ = model(pixel_values=dummy_input, input_boxes=dummy_boxes, multimask_output=False)

iterations = 100 # Reduced iterations to save time (100 is enough for avg)
print(f"Benchmarking {iterations} frames (1024x1024)...")
start_time = time.time()

with torch.no_grad():
    for _ in range(iterations):
        _ = model(pixel_values=dummy_input, input_boxes=dummy_boxes, multimask_output=False)
        torch.cuda.synchronize()

end_time = time.time()
total_time = end_time - start_time
fps = iterations / total_time
latency = (total_time / iterations) * 1000

print(f"\nRESULTS:")
print(f"Latency: {latency:.2f} ms")
print(f"Throughput: {fps:.2f} FPS")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import SamModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
from monai.losses import DiceCELoss

# ==========================================================
# 0. ABLATION-SPECIFIC TRAINING ENGINE (NO SCHEDULER)
# ==========================================================
def train_engine_ablation(model, train_loader, val_loader, epochs, desc):
    """A simplified training loop with a constant Learning Rate."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    best_val_dice = 0.0
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"{desc} | Ep {epoch+1}/{epochs} [Train]")

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_boxes = batch["input_boxes"].to(DEVICE)
            gt_masks = batch["ground_truth_mask"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            loss = loss_fn(outputs.pred_masks.squeeze(1), gt_masks)

            (loss / CONFIG["grad_accum"]).backward()

            if (step + 1) % CONFIG["grad_accum"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            current_loss = loss.item()
            epoch_loss_sum += current_loss
            num_batches += 1
            pbar.set_postfix({"Loss": f"{current_loss:.4f}"})

        avg_train_loss = epoch_loss_sum / num_batches

        # Validation Phase
        model.eval()
        val_dices = []
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE)
                input_boxes = batch["input_boxes"].to(DEVICE)
                gt = batch["ground_truth_mask"].numpy()[0, 0].astype(bool)

                outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
                pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(bool)

                inter = np.logical_and(pred, gt).sum()
                dice = (2. * inter) / (pred.sum() + gt.sum() + 1e-8)
                val_dices.append(dice)

        avg_val_dice = np.mean(val_dices)
        print(f"--> Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {CONFIG['lr']:.2e} (Constant)")

        # Save Best Model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), f"checkpoint_{desc}_best.pth")
            print(f"    🌟 Saved NEW BEST model based on Validation Dice!")

        history.append({"Epoch": epoch + 1, "Train_Loss": avg_train_loss, "Val_Dice": avg_val_dice})

    return model


# ==========================================================
# 1. SETUP & CONFIGURATION
# ==========================================================
ABLATION_EPOCHS = 3
WINNER_RANK = 4

def get_ablation_model(config_type):
    model = SamModel.from_pretrained(CONFIG["model_checkpoint"])
    if config_type == "lora_all_linear":
        lora_config = LoraConfig(r=4, lora_alpha=32, target_modules="all-linear", lora_dropout=0.05, bias="none")
        model = get_peft_model(model, lora_config)
        for name, param in model.named_parameters():
            if "mask_decoder" in name: param.requires_grad = True
    return model.to(DEVICE)


# ==========================================================
# 2. EXECUTE ABLATIONS
# ==========================================================

# --- A. Rank Ablation (1 vs 4) ---
print(f"\n>>> STEP 1: Rank Ablation (1 vs 4) for {ABLATION_EPOCHS} Epochs")
ranks = [1, 4]
rank_results = []

for r in ranks:
    set_seed(SEED)
    model = get_model(rank=r)

    model = train_engine_ablation(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=ABLATION_EPOCHS,
        desc=f"Ablation_Rank_{r}"
    )

    model.load_state_dict(torch.load(f"checkpoint_Ablation_Rank_{r}_best.pth", map_location=DEVICE))
    dices = evaluate_metrics(model, loader=test_loader, desc=f"Rank {r}")
    mean_dice = np.mean(dices)

    print(f"Rank {r} Mean Test Dice: {mean_dice:.4f}")
    rank_results.append({"Rank": r, "Dice": mean_dice})

    del model
    torch.cuda.empty_cache()

pd.DataFrame(rank_results).to_csv("ablation_rank.csv", index=False)


# --- B. Data Efficiency (50%) ---
print(f"\n>>> STEP 2: Data Efficiency (50%) for {ABLATION_EPOCHS} Epochs")
set_seed(SEED)
generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(len(train_ds), generator=generator)[:int(len(train_ds)*0.50)]
subset_ds = Subset(train_ds, indices.tolist())
loader_50 = DataLoader(subset_ds, batch_size=CONFIG["batch_size"], shuffle=True)

model = get_model(rank=WINNER_RANK)

model = train_engine_ablation(
    model,
    train_loader=loader_50,
    val_loader=val_loader,
    epochs=ABLATION_EPOCHS,
    desc="Ablation_Data_0.5"
)

model.load_state_dict(torch.load("checkpoint_Ablation_Data_0.5_best.pth", map_location=DEVICE))
dices_50 = evaluate_metrics(model, loader=test_loader, desc="Data 0.5")
mean_dice_50 = np.mean(dices_50)

print(f"Data 50% Mean Test Dice: {mean_dice_50:.4f}")
pd.DataFrame([{"Fraction": 0.5, "Dice": mean_dice_50}]).to_csv("ablation_data_50.csv", index=False)

del model
torch.cuda.empty_cache()


# --- C. LoRA Placement (All-Linear) ---
print("\n>>> STEP 3: LoRA Placement (All-Linear)")
set_seed(SEED)
model_all = get_ablation_model("lora_all_linear")

model_all = train_engine_ablation(
    model_all,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=ABLATION_EPOCHS,
    desc="Ablation_LoRA_All"
)

model_all.load_state_dict(torch.load("checkpoint_Ablation_LoRA_All_best.pth", map_location=DEVICE))
dices_all = evaluate_metrics(model_all, loader=test_loader, desc="LoRA All-Linear")
mean_dice_all = np.mean(dices_all)

print(f"LoRA All-Linear Mean Test Dice: {mean_dice_all:.4f}")

del model_all
torch.cuda.empty_cache()

print("\n>>> ALL ABLATIONS COMPLETE.")
print(f"Rank 1: {rank_results[0]['Dice']:.4f}")
print(f"Rank 4: {rank_results[1]['Dice']:.4f}")
print(f"Data 50%: {mean_dice_50:.4f}")
print(f"LoRA All-Linear: {mean_dice_all:.4f}")

# ==========================================================
# RE-RUN: RANK ABLATION (5 Epochs)
# ==========================================================
print(f"\n>>> Fixing Rank Ablation: Running 1 vs 4 for 5 Epochs")

# We specifically set this to 5 just for this test
RANK_ABLATION_EPOCHS = 5
ranks = [1, 4]
rank_results = []

for r in ranks:
    set_seed(SEED)
    model = get_model(rank=r)

    # Using the constant LR ablation engine
    model = train_engine_ablation(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=RANK_ABLATION_EPOCHS,
        desc=f"Ablation_Rank_{r}_5ep"
    )

    # Load best weights
    model.load_state_dict(torch.load(f"checkpoint_Ablation_Rank_{r}_5ep_best.pth", map_location=DEVICE))

    # Evaluate on the unseen test set
    dices = evaluate_metrics(model, loader=test_loader, desc=f"Rank {r} (5 Ep)")
    mean_dice = np.mean(dices)

    print(f"Rank {r} Mean Test Dice: {mean_dice:.4f}")
    rank_results.append({"Rank": r, "Dice": mean_dice})

    del model
    torch.cuda.empty_cache()

# Save the updated results
pd.DataFrame(rank_results).to_csv("ablation_rank_5ep.csv", index=False)

print("\n>>> RANK ABLATION (5 EPOCHS) COMPLETE.")
print(f"Rank 1: {rank_results[0]['Dice']:.4f}")
print(f"Rank 4: {rank_results[1]['Dice']:.4f}")

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import SamModel, SamProcessor
from peft import LoraConfig, get_peft_model
from tqdm.notebook import tqdm

set_seed(SEED)
# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_CHECKPOINT = "checkpoint_Final_Model_Optimized_best.pth"
MODEL_CHECKPOINT = "facebook/sam-vit-base"
RANK = 4

# External Dataset Paths (As provided)
CVC_PATHS = {
    "images": "/kaggle/input/datasets/balraj98/cvcclinicdb/PNG/Original",
    "masks": "/kaggle/input/datasets/balraj98/cvcclinicdb/PNG/Ground Truth"
}

ETIS_PATHS = {
    "images": "/kaggle/input/datasets/nguyenvoquocduong/etis-laribpolypdb/images",
    "masks": "/kaggle/input/datasets/nguyenvoquocduong/etis-laribpolypdb/masks"
}

# ==========================================
# 2. ROBUST DATASET CLASS
# ==========================================
class ExternalPolypDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor

        # Grab all valid files and sort them to ensure matching pairs
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(valid_exts)])

        if len(self.image_files) != len(self.mask_files):
            print(f"⚠️ WARNING: Mismatch in file counts! {len(self.image_files)} images vs {len(self.mask_files)} masks.")
            # Fallback: strict name matching if counts differ
            self.image_files = [f for f in self.image_files if f in self.mask_files]
            self.mask_files = self.image_files

    def __len__(self):
        return len(self.image_files)

    def get_bounding_box(self, mask):
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0: return [0, 0, 256, 256] # Empty mask fallback
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = mask.shape
        return [max(0, x_min), max(0, y_min), min(W, x_max), min(H, y_max)]

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Load and resize
        image = Image.open(img_path).convert("RGB").resize((256, 256))
        mask = Image.open(mask_path).convert("L").resize((256, 256), resample=Image.NEAREST)

        image_np = np.array(image)
        mask_np = (np.array(mask) > 128).astype(np.uint8)

        # Process for SAM
        inputs = self.processor(image_np, input_boxes=[[self.get_bounding_box(mask_np)]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        inputs["ground_truth_mask"] = torch.tensor(mask_np).float().unsqueeze(0)
        inputs["original_image"] = image_np
        inputs["filename"] = self.image_files[idx]

        return inputs

# ==========================================
# 3. EVALUATION FUNCTION
# ==========================================
def evaluate_external_dataset(model, dataset_name, image_dir, mask_dir, processor):
    print(f"\n" + "="*50)
    print(f"Evaluating Cross-Domain Generalization on: {dataset_name}")
    print("="*50)

    # Check if paths exist in the Kaggle environment
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"❌ ERROR: Paths for {dataset_name} not found. Please verify the dataset is added to the notebook.")
        return None

    dataset = ExternalPolypDataset(image_dir, mask_dir, processor)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Loaded {len(dataset)} samples from {dataset_name}.")

    model.eval()
    dices, ious, precisions, recalls = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Testing {dataset_name}")):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_boxes = batch["input_boxes"].to(DEVICE)
            gt = batch["ground_truth_mask"].numpy()[0, 0].astype(bool)

            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(bool)

            # Metrics calculation
            inter = np.logical_and(pred, gt).sum()
            dice = (2. * inter) / (pred.sum() + gt.sum() + 1e-8)
            iou = inter / (np.logical_or(pred, gt).sum() + 1e-8)
            precision = inter / (pred.sum() + 1e-8)
            recall = inter / (gt.sum() + 1e-8)

            dices.append(dice)
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)

            # --- Generate Qualitative Visualizations (First 3 Images) ---
            if i < 3:
                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.imshow(batch["original_image"][0])
                plt.axis('off')
                plt.title(f"Input ({batch['filename'][0]})")

                plt.subplot(1, 3, 2)
                plt.imshow(gt, cmap='gray')
                plt.axis('off')
                plt.title("Ground Truth")

                plt.subplot(1, 3, 3)
                plt.imshow(pred, cmap='gray')
                plt.axis('off')
                plt.title(f"PolypSAM-Lite (DSC: {dice:.4f})")

                plt.tight_layout()
                fig_name = f"External_{dataset_name}_Fig_{i+1}.pdf"
                plt.savefig(fig_name, bbox_inches='tight', dpi=300)
                plt.close()
                print(f"📸 Saved visualization: {fig_name}")

    # Print Results
    print(f"\n>>> FINAL METRICS: {dataset_name}")
    print(f"Dice Score: {np.mean(dices):.4f}")
    print(f"IoU Score:  {np.mean(ious):.4f}")
    print(f"Precision:  {np.mean(precisions):.4f}")
    print(f"Recall:     {np.mean(recalls):.4f}")

    return {"Dataset": dataset_name, "Dice": np.mean(dices), "IoU": np.mean(ious), "Precision": np.mean(precisions), "Recall": np.mean(recalls)}

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
print(">>> Rebuilding PolySAM-Lite Architecture...")
processor = SamProcessor.from_pretrained(MODEL_CHECKPOINT)
model = SamModel.from_pretrained(MODEL_CHECKPOINT)

lora_config = LoraConfig(r=RANK, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    if "mask_decoder" in name: param.requires_grad = True

model.to(DEVICE)

print(f">>> Loading Optimized Weights from {BEST_CHECKPOINT}...")
model.load_state_dict(torch.load(BEST_CHECKPOINT, map_location=DEVICE))
print("✅ Weights successfully loaded.")

results = []

# Evaluate CVC-ClinicDB
res_cvc = evaluate_external_dataset(model, "CVC-ClinicDB", CVC_PATHS["images"], CVC_PATHS["masks"], processor)
if res_cvc: results.append(res_cvc)

# Evaluate ETIS-LaribPolypDB
res_etis = evaluate_external_dataset(model, "ETIS-LaribPolypDB", ETIS_PATHS["images"], ETIS_PATHS["masks"], processor)
if res_etis: results.append(res_etis)

# Save cumulative results to CSV
if results:
    pd.DataFrame(results).to_csv("external_generalization_results.csv", index=False)
    print("\n✅ Saved numerical results to external_generalization_results.csv")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Optimized Data
try:
    df = pd.read_csv('raw_dices_Optimized.csv')
except FileNotFoundError:
    print("Error: 'raw_dices_Optimized.csv' not found. Ensure optimized evaluation run has finished.")
    exit()

baseline_dices = df['Baseline']
polysam_dices = df['PolySAM']

# 2. Calculate improvements for color coding
improvement = polysam_dices - baseline_dices
# Green for improvement, Red for regression, Gray for identical
colors = np.where(improvement > 0.01, 'seagreen',
                  np.where(improvement < -0.01, 'crimson', 'gray'))

# 3. Create the Plot
plt.figure(figsize=(7, 7), dpi=300)

# Plot the scatter points
plt.scatter(baseline_dices, polysam_dices, c=colors, alpha=0.6, edgecolors='w', linewidth=0.5, s=50)

# Plot the y=x reference line
plt.plot([0, 1.05], [0, 1.05], 'k--', lw=2, label='y = x (No Improvement)')

# Add grid and labels
plt.grid(True, linestyle=':', alpha=0.6)
plt.title('Instance-Level Performance: PolypSAM-Lite vs. Baseline', fontweight='bold')
plt.xlabel('Zero-Shot SAM Dice Score', fontweight='bold')
plt.ylabel('PolypSAM-Lite Dice Score', fontweight='bold')

# Set limits to focus on the high-performance region, but allow room for outliers
plt.xlim(0.0, 1.05)
plt.ylim(0.0, 1.05)

# Add a custom legend for the color coding
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Improved', markerfacecolor='seagreen', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Regressed', markerfacecolor='crimson', markersize=8),
    Line2D([0], [0], color='k', linestyle='--', lw=2, label='y = x')
]
plt.legend(handles=legend_elements, loc='lower right')

# 4. Save the figure
save_name = 'fig_scatter_comparison_Optimized.pdf'
plt.savefig(save_name, bbox_inches='tight')
plt.show()

print(f"Saved pairwise scatter plot to '{save_name}'")

# Print a quick statistic for paper text
num_improved = sum(improvement > 0.01)
num_regressed = sum(improvement < -0.01)
total = len(df)
print(f"\nQuick Stat for Paper:")
print(f"PolySAM-Lite improved the Dice score by >1% on {num_improved}/{total} images ({(num_improved/total)*100:.1f}%).")

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Ablation Data
# LoRA Rank (5 Epochs)
rank_labels = ['Rank 1', 'Rank 4']
rank_scores = [0.9454, 0.9456]

# LoRA Target (3 Epochs)
target_labels = ['All-Linear', 'QKV (Ours)']
target_scores = [0.9420, 0.9423]

# Data Efficiency (3 Epochs)
data_labels = ['50% Data', '100% Data']
data_scores = [0.9225, 0.9423]

# 2. Setup Figure
# Using a 1x3 grid for the three distinct ablations
fig, axes = plt.subplots(1, 3, figsize=(12, 5), dpi=300)
sns.set_theme(style="whitegrid")

# Define uniform y-axis limits to make visual comparison fair across subplots
# Zooming in slightly (0.91 to 0.96) because the margins are small but important
Y_MIN = 0.91
Y_MAX = 0.955

# Color palettes for a clean, academic look
color_baseline = 'lightslategray'
color_winner = 'darkorange'

# --- Subplot 1: LoRA Rank ---
bars1 = axes[0].bar(rank_labels, rank_scores, color=[color_baseline, color_winner], edgecolor='black', width=0.6)
axes[0].set_title('a. Rank Capacity', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Dice Similarity Coefficient (DSC)', fontweight='bold')
axes[0].set_ylim(Y_MIN, Y_MAX)

# --- Subplot 2: LoRA Target ---
bars2 = axes[1].bar(target_labels, target_scores, color=[color_baseline, color_winner], edgecolor='black', width=0.6)
axes[1].set_title('b. Target Modules', fontweight='bold', fontsize=12)
axes[1].set_ylim(Y_MIN, Y_MAX)
axes[1].tick_params(labelleft=False) # Hide y-axis numbers for middle plots to save space

# --- Subplot 3: Data Efficiency ---
bars3 = axes[2].bar(data_labels, data_scores, color=[color_baseline, color_winner], edgecolor='black', width=0.6)
axes[2].set_title('c. Training Data', fontweight='bold', fontsize=12)
axes[2].set_ylim(Y_MIN, Y_MAX)
axes[2].tick_params(labelleft=False)

# 3. Add text labels on top of the bars
def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart."""
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Format the label to 4 decimal places
        label = f"{y_value:.4f}"

        # Create annotation
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, spacing),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=10
        )

add_value_labels(axes[0])
add_value_labels(axes[1])
add_value_labels(axes[2])

# 4. Final Formatting & Save
plt.tight_layout()
plt.savefig('fig_ablation_studies_combined.pdf', bbox_inches='tight')
plt.show()

print("Saved comprehensive ablation figure as 'fig_ablation_studies_combined.pdf'")

import os
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# 1. DATASET PATHS
# ==========================================
datasets = {
    "Kvasir-SEG": {
        "images": "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/images",
        "masks": "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/masks"
    },
    "CVC-ClinicDB": {
        "images": "/kaggle/input/datasets/balraj98/cvcclinicdb/PNG/Original",
        "masks": "/kaggle/input/datasets/balraj98/cvcclinicdb/PNG/Ground Truth"
    },
    "ETIS-LaribPolypDB": {
        "images": "/kaggle/input/datasets/nguyenvoquocduong/etis-laribpolypdb/images",
        "masks": "/kaggle/input/datasets/nguyenvoquocduong/etis-laribpolypdb/masks"
    }
}

# ==========================================
# 2. HELPER TO FETCH SAMPLES
# ==========================================
def get_sample_pair(image_dir, mask_dir):
    """Fetches the first valid image and mask pair from the directories."""
    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    # Get sorted lists of files to ensure they match
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(valid_exts)])

    if not image_files or not mask_files:
        raise FileNotFoundError(f"No valid images found in {image_dir} or {mask_dir}")

    # Pick the very first pair
    img_path = os.path.join(image_dir, image_files[0])
    mask_path = os.path.join(mask_dir, mask_files[0])

    return Image.open(img_path).convert("RGB"), Image.open(mask_path).convert("L")

# ==========================================
# 3. GENERATE THE FIGURE
# ==========================================
fig, axes = plt.subplots(3, 2, figsize=(6, 9), dpi=300)

for i, (dataset_name, paths) in enumerate(datasets.items()):
    try:
        raw_img, mask_img = get_sample_pair(paths["images"], paths["masks"])

        # Plot Raw Image
        axes[i, 0].imshow(raw_img)
        axes[i, 0].axis('off')
        if i == 0: axes[i, 0].set_title("Endoscopic Image", fontweight='bold')

        # Add Dataset Label to the left of the image
        axes[i, 0].text(-0.1, 0.5, dataset_name, transform=axes[i, 0].transAxes,
                        fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)

        # Plot Ground Truth Mask
        axes[i, 1].imshow(mask_img, cmap='gray')
        axes[i, 1].axis('off')
        if i == 0: axes[i, 1].set_title("Ground Truth Mask", fontweight='bold')

    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')

plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.1)

# Save the figure
save_path = "fig_dataset_samples.pdf"
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"Saved dataset sample figure as '{save_path}'")

import os
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# 1. DATASET PATHS
# ==========================================
datasets = {
    "Kvasir-SEG": {
        "images": "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/images",
        "masks": "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/masks"
    },
    "CVC-ClinicDB": {
        "images": "/kaggle/input/datasets/balraj98/cvcclinicdb/PNG/Original",
        "masks": "/kaggle/input/datasets/balraj98/cvcclinicdb/PNG/Ground Truth"
    },
    "ETIS-LaribPolypDB": {
        "images": "/kaggle/input/datasets/nguyenvoquocduong/etis-laribpolypdb/images",
        "masks": "/kaggle/input/datasets/nguyenvoquocduong/etis-laribpolypdb/masks"
    }
}

# ==========================================
# 2. HELPER TO FETCH SAMPLES
# ==========================================
def get_sample_pair(image_dir, mask_dir):
    """Fetches the first valid image and mask pair from the directories."""
    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(valid_exts)])

    if not image_files or not mask_files:
        raise FileNotFoundError(f"No valid images found in {image_dir} or {mask_dir}")

    img_path = os.path.join(image_dir, image_files[0])
    mask_path = os.path.join(mask_dir, mask_files[0])

    return Image.open(img_path).convert("RGB"), Image.open(mask_path).convert("L")

# ==========================================
# 3. GENERATE THE HORIZONTAL FIGURE
# ==========================================
# 2 Rows (Input, Mask) x 3 Columns (Datasets)
fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=300)

for i, (dataset_name, paths) in enumerate(datasets.items()):
    try:
        raw_img, mask_img = get_sample_pair(paths["images"], paths["masks"])

        # --- Top Row: Raw Images ---
        axes[0, i].imshow(raw_img)
        axes[0, i].axis('off')
        axes[0, i].set_title(dataset_name, fontweight='bold', fontsize=14, pad=10)

        # Add Row Label on the far left of the top row
        if i == 0:
            axes[0, i].text(-0.1, 0.5, "Input Image", transform=axes[0, i].transAxes,
                            fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)

        # --- Bottom Row: Ground Truth Masks ---
        axes[1, i].imshow(mask_img, cmap='gray')
        axes[1, i].axis('off')

        # Add Row Label on the far left of the bottom row
        if i == 0:
            axes[1, i].text(-0.1, 0.5, "Ground Truth", transform=axes[1, i].transAxes,
                            fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)

    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        axes[0, i].axis('off')
        axes[1, i].axis('off')

# Compress the whitespace between images
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Save the figure
save_path = "fig_dataset_samples_horizontal.pdf"
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"Saved horizontal dataset sample figure as '{save_path}'")

import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

print(">>> Re-initializing PolySAM-Lite...")
model = get_model(rank=4)

# Load the best 10-epoch weights
checkpoint_path = "checkpoint_Final_Model_Optimized_best.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.eval()

# Tracking lists
dices, ious = [], []
precisions, recalls = [], []
specificities, npvs = [], []

print(">>> Running Comprehensive Evaluation on Test Set...")

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_boxes = batch["input_boxes"].to(DEVICE)
        gt = batch["ground_truth_mask"].numpy()[0, 0].astype(bool)

        # Inference
        outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
        pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(bool)

        # Pixel-level Confusion Matrix components
        tp = np.logical_and(pred, gt).sum()
        tn = np.logical_and(~pred, ~gt).sum()
        fp = np.logical_and(pred, ~gt).sum()
        fn = np.logical_and(~pred, gt).sum()

        # 1. Standard Metrics
        dice = (2. * tp) / (pred.sum() + gt.sum() + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8) # Positive Predictive Value (PPV)
        recall = tp / (tp + fn + 1e-8)    # Sensitivity

        # 2. New Metrics
        specificity = tn / (tn + fp + 1e-8)
        npv = tn / (tn + fn + 1e-8)

        # Append to lists
        dices.append(dice)
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        npvs.append(npv)

# Calculate Means
final_results = {
    "Dice (DSC)": np.mean(dices),
    "IoU (mIoU)": np.mean(ious),
    "Precision (PPV)": np.mean(precisions),
    "Recall (Sens)": np.mean(recalls),
    "Specificity (TNR)": np.mean(specificities),
    "NPV": np.mean(npvs)
}

print("\n" + "="*40)
print("FINAL COMPREHENSIVE METRICS")
print("="*40)
for metric, value in final_results.items():
    print(f"{metric:<18}: {value:.4f}")
print("="*40)

# Save the expanded results
pd.DataFrame([final_results]).to_csv("results_table_Comprehensive.csv", index=False)
print("✅ Saved to 'results_table_Comprehensive.csv'")

# ==========================================
# 3. UPDATED EVALUATION FUNCTION
# ==========================================
def evaluate_external_dataset_comprehensive(model, dataset_name, image_dir, mask_dir, processor):
    print(f"\n" + "="*50)
    print(f"Evaluating Cross-Domain Generalization on: {dataset_name}")
    print("="*50)

    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"❌ ERROR: Paths for {dataset_name} not found.")
        return None

    dataset = ExternalPolypDataset(image_dir, mask_dir, processor)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Loaded {len(dataset)} samples from {dataset_name}.")

    model.eval()
    dices, ious, precisions, recalls = [], [], [], []
    specificities, npvs = [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Testing {dataset_name}")):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_boxes = batch["input_boxes"].to(DEVICE)
            gt = batch["ground_truth_mask"].numpy()[0, 0].astype(bool)

            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(bool)

            # Pixel-level Confusion Matrix components
            tp = np.logical_and(pred, gt).sum()
            tn = np.logical_and(~pred, ~gt).sum()
            fp = np.logical_and(pred, ~gt).sum()
            fn = np.logical_and(~pred, gt).sum()

            # Standard Metrics
            dice = (2. * tp) / (pred.sum() + gt.sum() + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            # New Metrics
            specificity = tn / (tn + fp + 1e-8)
            npv = tn / (tn + fn + 1e-8)

            dices.append(dice)
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)
            specificities.append(specificity)
            npvs.append(npv)

            # Generate Qualitative Visualizations (First 3 Images)
            if i < 3:
                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.imshow(batch["original_image"][0])
                plt.axis('off')
                plt.title(f"Input ({batch['filename'][0]})")

                plt.subplot(1, 3, 2)
                plt.imshow(gt, cmap='gray')
                plt.axis('off')
                plt.title("Ground Truth")

                plt.subplot(1, 3, 3)
                plt.imshow(pred, cmap='gray')
                plt.axis('off')
                plt.title(f"PolySAM-Lite (DSC: {dice:.4f})")

                plt.tight_layout()
                fig_name = f"External_{dataset_name}_Fig_{i+1}.pdf"
                plt.savefig(fig_name, bbox_inches='tight', dpi=300)
                plt.close()

    # Calculate Means
    final_results = {
        "Dataset": dataset_name,
        "Dice": np.mean(dices),
        "IoU": np.mean(ious),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "Specificity": np.mean(specificities),
        "NPV": np.mean(npvs)
    }

    # Print Results
    print(f"\n>>> FINAL METRICS: {dataset_name}")
    for metric, value in final_results.items():
        if metric != "Dataset":
            print(f"{metric:<15}: {value:.4f}")

    return final_results

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
print(">>> Rebuilding PolySAM-Lite Architecture...")
processor = SamProcessor.from_pretrained(MODEL_CHECKPOINT)
model = SamModel.from_pretrained(MODEL_CHECKPOINT)

lora_config = LoraConfig(r=RANK, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    if "mask_decoder" in name: param.requires_grad = True

model.to(DEVICE)

print(f">>> Loading Optimized Weights from {BEST_CHECKPOINT}...")
model.load_state_dict(torch.load(BEST_CHECKPOINT, map_location=DEVICE))
print("✅ Weights successfully loaded.")

results = []

# Evaluate CVC-ClinicDB
res_cvc = evaluate_external_dataset_comprehensive(model, "CVC-ClinicDB", CVC_PATHS["images"], CVC_PATHS["masks"], processor)
if res_cvc: results.append(res_cvc)

# Evaluate ETIS-LaribPolypDB
res_etis = evaluate_external_dataset_comprehensive(model, "ETIS-LaribPolypDB", ETIS_PATHS["images"], ETIS_PATHS["masks"], processor)
if res_etis: results.append(res_etis)

# Save cumulative results to CSV
if results:
    pd.DataFrame(results).to_csv("external_generalization_results_Comprehensive.csv", index=False)
    print("\n✅ Saved numerical results to external_generalization_results_Comprehensive.csv")

"""OLD STUFF BELOW"""

# ==============================================================================
# PolySAM-Lite: Final Scientific Pipeline
# 1. Zero-Shot Baseline (Seed 42)
# 2. Ablation A: Rank 1 vs 4 (3 Epochs)
# 3. Ablation B: Data 50% vs 100% (3 Epochs)
# 4. Final Production: Rank 4, 100% Data (5 Epochs)
# ==============================================================================
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm.notebook import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import SamModel, SamProcessor
from peft import LoraConfig, get_peft_model
from monai.losses import DiceCELoss
from scipy.stats import ttest_rel

# --- 1. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ABLATION_EPOCHS = 3
FINAL_EPOCHS = 5
SEED = 42

CONFIG = {
    "split_root": "/kaggle/input/kvasirseg",
    "image_root": "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/images",
    "mask_root": "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/masks",
    "model_checkpoint": "facebook/sam-vit-base",
    "lr": 1e-4,
    "batch_size": 1,
    "grad_accum": 4
}

# --- 2. UTILS ---
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f">>> Seed Locked: {seed}")

class KvasirSAMDataset(Dataset):
    def __init__(self, split_root, image_root, mask_root, split_file, processor):
        self.image_root = image_root
        self.mask_root = mask_root
        self.processor = processor
        with open(os.path.join(split_root, split_file), 'r') as f:
            self.file_names = [line.strip() for line in f.readlines() if line.strip()]
    def __len__(self): return len(self.file_names)
    def get_bounding_box(self, mask):
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0: return [0, 0, 1, 1]
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = mask.shape
        noise = np.random.randint(0, 10)
        return [max(0, x_min-noise), max(0, y_min-noise), min(W, x_max+noise), min(H, y_max+noise)]
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = Image.open(os.path.join(self.image_root, file_name + ".jpg")).convert("RGB").resize((256, 256))
        mask = Image.open(os.path.join(self.mask_root, file_name + ".jpg")).convert("L").resize((256, 256), resample=Image.NEAREST)
        image_np = np.array(image)
        mask_np = (np.array(mask) > 128).astype(np.uint8)
        inputs = self.processor(image_np, input_boxes=[[self.get_bounding_box(mask_np)]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask_np).float().unsqueeze(0)
        inputs["original_image"] = image_np
        return inputs

print(">>> Initializing Data...")
processor = SamProcessor.from_pretrained(CONFIG["model_checkpoint"])
train_ds = KvasirSAMDataset(CONFIG["split_root"], CONFIG["image_root"], CONFIG["mask_root"], "train.txt", processor)
val_ds = KvasirSAMDataset(CONFIG["split_root"], CONFIG["image_root"], CONFIG["mask_root"], "val.txt", processor)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

def get_model(rank=None, zero_shot=False):
    model = SamModel.from_pretrained(CONFIG["model_checkpoint"])
    if not zero_shot:
        lora_config = LoraConfig(r=rank, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.05, bias="none")
        model = get_peft_model(model, lora_config)
        for name, param in model.named_parameters():
            if "mask_decoder" in name: param.requires_grad = True
    else:
        # For Zero-Shot, we freeze everything
        for param in model.parameters():
            param.requires_grad = False
    return model.to(DEVICE)

import torch
import pandas as pd
import os

def train_engine(model, loader, epochs, desc, start_epoch=0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    model.train()

    # List to store loss history
    history = []

    # Loop from start_epoch to (start_epoch + epochs)
    for epoch in range(start_epoch, start_epoch + epochs):
        pbar = tqdm(loader, desc=f"{desc} | Ep {epoch+1}/{start_epoch + epochs}", leave=True)

        # Variable to track loss for the entire epoch
        epoch_loss_sum = 0.0
        num_batches = 0

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_boxes = batch["input_boxes"].to(DEVICE)
            gt_masks = batch["ground_truth_mask"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            loss = loss_fn(outputs.pred_masks.squeeze(1), gt_masks)

            (loss / CONFIG["grad_accum"]).backward()

            if (step + 1) % CONFIG["grad_accum"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            # --- Track Loss ---
            current_loss = loss.item()
            epoch_loss_sum += current_loss
            num_batches += 1

            pbar.set_postfix({"Loss": f"{current_loss:.4f}"})

        # --- End of Epoch Actions ---

        # 1. Calculate Average Loss
        avg_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0.0

        # 2. Append to history and Save CSV
        # We overwrite the file each epoch so it always contains the full history up to that point
        history.append({"Epoch": epoch + 1, "Loss": avg_loss})
        log_filename = f"training_log_{desc.replace(' ', '_')}.csv"
        pd.DataFrame(history).to_csv(log_filename, index=False)

        # 3. Save Model Checkpoint
        checkpoint_filename = f"checkpoint_{desc.replace(' ', '_')}_ep{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_filename)

        print(f"--> Saved {checkpoint_filename} and updated {log_filename}")

    return model

def evaluate_metrics(model, desc="Eval"):
    model.eval()
    dice_list = []
    print(f"Evaluating {desc}...")
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_boxes = batch["input_boxes"].to(DEVICE)
            gt = batch["ground_truth_mask"].numpy()[0, 0].astype(bool)
            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(bool)
            dice_list.append((2. * np.logical_and(pred, gt).sum()) / (pred.sum() + gt.sum() + 1e-8))
    return dice_list

# --- STEP 0: ZERO-SHOT BASELINE (Seed 42) ---
print(f"\n>>> STEP 0: Running Zero-Shot Baseline...")
set_seed(SEED)
baseline_model = get_model(zero_shot=True)
baseline_dices = evaluate_metrics(baseline_model, desc="Zero-Shot SAM")
baseline_mean = np.mean(baseline_dices)
print(f"Baseline Mean Dice: {baseline_mean:.4f}")

# ==============================================================================
# PolySAM-Lite: Final Scientific Pipeline
# 1. Zero-Shot Baseline (Seed 42)
# 2. Ablation A: Rank 1 vs 4 (3 Epochs)
# 3. Ablation B: Data 50% vs 100% (3 Epochs)
# 4. Final Production: Rank 4, 100% Data (5 Epochs)
# ==============================================================================
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm.notebook import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import SamModel, SamProcessor
from peft import LoraConfig, get_peft_model
from monai.losses import DiceCELoss
from scipy.stats import ttest_rel

# --- 1. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ABLATION_EPOCHS = 3
FINAL_EPOCHS = 5
SEED = 42

CONFIG = {
    "split_root": "/kaggle/input/kvasirseg",
    "image_root": "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/images",
    "mask_root": "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/masks",
    "model_checkpoint": "facebook/sam-vit-base",
    "lr": 1e-4,
    "batch_size": 1,
    "grad_accum": 4
}

# --- 2. UTILS ---
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f">>> Seed Locked: {seed}")

class KvasirSAMDataset(Dataset):
    def __init__(self, split_root, image_root, mask_root, split_file, processor):
        self.image_root = image_root
        self.mask_root = mask_root
        self.processor = processor
        with open(os.path.join(split_root, split_file), 'r') as f:
            self.file_names = [line.strip() for line in f.readlines() if line.strip()]
    def __len__(self): return len(self.file_names)
    def get_bounding_box(self, mask):
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0: return [0, 0, 1, 1]
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = mask.shape
        noise = np.random.randint(0, 10)
        return [max(0, x_min-noise), max(0, y_min-noise), min(W, x_max+noise), min(H, y_max+noise)]
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = Image.open(os.path.join(self.image_root, file_name + ".jpg")).convert("RGB").resize((256, 256))
        mask = Image.open(os.path.join(self.mask_root, file_name + ".jpg")).convert("L").resize((256, 256), resample=Image.NEAREST)
        image_np = np.array(image)
        mask_np = (np.array(mask) > 128).astype(np.uint8)
        inputs = self.processor(image_np, input_boxes=[[self.get_bounding_box(mask_np)]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask_np).float().unsqueeze(0)
        inputs["original_image"] = image_np
        return inputs

print(">>> Initializing Data...")
processor = SamProcessor.from_pretrained(CONFIG["model_checkpoint"])
train_ds = KvasirSAMDataset(CONFIG["split_root"], CONFIG["image_root"], CONFIG["mask_root"], "train.txt", processor)
val_ds = KvasirSAMDataset(CONFIG["split_root"], CONFIG["image_root"], CONFIG["mask_root"], "val.txt", processor)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

def get_model(rank=None, zero_shot=False):
    model = SamModel.from_pretrained(CONFIG["model_checkpoint"])
    if not zero_shot:
        lora_config = LoraConfig(r=rank, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.05, bias="none")
        model = get_peft_model(model, lora_config)
        for name, param in model.named_parameters():
            if "mask_decoder" in name: param.requires_grad = True
    else:
        # For Zero-Shot, we freeze everything
        for param in model.parameters():
            param.requires_grad = False
    return model.to(DEVICE)

def train_engine(model, loader, epochs, desc):
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"{desc} | Ep {epoch+1}/{epochs}", leave=True)
        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_boxes = batch["input_boxes"].to(DEVICE)
            gt_masks = batch["ground_truth_mask"].to(DEVICE)
            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            loss = loss_fn(outputs.pred_masks.squeeze(1), gt_masks)
            (loss / CONFIG["grad_accum"]).backward()
            if (step + 1) % CONFIG["grad_accum"] == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    return model

def evaluate_metrics(model, desc="Eval"):
    model.eval()
    dice_list = []
    print(f"Evaluating {desc}...")
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_boxes = batch["input_boxes"].to(DEVICE)
            gt = batch["ground_truth_mask"].numpy()[0, 0].astype(bool)
            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(bool)
            dice_list.append((2. * np.logical_and(pred, gt).sum()) / (pred.sum() + gt.sum() + 1e-8))
    return dice_list

# --- STEP 0: ZERO-SHOT BASELINE (Seed 42) ---
print(f"\n>>> STEP 0: Running Zero-Shot Baseline...")
set_seed(SEED)
baseline_model = get_model(zero_shot=True)
baseline_dices = evaluate_metrics(baseline_model, desc="Zero-Shot SAM")
baseline_mean = np.mean(baseline_dices)
print(f"Baseline Mean Dice: {baseline_mean:.4f}")

# --- STEP 1: ABLATION A (Rank 1 vs 4) ---
print(f"\n>>> STEP 1: Rank Ablation (1 vs 4) - {ABLATION_EPOCHS} Epochs")
ranks = [1, 4]
rank_results = []
for r in ranks:
    set_seed(SEED)
    model = get_model(rank=r)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    model = train_engine(model, train_loader, epochs=ABLATION_EPOCHS, desc=f"Rank {r}")
    dices = evaluate_metrics(model, desc=f"Rank {r}")
    mean_dice = np.mean(dices)
    print(f"Rank {r} Mean Dice: {mean_dice:.4f}")
    rank_results.append({"Rank": r, "Dice": mean_dice})
pd.DataFrame(rank_results).to_csv("ablation_rank.csv", index=False)

WINNER_RANK = 4 # Hardcoded Choice

# --- STEP 2: ABLATION B (Data 50% vs 100%) ---
print(f"\n>>> STEP 2: Data Efficiency (50% vs 100%) - {ABLATION_EPOCHS} Epochs")
fractions = [0.50]
data_results = []
for frac in fractions:
    set_seed(SEED)
    if frac == 1.0: loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    else:
        indices = torch.randperm(len(train_ds))[:int(len(train_ds)*frac)]
        subset_ds = Subset(train_ds, indices)
        loader = DataLoader(subset_ds, batch_size=1, shuffle=True)

    model = get_model(rank=WINNER_RANK)
    model = train_engine(model, loader, epochs=ABLATION_EPOCHS, desc=f"Data {frac}")
    dices = evaluate_metrics(model, desc=f"Data {frac}")
    mean_dice = np.mean(dices)
    print(f"Data {frac} Mean Dice: {mean_dice:.4f}")
    data_results.append({"Fraction": frac, "Dice": mean_dice})
pd.DataFrame(data_results).to_csv("ablation_data.csv", index=False)

WINNER_RANK = 4 # Hardcoded Choice

# --- STEP 2.5: Re-calculate Zero-Shot Baseline ---
print("\n>>> STEP 2.5: Re-calculating Zero-Shot Baseline")
set_seed(SEED)
model_base = get_model(zero_shot=True)
baseline_dices = evaluate_metrics(model_base, desc="Zero-Shot Baseline")
del model_base
torch.cuda.empty_cache()

# --- STEP 3: RESUME TRAINING (Epochs 6-10) ---
print(f"\n>>> STEP 3: Resuming Production (Rank {WINNER_RANK}, Epochs 6-10)")
set_seed(SEED)
final_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

# Initialize architecture and load the epoch 5 weights
model = get_model(rank=WINNER_RANK)
model.load_state_dict(torch.load("checkpoint_Final_Model_ep5.pth", map_location=DEVICE))
print(">>> Epoch 5 weights loaded successfully! Resuming...")

# Run 5 MORE epochs, starting the counter at 5 (so it logs as 6, 7, 8, 9, 10)
model = train_engine(
    model,
    final_loader,
    epochs=5,
    desc="Final_Model_Extended",
    start_epoch=5
)

# --- STEP 4: ARTIFACTS & STATISTICS ---
print(">>> Generating Final Artifacts...")
model.eval()
poly_dices, iou_scores, precisions, recalls = [], [], [], []

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_boxes = batch["input_boxes"].to(DEVICE)
        gt = batch["ground_truth_mask"].numpy()[0, 0].astype(bool)

        outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
        pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(bool)

        # Metrics
        inter = np.logical_and(pred, gt).sum()
        dice = (2. * inter) / (pred.sum() + gt.sum() + 1e-8)
        poly_dices.append(dice)
        iou_scores.append(inter / (np.logical_or(pred, gt).sum() + 1e-8))
        precisions.append(inter / (pred.sum() + 1e-8))
        recalls.append(inter / (gt.sum() + 1e-8))

        # Save Images
        if i < 3:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 3, 1); plt.imshow(batch["original_image"][0]); plt.axis('off'); plt.title("Input")
            plt.subplot(1, 3, 2); plt.imshow(gt, cmap='gray'); plt.axis('off'); plt.title("Ground Truth")
            plt.subplot(1, 3, 3); plt.imshow(pred, cmap='gray'); plt.axis('off'); plt.title(f"PolySAM-Lite (10 Ep)")
            plt.tight_layout()
            plt.savefig(f"Final_Figure_10ep_{i}.pdf", bbox_inches='tight', dpi=300)
            plt.close()

# T-TEST vs BASELINE
t_stat, p_value = ttest_rel(poly_dices, baseline_dices)
print(f"\n>>> Paired T-Test Results")
print(f"PolySAM (10 Ep) Mean: {np.mean(poly_dices):.4f}")
print(f"Baseline Mean: {np.mean(baseline_dices):.4f}")
print(f"P-Value: {p_value:.4e}")

# SAVE STATS
final_metrics = {
    "Dice": np.mean(poly_dices),
    "IoU": np.mean(iou_scores),
    "Precision": np.mean(precisions),
    "Recall": np.mean(recalls),
    "P_Value": p_value
}
pd.DataFrame([final_metrics]).to_csv("results_table_10ep.csv", index=False)
pd.DataFrame({"Baseline": baseline_dices, "PolySAM": poly_dices}).to_csv("raw_dices_10ep.csv", index=False)

# BOX PLOT
plt.figure(figsize=(6, 5), dpi=300)
plt.boxplot([baseline_dices, poly_dices], labels=['Zero-Shot SAM', f'PolySAM-Lite (10 Ep)'], patch_artist=True)
plt.title(f'Significance Analysis (p={p_value:.2e})')
plt.ylabel('Dice Score')
plt.grid(True, alpha=0.3)
plt.savefig("fig_boxplot_stats_10ep.pdf")

print(">>> ALL DONE.")

WINNER_RANK = 4 # Hardcoded Choice
# --- STEP 3: FINAL PRODUCTION (Rank 4, 100%, 5 Epochs) ---
print(f"\n>>> STEP 3: Final Production (Rank {WINNER_RANK}, {FINAL_EPOCH} Epochs)")
set_seed(SEED)
final_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
model = get_model(rank=WINNER_RANK)
model = train_engine(model, final_loader, epochs=FINAL_EPOCH, desc="Final Model")

# --- STEP 4: ARTIFACTS & STATISTICS ---
print(">>> Generating Final Artifacts...")
model.eval()
poly_dices, iou_scores, precisions, recalls = [], [], [], []

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_boxes = batch["input_boxes"].to(DEVICE)
        gt = batch["ground_truth_mask"].numpy()[0, 0].astype(bool)

        outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
        pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(bool)

        # Metrics
        inter = np.logical_and(pred, gt).sum()
        dice = (2. * inter) / (pred.sum() + gt.sum() + 1e-8)
        poly_dices.append(dice)
        iou_scores.append(inter / (np.logical_or(pred, gt).sum() + 1e-8))
        precisions.append(inter / (pred.sum() + 1e-8))
        recalls.append(inter / (gt.sum() + 1e-8))

        # Save Images
        if i < 3:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 3, 1); plt.imshow(batch["original_image"][0]); plt.axis('off'); plt.title("Input")
            plt.subplot(1, 3, 2); plt.imshow(gt, cmap='gray'); plt.axis('off'); plt.title("Ground Truth")
            plt.subplot(1, 3, 3); plt.imshow(pred, cmap='gray'); plt.axis('off'); plt.title(f"PolySAM-Lite")
            plt.tight_layout()
            plt.savefig(f"Final_Figure_{i}.pdf", bbox_inches='tight', dpi=300)
            plt.close()

# T-TEST vs BASELINE
t_stat, p_value = ttest_rel(poly_dices, baseline_dices)
print(f"\n>>> Paired T-Test Results")
print(f"PolySAM Mean: {np.mean(poly_dices):.4f}")
print(f"Baseline Mean: {np.mean(baseline_dices):.4f}")
print(f"P-Value: {p_value:.4e}")

# SAVE STATS
final_metrics = {
    "Dice": np.mean(poly_dices),
    "IoU": np.mean(iou_scores),
    "Precision": np.mean(precisions),
    "Recall": np.mean(recalls),
    "P_Value": p_value
}
pd.DataFrame([final_metrics]).to_csv("results_table.csv", index=False)
pd.DataFrame({"Baseline": baseline_dices, "PolySAM": poly_dices}).to_csv("raw_dices.csv", index=False)

# BOX PLOT
plt.figure(figsize=(6, 5), dpi=300)
plt.boxplot([baseline_dices, poly_dices], labels=['Zero-Shot SAM', f'PolySAM-Lite'], patch_artist=True)
plt.title(f'Significance Analysis (p={p_value:.2e})')
plt.ylabel('Dice Score')
plt.grid(True, alpha=0.3)
plt.savefig("fig_boxplot_stats.pdf")

print(">>> ALL DONE.")

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# PolySAM-Lite: Efficient SAM Adaptation via LoRA
# Author: Umar Hasan (Generated by Gemini)
# Environment: Kaggle (2x T4 GPUs)
# ==========================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# Install dependencies if not present
# !pip install -q transformers peft monai

from transformers import SamModel, SamProcessor
from peft import LoraConfig, get_peft_model
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

# --- Configuration ---
CONFIG = {
    "data_root": "/kaggle/input/kvasirseg/Kvasir-SEG",
    "model_checkpoint": "facebook/sam-vit-base",
    "batch_size": 1,          # T4 can handle 4-8 depending on resolution
    "gradient_accumulation_steps": 4, # Simulate Batch Size = 4
    "epochs": 5,              # Quick adaptation
    "lr": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lora_rank": 4,
    "lora_alpha": 32
}

print(f"Running on device: {CONFIG['device']}")

# --- 1. Dataset Class ---
class KvasirSAMDataset(Dataset):
    def __init__(self, root_dir, split_file, processor):
        self.root_dir = root_dir
        self.processor = processor

        # Load file names
        with open(os.path.join(root_dir, split_file), 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.file_names)

    def get_bounding_box(self, ground_truth_map):
        # Get bounding box from mask for SAM prompt
        # SAM expects format [x_min, y_min, x_max, y_max]
        y_indices, x_indices = np.where(ground_truth_map > 0)
        if len(x_indices) == 0: # Empty mask fallback
            return [0, 0, 1, 1]
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # Add small random noise/perturbation to box to make model robust
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))

        return [x_min, y_min, x_max, y_max]

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        # Kvasir filenames might differ, adjusting based on standard structure
        img_path = os.path.join(self.root_dir, "images", file_name)
        mask_path = os.path.join(self.root_dir, "masks", file_name)

        # If files don't have extensions in txt, try adding jpg
        if not os.path.exists(img_path):
            img_path += ".jpg"
            mask_path += ".jpg"

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Grayscale

        # Resize for consistency (SAM handles its own resizing, but we prep arrays)
        image = image.resize((256, 256))
        mask = mask.resize((256, 256), resample=Image.NEAREST)

        image_np = np.array(image)
        mask_np = np.array(mask)
        mask_np = (mask_np > 128).astype(np.uint8) # Binary mask

        # Generate Box Prompt from Mask
        prompt_box = self.get_bounding_box(mask_np)

        # Process Inputs for SAM
        inputs = self.processor(
            image_np,
            input_boxes=[[prompt_box]],
            return_tensors="pt"
        )

        # Squeeze batch dims added by processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask_np).float().unsqueeze(0) # (1, H, W)

        return inputs

# --- 2. Load Model & Apply LoRA (CORRECTED FOR FUSED LAYERS) ---
import torch.nn as nn

processor = SamProcessor.from_pretrained(CONFIG["model_checkpoint"])
model = SamModel.from_pretrained(CONFIG["model_checkpoint"])

# 1. Define Standard Target Modules for Fused SAM
targets = ["qkv"]

print(f"Applying LoRA to Fused Attention Layers: {targets}")

# --- 2. Load Model & Apply LoRA (Manual Decoder Training) ---
import torch.nn as nn

processor = SamProcessor.from_pretrained(CONFIG["model_checkpoint"])
model = SamModel.from_pretrained(CONFIG["model_checkpoint"])

# 1. Define Standard Target Modules for Fused SAM
targets = ["qkv"]
print(f"Applying LoRA to Fused Attention Layers: {targets}")

# 2. Configure LoRA (Remove modules_to_save to prevent wrapper crash)
lora_config = LoraConfig(
    r=CONFIG["lora_rank"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=targets,
    lora_dropout=0.05,
    bias="none"
    # REMOVED: modules_to_save=["mask_decoder"] -> This caused the TypeError
)

model = get_peft_model(model, lora_config)

# 3. Manually Unfreeze the Mask Decoder (Bypassing the PEFT wrapper bug)
# We iterate through parameters and unlock the decoder gradients manually
for name, param in model.named_parameters():
    if "mask_decoder" in name:
        param.requires_grad = True

print("--- SUCCESS! Trainable Parameters ---")
model.print_trainable_parameters()
model.to(CONFIG["device"])

# --- 3. Data Loaders (FIXED PATHS) ---
import os

CONFIG["split_root"] = "/kaggle/input/kvasirseg"
CONFIG["image_root"] = "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/images"
CONFIG["mask_root"] = "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/masks"

class KvasirSAMDataset(Dataset):
    def __init__(self, split_root, image_root, mask_root, split_file, processor):
        self.image_root = image_root
        self.mask_root = mask_root
        self.processor = processor

        # Load file names from the split file (e.g., /kaggle/input/kvasirseg/train.txt)
        with open(os.path.join(split_root, split_file), 'r') as f:
            # Strip newlines and ensure no empty lines
            self.file_names = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.file_names)

    def get_bounding_box(self, ground_truth_map):
        y_indices, x_indices = np.where(ground_truth_map > 0)
        if len(x_indices) == 0:
            return [0, 0, 1, 1] # Fallback for empty masks
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # Add random noise for robust training
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 10))
        x_max = min(W, x_max + np.random.randint(0, 10))
        y_min = max(0, y_min - np.random.randint(0, 10))
        y_max = min(H, y_max + np.random.randint(0, 10))

        return [x_min, y_min, x_max, y_max]

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        # Construct paths. The user said text files have NO extension, so we append .jpg
        img_path = os.path.join(self.image_root, file_name + ".jpg")
        mask_path = os.path.join(self.mask_root, file_name + ".jpg")

        # Safety check for file existence
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Could not find image: {img_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize to 256x256 for memory efficiency
        image = image.resize((256, 256))
        mask = mask.resize((256, 256), resample=Image.NEAREST)

        image_np = np.array(image)
        mask_np = np.array(mask)
        mask_np = (mask_np > 128).astype(np.uint8) # Thresholding

        # Generate Box Prompt
        prompt_box = self.get_bounding_box(mask_np)

        # Process inputs for SAM
        inputs = self.processor(
            image_np,
            input_boxes=[[prompt_box]],
            return_tensors="pt"
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask_np).float().unsqueeze(0)

        return inputs

# Re-initialize Data Loaders with the new paths
train_ds = KvasirSAMDataset(
    CONFIG["split_root"], CONFIG["image_root"], CONFIG["mask_root"],
    "train.txt", processor
)
val_ds = KvasirSAMDataset(
    CONFIG["split_root"], CONFIG["image_root"], CONFIG["mask_root"],
    "val.txt", processor
)

train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)

print(f"Data Loaded Successfully!")
print(f"Training Images: {len(train_ds)}")
print(f"Validation Images: {len(val_ds)}")

# --- 4. Training Loop (With Gradient Accumulation) ---
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
seg_loss_func = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

print(f"\nStarting training for {CONFIG['epochs']} epochs...")

for epoch in range(CONFIG["epochs"]):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad() # Initialize gradient zero

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

    for step, batch in enumerate(pbar):
        # Move to GPU
        pixel_values = batch["pixel_values"].to(CONFIG["device"])
        input_boxes = batch["input_boxes"].to(CONFIG["device"])
        gt_masks = batch["ground_truth_mask"].to(CONFIG["device"])

        # Forward Pass
        outputs = model(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=False
        )

        pred_masks = outputs.pred_masks.squeeze(1)
        loss = seg_loss_func(pred_masks, gt_masks)

        # Normalize loss for accumulation
        loss = loss / CONFIG["gradient_accumulation_steps"]
        loss.backward()

        # Step optimizer only after N accumulation steps
        if (step + 1) % CONFIG["gradient_accumulation_steps"] == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * CONFIG["gradient_accumulation_steps"]
        pbar.set_postfix({"Loss": loss.item() * CONFIG["gradient_accumulation_steps"]})

# --- 5. Save & Visualize ---
model.save_pretrained("PolySAM-Lite-Adapter")
print("Model Saved!")

# Quick Viz
idx = 0
batch = next(iter(val_loader))
with torch.no_grad():
    outputs = model(
        pixel_values=batch["pixel_values"].to(CONFIG["device"]),
        input_boxes=batch["input_boxes"].to(CONFIG["device"]),
        multimask_output=False
    )
    pred = torch.sigmoid(outputs.pred_masks[idx].squeeze(1)).cpu().numpy() > 0.5

plt.figure(figsize=(10,5))
plt.subplot(1,3,1); plt.title("Input"); plt.imshow(batch["pixel_values"][idx].permute(1,2,0))
plt.subplot(1,3,2); plt.title("GT Mask"); plt.imshow(batch["ground_truth_mask"][idx].squeeze())
plt.subplot(1,3,3); plt.title("Prediction"); plt.imshow(pred.squeeze())
plt.show()

# ==========================================
# PolySAM-Lite: Full Ablation Study (Rank & Data Efficiency)
# Author: Generated for Scientific Reports Submission
# ==========================================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import SamModel, SamProcessor
from peft import LoraConfig, get_peft_model
from monai.losses import DiceCELoss

# --- 1. Configuration & Paths ---
CONFIG = {
    "split_root": "/kaggle/input/kvasirseg",
    "image_root": "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/images",
    "mask_root": "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/masks",
    "model_checkpoint": "facebook/sam-vit-base",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 1e-4,
    "epochs": 5,           # Sufficient for ablation trends
    "batch_size": 1,       # Crucial for T4 VRAM
    "grad_accum_steps": 4  # Simulate Batch Size=4
}

print(f"Running Ablation on {CONFIG['device']}...")

# --- 2. Dataset Class (Standardized) ---
class KvasirSAMDataset(Dataset):
    def __init__(self, split_root, image_root, mask_root, split_file, processor):
        self.image_root = image_root
        self.mask_root = mask_root
        self.processor = processor
        with open(os.path.join(split_root, split_file), 'r') as f:
            self.file_names = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self): return len(self.file_names)

    def get_bounding_box(self, mask):
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0: return [0, 0, 1, 1]
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # Add noise for training robustness
        H, W = mask.shape
        noise = np.random.randint(0, 10)
        return [max(0, x_min-noise), max(0, y_min-noise), min(W, x_max+noise), min(H, y_max+noise)]

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = os.path.join(self.image_root, file_name + ".jpg")
        mask_path = os.path.join(self.mask_root, file_name + ".jpg")

        image = Image.open(img_path).convert("RGB").resize((256, 256))
        mask = Image.open(mask_path).convert("L").resize((256, 256), resample=Image.NEAREST)

        image_np = np.array(image)
        mask_np = (np.array(mask) > 128).astype(np.uint8)
        prompt_box = self.get_bounding_box(mask_np)

        inputs = self.processor(image_np, input_boxes=[[prompt_box]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask_np).float().unsqueeze(0)
        return inputs

# Initialize Processor & Data
processor = SamProcessor.from_pretrained(CONFIG["model_checkpoint"])
train_ds = KvasirSAMDataset(CONFIG["split_root"], CONFIG["image_root"], CONFIG["mask_root"], "train.txt", processor)
val_ds = KvasirSAMDataset(CONFIG["split_root"], CONFIG["image_root"], CONFIG["mask_root"], "val.txt", processor)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# --- 3. Helper Functions ---
def get_model(rank):
    """Re-initializes a fresh model with specific Rank"""
    model = SamModel.from_pretrained(CONFIG["model_checkpoint"])
    lora_config = LoraConfig(
        r=rank, lora_alpha=32, target_modules=["qkv"],
        lora_dropout=0.05, bias="none"
    )
    model = get_peft_model(model, lora_config)
    # Manually unfreeze decoder
    for name, param in model.named_parameters():
        if "mask_decoder" in name: param.requires_grad = True
    return model.to(CONFIG["device"])

def train_engine(model, train_loader, desc):
    """Standard Training Loop"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    model.train()

    for epoch in range(CONFIG["epochs"]):
        epoch_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"{desc} | Epoch {epoch+1}", leave=False)

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(CONFIG["device"])
            input_boxes = batch["input_boxes"].to(CONFIG["device"])
            gt_masks = batch["ground_truth_mask"].to(CONFIG["device"])

            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            pred_masks = outputs.pred_masks.squeeze(1)
            loss = loss_fn(pred_masks, gt_masks)

            # Gradient Accumulation
            loss = loss / CONFIG["grad_accum_steps"]
            loss.backward()

            if (step + 1) % CONFIG["grad_accum_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
    return model

def evaluate(model):
    """Returns Dice Score"""
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(CONFIG["device"])
            input_boxes = batch["input_boxes"].to(CONFIG["device"])
            gt_masks = batch["ground_truth_mask"].numpy()

            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(np.uint8)
            gt = gt_masks[0, 0].astype(np.uint8)

            intersection = np.logical_and(pred, gt).sum()
            dice = (2. * intersection) / (pred.sum() + gt.sum() + 1e-8)
            dice_scores.append(dice)
    return np.mean(dice_scores)

# --- 4. EXPERIMENT A: Rank Ablation ---
print("\n>>> STARTING EXPERIMENT A: RANK ABLATION")
ranks = [1, 8, 16]
rank_results = []

for r in ranks:
    torch.cuda.empty_cache()
    print(f"--- Training Rank {r} ---")
    model = get_model(r)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    model = train_engine(model, train_loader, desc=f"Rank {r}")
    score = evaluate(model)

    # Count Trainable Params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Rank {r} Result: Dice={score:.4f} | Params={params}")
    rank_results.append({"Rank": r, "Dice": score, "Params": params})

# Save and Plot A
df_rank = pd.DataFrame(rank_results)
df_rank.to_csv("ablation_rank.csv", index=False)

plt.figure(figsize=(6, 4), dpi=300)
sns.lineplot(data=df_rank, x="Rank", y="Dice", marker="o", color="firebrick")
plt.title("Impact of LoRA Rank on Segmentation Performance")
plt.ylabel("Dice Similarity Coefficient")
plt.xlabel("LoRA Rank (r)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig_ablation_rank.pdf")
print("Saved fig_ablation_rank.pdf")

# --- 5. EXPERIMENT B: Data Efficiency ---
print("\n>>> STARTING EXPERIMENT B: DATA EFFICIENCY")
fractions = [0.10, 0.25, 0.50] # 10%, 25%, 50%
data_results = []

for frac in fractions:
    torch.cuda.empty_cache()
    print(f"--- Training with {frac*100}% Data ---")

    # Create Subset
    indices = torch.randperm(len(train_ds))[:int(len(train_ds)*frac)]
    subset_ds = Subset(train_ds, indices)
    subset_loader = DataLoader(subset_ds, batch_size=1, shuffle=True)

    # Train Standard Model (Rank=4)
    model = get_model(rank=4)
    model = train_engine(model, subset_loader, desc=f"Data {frac}")
    score = evaluate(model)

    print(f"Data {frac*100}% Result: Dice={score:.4f}")
    data_results.append({"Fraction": frac*100, "Dice": score})

data_results.append({"Fraction": 100, "Dice": 0.9050})

# Save and Plot B
df_data = pd.DataFrame(data_results).sort_values("Fraction")
df_data.to_csv("ablation_data.csv", index=False)

plt.figure(figsize=(6, 4), dpi=300)
sns.barplot(data=df_data, x="Fraction", y="Dice", color="steelblue")
plt.title("Data Efficiency of PolySAM-Lite")
plt.ylabel("Dice Similarity Coefficient")
plt.xlabel("Percentage of Training Data Used")
plt.ylim(0.7, 1.0) # Zoom in to show differences
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("fig_ablation_data.pdf")
print("Saved fig_ablation_data.pdf")

print("\n=== ALL EXPERIMENTS COMPLETED ===")

# Quick Viz
idx = 0
batch = next(iter(val_loader))
with torch.no_grad():
    outputs = model(
        pixel_values=batch["pixel_values"].to(CONFIG["device"]),
        input_boxes=batch["input_boxes"].to(CONFIG["device"]),
        multimask_output=False
    )
    pred = torch.sigmoid(outputs.pred_masks[idx].squeeze(1)).cpu().numpy() > 0.5

plt.figure(figsize=(10, 5))

# 1. Input Image
plt.subplot(1, 3, 1)
plt.title("Input")
plt.imshow(batch["pixel_values"][idx].permute(1, 2, 0))

# 2. Ground Truth
plt.subplot(1, 3, 2)
plt.title("GT Mask")
plt.imshow(batch["ground_truth_mask"][idx].squeeze())

# 3. Prediction
plt.subplot(1, 3, 3)
plt.title("Prediction")
plt.imshow(pred.squeeze())

# --- SAVE COMMAND ---
# bbox_inches='tight' prevents labels from being cut off
plt.savefig("visualization.pdf", format="pdf", bbox_inches="tight")

plt.show()

# ==========================================
# PolySAM-Lite: Evaluation & Benchmark Script
# ==========================================

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor
from peft import PeftModel, LoraConfig
from PIL import Image
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# --- Configuration (Same as Training) ---
CONFIG = {
    "split_root": "/kaggle/input/kvasirseg",
    "image_root": "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/images",
    "mask_root": "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/masks",
    "model_checkpoint": "facebook/sam-vit-base",
    "lora_adapter_path": "PolySAM-Lite-Adapter",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- Metrics Helper ---
def compute_metrics(pred_mask, gt_mask):
    # Flatten
    pred = pred_mask.flatten().astype(bool)
    gt = gt_mask.flatten().astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    dice = (2. * intersection) / (pred.sum() + gt.sum() + 1e-8)
    iou = intersection / (union + 1e-8)

    # Precision / Recall
    precision = intersection / (pred.sum() + 1e-8)
    recall = intersection / (gt.sum() + 1e-8)

    return {"Dice": dice, "IoU": iou, "Precision": precision, "Recall": recall}

# --- 1. Load Data (Validation Set Only) ---
# (Paste the KvasirSAMDataset class from previous step here if needed,
# or ensure it's still in memory. I assume it's defined.)
from torch.utils.data import Dataset # Re-import if necessary

class KvasirSAMDataset(Dataset):
    def __init__(self, split_root, image_root, mask_root, split_file, processor):
        self.image_root = image_root
        self.mask_root = mask_root
        self.processor = processor
        with open(os.path.join(split_root, split_file), 'r') as f:
            self.file_names = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self): return len(self.file_names)

    def get_bounding_box(self, ground_truth_map):
        y_indices, x_indices = np.where(ground_truth_map > 0)
        if len(x_indices) == 0: return [0, 0, 1, 1]
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # No noise for evaluation! We want precise bounding boxes.
        return [x_min, y_min, x_max, y_max]

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = os.path.join(self.image_root, file_name + ".jpg")
        mask_path = os.path.join(self.mask_root, file_name + ".jpg")

        image = Image.open(img_path).convert("RGB").resize((256, 256))
        mask = Image.open(mask_path).convert("L").resize((256, 256), resample=Image.NEAREST)

        image_np = np.array(image)
        mask_np = (np.array(mask) > 128).astype(np.uint8)
        prompt_box = self.get_bounding_box(mask_np)

        inputs = self.processor(image_np, input_boxes=[[prompt_box]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = mask_np # Keep as numpy for metrics
        inputs["original_image"] = image_np   # Keep for visualization
        return inputs

processor = SamProcessor.from_pretrained(CONFIG["model_checkpoint"])
val_ds = KvasirSAMDataset(
    CONFIG["split_root"], CONFIG["image_root"], CONFIG["mask_root"],
    "val.txt", processor
)
# Batch size 1 is crucial for accurate per-image metrics
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# --- 2. Evaluation Loop ---
def evaluate_model(model, name="Model"):
    model.eval()
    model.to(CONFIG["device"])
    results = []

    print(f"Evaluating {name}...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            # Move inputs to GPU
            pixel_values = batch["pixel_values"].to(CONFIG["device"])
            input_boxes = batch["input_boxes"].to(CONFIG["device"])

            # Forward
            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)

            # Post-process
            pred_mask = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(np.uint8)
            gt_mask = batch["ground_truth_mask"][0].numpy().astype(np.uint8)

            # Calculate Metrics
            metrics = compute_metrics(pred_mask, gt_mask)
            results.append(metrics)

            # Save Visualization for the first 3 images (For the Paper!)
            if i < 3 and name == "PolySAM-Lite":
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 3, 1); plt.title("Input Image"); plt.imshow(batch["original_image"][0])
                plt.subplot(1, 3, 2); plt.title("Ground Truth"); plt.imshow(gt_mask, cmap='gray')
                plt.subplot(1, 3, 3); plt.title(f"{name} Pred"); plt.imshow(pred_mask, cmap='gray')
                plt.savefig(f"Figure_{i}_{name}.pdf")
                plt.close()

    df = pd.DataFrame(results)
    print(f"\n--- {name} Results ---")
    print(df.mean())
    return df.mean()

# --- 3. Run Comparison ---

# A. Evaluate Zero-Shot (Baseline)
# Load base model WITHOUT LoRA
base_model = SamModel.from_pretrained(CONFIG["model_checkpoint"])
baseline_results = evaluate_model(base_model, name="SAM-ZeroShot")

# B. Evaluate PolySAM-Lite
# Load LoRA adapters
# Note: "targets=['qkv']" ensures we match the training config
from peft import LoraConfig
# We need to explicitly define config if loading fails to infer
lora_config = LoraConfig(r=4, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.05, bias="none")

# Load model with adapter
trained_model = PeftModel.from_pretrained(
    base_model,
    CONFIG["lora_adapter_path"],
    config=lora_config
)
polysam_results = evaluate_model(trained_model, name="PolySAM-Lite")

# --- 4. Generate Table for Paper ---
print("\n====== FINAL PAPER TABLE ======")
comparison = pd.DataFrame([baseline_results, polysam_results])
comparison.index = ["SAM (Baseline)", "PolySAM-Lite (Ours)"]
print(comparison)
comparison.to_csv("results_table.csv")

import matplotlib.pyplot as plt
import numpy as np

# 1. The Data
epochs = [1, 2, 3, 4, 5]
loss_values = [0.118, 0.091, 0.075, 0.069, 0.071]

# 2. Setup Plot Style (Publication Quality)
plt.figure(figsize=(8, 5), dpi=300)  # High resolution (300 DPI) for PDF
plt.style.use('seaborn-v0_8-whitegrid') # Clean academic style

# 3. Plot the Line
plt.plot(epochs, loss_values,
         marker='o',         # Circular markers for data points
         linestyle='-',      # Solid line
         color='#2E86C1',    # Professional Blue
         linewidth=2.5,      # Thicker line for visibility
         markersize=8,       # Visible markers
         label='Training Loss')

# 4. Formatting
plt.title('Training Convergence Dynamics: PolySAM-Lite', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epochs', fontsize=12, fontweight='bold')
plt.ylabel('Dice-Cross Entropy Loss', fontsize=12, fontweight='bold')

# Ensure X-axis only shows integers 1 through 5
plt.xticks(epochs)
plt.ylim(0, max(loss_values) + 0.02) # Start Y-axis at 0 for context

# Add a light grid for readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# 5. Save the Figure
plt.tight_layout()
plt.savefig('loss_curve.pdf', bbox_inches='tight', dpi=300)
print("Figure saved as loss_curve.pdf")

# Show the plot
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# --- ROC & Confusion Matrix Generator ---
print("Generating ROC and Confusion Matrix...")
model.eval()

all_preds_prob = []  # For ROC (Probabilities)
all_preds_bin = []   # For CM (Binary)
all_gts = []         # Ground Truth

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Collecting Stats"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_boxes = batch["input_boxes"].to(DEVICE)
        gt = batch["ground_truth_mask"].numpy().flatten().astype(int)

        # Forward pass
        outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)

        # Probabilities (Sigmoid)
        probs = torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

        # Subsample pixels to save RAM (Optional: take every 10th pixel if OOM)
        # Here we take all, assuming standard RAM is enough for 120 images
        all_preds_prob.extend(probs)
        all_preds_bin.extend(preds)
        all_gts.extend(gt)

# 1. ROC Curve
fpr, tpr, _ = roc_curve(all_gts, all_preds_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5), dpi=300)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'PolySAM-Lite (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('fig_roc.pdf')
print(f"Saved fig_roc.png (AUC={roc_auc:.4f})")

# 2. Confusion Matrix
cm = confusion_matrix(all_gts, all_preds_bin, normalize='true') # Normalized
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Background', 'Polyp'])
fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
disp.plot(cmap='Blues', ax=ax, values_format='.2%')
plt.title("Normalized Confusion Matrix")
plt.savefig('fig_confusion_matrix.pdf')
print("Saved fig_confusion_matrix.pdf")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Hardcoded Data based on our experiment logs
data_rank = {
    'Rank': ['1', '4'],
    'Dice': [0.9242, 0.9328]
}
df_rank = pd.DataFrame(data_rank)

data_efficiency = {
    'Percent': ['50%', '100%'],
    'Dice': [0.9240, 0.9328]
}
df_data = pd.DataFrame(data_efficiency)

# Setup figure
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

# Plot 1: Rank Sensitivity
sns.barplot(x='Rank', y='Dice', data=df_rank, ax=axes[0], palette='viridis', edgecolor='black')
axes[0].set_title('Rank Sensitivity Analysis')
axes[0].set_xlabel('LoRA Rank (r)')
axes[0].set_ylabel('Dice Score')
axes[0].set_ylim(0.90, 0.95) # Zoom in to show the difference
for i, v in enumerate(df_rank['Dice']):
    axes[0].text(i, v + 0.001, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')

# Plot 2: Data Efficiency
sns.barplot(x='Percent', y='Dice', data=df_data, ax=axes[1], palette='magma', edgecolor='black')
axes[1].set_title('Data Efficiency Analysis')
axes[1].set_xlabel('Training Data Percentage')
axes[1].set_ylabel('Dice Score')
axes[1].set_ylim(0.90, 0.95)
for i, v in enumerate(df_data['Dice']):
    axes[1].text(i, v + 0.001, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('fig_ablation_combined.pdf', bbox_inches='tight')
plt.savefig('fig_ablation_combined.png', bbox_inches='tight')
print("Figure saved as fig_ablation_combined.pdf")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Hardcoded Data based on our experiment logs
data_rank = {
    'Rank': ['1', '4'],
    'Dice': [0.9242, 0.9328]
}
df_rank = pd.DataFrame(data_rank)

data_efficiency = {
    'Percent': ['50%', '100%'],
    'Dice': [0.9240, 0.9328]
}
df_data = pd.DataFrame(data_efficiency)

# --- Plot 1: Rank Sensitivity ---
plt.figure(figsize=(5, 4), dpi=300)
ax1 = sns.barplot(x='Rank', y='Dice', data=df_rank, palette='viridis', edgecolor='black')
ax1.set_title('Rank Sensitivity Analysis')
ax1.set_xlabel('LoRA Rank (r)')
ax1.set_ylabel('Dice Score')
ax1.set_ylim(0.90, 0.95) # Zoom in to show the difference
for i, v in enumerate(df_rank['Dice']):
    ax1.text(i, v + 0.001, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('fig_ablation_rank.pdf', bbox_inches='tight')
print("Figure saved as fig_ablation_rank.pdf")
plt.close()

# --- Plot 2: Data Efficiency ---
plt.figure(figsize=(5, 4), dpi=300)
ax2 = sns.barplot(x='Percent', y='Dice', data=df_data, palette='magma', edgecolor='black')
ax2.set_title('Data Efficiency Analysis')
ax2.set_xlabel('Training Data Percentage')
ax2.set_ylabel('Dice Score')
ax2.set_ylim(0.90, 0.95)
for i, v in enumerate(df_data['Dice']):
    ax2.text(i, v + 0.001, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('fig_ablation_data.pdf', bbox_inches='tight')
print("Figure saved as fig_ablation_data.pdf")
plt.close()

import matplotlib.pyplot as plt
import torch
import numpy as np

# Ensure model is in eval mode
model.eval()

worst_dice = 1.0
worst_batch = None
worst_pred = None
worst_gt = None

print("Scanning for the hardest case (Failure Mode)...")

with torch.no_grad():
    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_boxes = batch["input_boxes"].to(DEVICE)
        gt = batch["ground_truth_mask"].numpy()[0, 0].astype(bool)

        # Inference
        outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
        pred = (torch.sigmoid(outputs.pred_masks[0, 0, 0]).cpu().numpy() > 0.5).astype(bool)

        # Calculate Dice for this single image
        intersection = np.logical_and(pred, gt).sum()
        dice = (2. * intersection) / (pred.sum() + gt.sum() + 1e-8)

        # Track the worst performance
        if dice < worst_dice and gt.sum() > 0: # Ensure ground truth is not empty
            worst_dice = dice
            worst_batch = batch
            worst_pred = pred
            worst_gt = gt

print(f"Found Hardest Case! Dice Score: {worst_dice:.4f}")

# Plot and Save
if worst_batch is not None:
    plt.figure(figsize=(10, 4))

    # Input Image
    plt.subplot(1, 3, 1)
    plt.imshow(worst_batch["original_image"][0])
    plt.axis('off')
    plt.title(f"Input (Dice: {worst_dice:.2f})")

    # Ground Truth
    plt.subplot(1, 3, 2)
    plt.imshow(worst_gt, cmap='gray')
    plt.axis('off')
    plt.title("Ground Truth")

    # Prediction (Failure)
    plt.subplot(1, 3, 3)
    plt.imshow(worst_pred, cmap='gray')
    plt.axis('off')
    plt.title("PolySAM-Lite Prediction")

    plt.tight_layout()
    plt.savefig("Final_Figure_4.pdf", bbox_inches='tight', dpi=300)
    plt.savefig("Final_Figure_4.png", bbox_inches='tight', dpi=300)
    print("Saved Final_Figure_4.pdf (Hard Case)")
else:
    print("No valid validation samples found.")

