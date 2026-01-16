import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torchvision import datasets, transforms
from datetime import datetime
import argparse
import logging
import time
import json
import sys
import os

from models.resnet18 import ResNet18

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}

def dtype_type(s):
    if s.lower() not in DTYPE_MAP:
        raise argparse.ArgumentTypeError(
            f"Invalid dtype '{s}'. Choose from {list(DTYPE_MAP.keys())}"
        )
    return DTYPE_MAP[s.lower()]

parser = argparse.ArgumentParser(description="Train ResNet18 on CIFAR100 in FP")
parser.add_argument("--dtype", type=dtype_type, default=torch.float32, help="Floating-point dtype to use (default: float32)")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training (default: cuda:0)")
parser.add_argument("--log_interval", type=int, default=10, help="Batches between logging training status (default: 10)")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default: 128)")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (default: 0.1)")
parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
args = parser.parse_args()

log_filename = f"resnet18_fp_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
history_filename = f"resnet18_fp_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)

best_model_filename = os.path.join(save_dir, f"resnet18_fp_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
final_model_filename = os.path.join(save_dir, f"resnet18_fp_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

dtype = args.dtype
device = args.device
log_interval = args.log_interval
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
momentum = args.momentum
weight_decay = args.weight_decay

mean = (0.50707548856735229492, 0.48654884099960327148, 0.44091776013374328613)
std = (0.26733365654945373535, 0.25643849372863769531, 0.27615079283714294434)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    ),
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    ),
    batch_size=batch_size,
    shuffle=False,
)

model = ResNet18(100, dtype=dtype, device=device)

criterion = nn.NLLLoss()
optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

logger.info(f"Training ResNet18 on CIFAR-100 with FP (dtype = {dtype}) on device {device}")
logger.info(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, momentum={momentum}, weight_decay={weight_decay}")

history = []
best_val_acc = -1.0
for epoch in range(epochs):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    epoch_train_start = time.time()
    batch_start = time.time()

    model.train()
    for i, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        data = data.to(device).to(dtype)
        target = target.to(device)
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct = (preds == target).sum().item()
            total_correct += correct
            total_samples += target.size(0)

            if (i + 1) % log_interval == 0:
                batch_end = time.time()
                logger.info(
                    f"Batch {i + 1} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Correct: {correct}/{target.size(0)} | "
                    f"Elapsed Time: {(batch_end - batch_start):.2f}s"
                )
                batch_start = batch_end

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples * 100
    epoch_train_end = time.time()

    logger.info(
        f"Epoch [{epoch+1}/{epochs}] - "
        f"Loss: {avg_loss:.4f} | "
        f"Correct: {total_correct}/{total_samples} "
        f"({accuracy:.2f}%, {(epoch_train_end - epoch_train_start):.2f}s)"
    )

    model.eval()
    val_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    val_samples = 0
    epoch_val_start = time.time()

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device).to(dtype)
            target = target.to(device)
            output = model(data)

            loss = criterion(output, target)
            val_loss += loss.item()

            pred1 = output.argmax(dim=1)
            top1_correct += (pred1 == target).sum().item()

            _, pred5 = output.topk(5, dim=1, largest=True, sorted=True)
            top5_correct += (pred5 == target.unsqueeze(1)).any(dim=1).sum().item()

            val_samples += target.size(0)

    avg_val_loss = val_loss / len(test_loader)
    top1_acc = top1_correct / val_samples * 100
    top5_acc = top5_correct / val_samples * 100
    epoch_val_end = time.time()

    logger.info(
        f"Validation {(epoch_val_end - epoch_val_start):.2f}s - "
        f"Loss: {avg_val_loss:.4f} | "
        f"Top1: {top1_correct}/{val_samples} ({top1_acc:.2f}%) | "
        f"Top5: {top5_correct}/{val_samples} ({top5_acc:.2f}%)"
    )

    history.append({
        "epoch": epoch + 1,
        "train": {
            "loss": avg_loss,
            "accuracy": accuracy,
            "correct": total_correct,
            "samples": total_samples,
            "time": epoch_train_end - epoch_train_start,
        },
        "val": {
            "loss": avg_val_loss,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "top1_correct": top1_correct,
            "top5_correct": top5_correct,
            "samples": val_samples,
            "time": epoch_val_end - epoch_val_start,
        }
    })

    with open(history_filename, "w") as f:
        json.dump(history, f, indent=2)

    if top1_acc > best_val_acc:
        best_val_acc = top1_acc
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
        }, best_model_filename)
        logger.info(f"New best model saved at epoch {epoch+1} with Top1 Acc: {top1_acc:.2f}% (Top5: {top5_acc:.2f}%)")

logger.info("Training complete.")
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}, final_model_filename)
logger.info(f"Final model saved to {final_model_filename}")