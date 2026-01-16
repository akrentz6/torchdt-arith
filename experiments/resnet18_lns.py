import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchdt.lns import LNS64
from torchdt.optim import TritonSGD, TritonMadam, lr_scheduler
from torchdt.transforms import ToDType, DTypeNormalize
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

parser = argparse.ArgumentParser(description="Train ResNet18 on CIFAR100 in LNS")
parser.add_argument("--prec", type=int, default=16, help="Precision for LNS dtype (default: 16)")
parser.add_argument("--table", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Use lookup table for LNS operations (default: True)")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training (default: cuda:0)")
parser.add_argument("--log_interval", type=int, default=10, help="Batches between logging training status (default: 10)")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default: 128)")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "madam"], help="Optimizer to use (default: sgd)")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (default: 0.1)")
parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="SGD weight decay (default: 1e-4)")
parser.add_argument("--beta", type=float, default=0.999, help="Madam beta parameter (default: 0.9)")
args = parser.parse_args()

date = datetime.now()
log_filename = f"resnet18_lns_train_{date.strftime('%Y%m%d_%H%M%S')}.log"
history_filename = f"resnet18_lns_history_{date.strftime('%Y%m%d_%H%M%S')}.json"

save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)

best_model_filename = os.path.join(save_dir, f"resnet18_lns_best_{date.strftime('%Y%m%d_%H%M%S')}.pt")
final_model_filename = os.path.join(save_dir, f"resnet18_lns_final_{date.strftime('%Y%m%d_%H%M%S')}.pt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

dtype = LNS64
prec = args.prec
table = args.table
device = args.device
log_interval = args.log_interval
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
momentum = args.momentum
weight_decay = args.weight_decay
beta = args.beta

dtype.set_prec(prec, table=table, table_device=device)
dtype.enable_triton()

mean = (0.50707548856735229492, 0.48654884099960327148, 0.44091776013374328613)
std = (0.26733365654945373535, 0.25643849372863769531, 0.27615079283714294434)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    ToDType(dtype, device=device),
    DTypeNormalize(dtype, mean=mean, std=std, device=device),
])
test_transform = transforms.Compose([
    ToDType(dtype, device=device),
    DTypeNormalize(dtype, mean=mean, std=std, device=device),
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

# hack to use our chosen device without erroring
@dtype.register_op("scalar_from_float")
def scalar_from_float(cls, x):
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    return cls.from_float(x_tensor)

model = ResNet18(100, dtype=dtype, device=device)

criterion = nn.NLLLoss()
if args.optimizer == "sgd":
    optimizer = TritonSGD(dtype, device, model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
elif args.optimizer == "madam":
    optimizer = TritonMadam(dtype, device, model.parameters(), lr=lr, beta=beta, use_pow=True)
else:
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

logger.info(f"Training ResNet18 on CIFAR-100 with LNS64 (f={prec}, lookup table={table}) on device {device}")
if args.optimizer == "sgd":
    logger.info(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, momentum={momentum}, weight_decay={weight_decay}")
elif args.optimizer == "madam":
    logger.info(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, beta={beta}")

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

        target = target.to(device)
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += loss.item()
            preds = output.to_float().argmax(dim=1)
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
            target = target.to(device)
            output = model(data)

            loss = criterion(output, target)
            val_loss += loss.item()

            logits = output.to_float()

            pred1 = logits.argmax(dim=1)
            top1_correct += (pred1 == target).sum().item()

            _, pred5 = logits.topk(5, dim=1, largest=True, sorted=True)
            top5_correct += (pred5 == target.unsqueeze(1)).any(dim=1).sum().item()

            val_samples += target.size(0)

    avg_val_loss = val_loss / len(test_loader)
    top1_acc = top1_correct / val_samples * 100
    top5_acc = top5_correct / val_samples * 100
    epoch_val_end = time.time()

    logger.info(
        f"Validation - "
        f"Loss: {avg_val_loss:.4f} | "
        f"Top1: {top1_correct}/{val_samples} ({top1_acc:.2f}%) | "
        f"Top5: {top5_correct}/{val_samples} ({top5_acc:.2f}%) "
        f"({(epoch_val_end - epoch_val_start):.2f}s)"
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