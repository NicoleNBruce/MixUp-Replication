import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataloader import get_cifar_loaders
from models.resnet import resnet18

def parse_args():
    parser = argparse.ArgumentParser("FGSM / I-FGSM Adversarial Evaluation")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--source", type=str, required=True, help="Model to generate attacks")
    parser.add_argument("--target", type=str, required=True, help="Model to evaluate against attacks (can be same as source)")
    parser.add_argument("--attack", type=str, choices=["fgsm", "ifgsm"], default="fgsm")
    parser.add_argument("--epsilon", type=float, default=8.0/255.0)
    parser.add_argument("--alpha", type=float, default=2.0/255.0, help="Step size for I-FGSM")
    parser.add_argument("--iters", type=int, default=10, help="Iterations for I-FGSM")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, test_loader, num_classes = get_cifar_loaders(dataset=args.dataset, data_dir=args.data_dir, batch_size=args.batch_size)
    
    source_model = resnet18(num_classes).to(device)
    target_model = resnet18(num_classes).to(device)
    
    #loading weights
    s_ckp = torch.load(args.source, map_location=device, weights_only=False)
    t_ckp = torch.load(args.target, map_location=device, weights_only=False)
    
#stripping 'module.' prefix from state_dict keys to ensure compatibility 
#when loading weights trained via nn.DataParallel into a single-GPU model.
    s_state = {k.replace('module.', ''): v for k, v in s_ckp["model_state"].items()}
    t_state = {k.replace('module.', ''): v for k, v in t_ckp["model_state"].items()}
    
    source_model.load_state_dict(s_state)
    target_model.load_state_dict(t_state)
    
    source_model.eval()
    target_model.eval()
    
    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    
    #getting exact bounds for the dataset to calculate valid normalized min/max per channel
    db_mean = torch.tensor([0.4914, 0.4822, 0.4465] if args.dataset == "cifar10" else [0.5071, 0.4867, 0.4408], device=device).view(1, 3, 1, 1)
    db_std = torch.tensor([0.2023, 0.1994, 0.2010] if args.dataset == "cifar10" else [0.2675, 0.2565, 0.2761], device=device).view(1, 3, 1, 1)
    lower_bound = (0.0 - db_mean) / db_std
    upper_bound = (1.0 - db_mean) / db_std
    
    progress = tqdm(test_loader, desc=f"Evaluating {args.attack}")
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True
        
        if args.attack == "fgsm":
            outputs = source_model(images)
            loss = criterion(outputs, labels)
            source_model.zero_grad()
            loss.backward()
            
            #createing FGSM attack
            adv_images = images + args.epsilon * images.grad.data.sign()
            #strictly limiting the generated perturbation back to real image space
            adv_images = torch.max(torch.min(adv_images, upper_bound), lower_bound)
            
        elif args.attack == "ifgsm":
            adv_images = images.clone().detach().requires_grad_(True)
            for _ in range(args.iters):
                outputs = source_model(adv_images)
                loss = criterion(outputs, labels)
                source_model.zero_grad()
                loss.backward()
                
                adv_images = adv_images + args.alpha * adv_images.grad.data.sign()
                eta = torch.clamp(adv_images - images, min=-args.epsilon, max=args.epsilon)
                adv_images = torch.max(torch.min(images + eta, upper_bound), lower_bound).detach().requires_grad_(True)
                
        #evaluate attack on model
        with torch.no_grad():
            outputs = target_model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        progress.set_postfix({'Error': f"{100 * (1 - correct / total):.2f}%"})
            
    print(f"\n{args.attack.upper()} Attack Top-1 Error: {100 * (1 - correct / total):.2f}%")

if __name__ == "__main__":
    main()
