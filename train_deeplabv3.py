import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import argparse

class VOCSegmentationDataset(Dataset):
    """Dataset cho VOC Segmentation với enhanced masks"""
    
    def __init__(self, image_dir, mask_dir, split='train', transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Lấy danh sách ảnh từ mask_dir
        self.images = sorted([f.replace('.png', '') 
                             for f in os.listdir(mask_dir) 
                             if f.endswith('.png')])
        
        print(f"{split} dataset: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, f'{img_name}.jpg')
        image = Image.open(img_path).convert('RGB')
        
        # Load enhanced mask
        mask_path = os.path.join(self.mask_dir, f'{img_name}.png')
        mask = Image.open(mask_path)
        
        # Resize
        image = image.resize((512, 512), Image.BILINEAR)
        mask = mask.resize((512, 512), Image.NEAREST)
        
        if self.transform:
            image = self.transform(image)
        
        mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train một epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # Calculate pixel accuracy
            preds = outputs.argmax(dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()
    
    avg_loss = total_loss / len(dataloader)
    pixel_acc = correct_pixels / total_pixels
    
    return avg_loss, pixel_acc


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = VOCSegmentationDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        split='train',
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    print("Loading DeepLabV3+ ResNet-101...")
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=True,
        num_classes=21
    )
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=10, 
        gamma=0.1
    )
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate (sử dụng train set để monitor)
        val_loss, val_acc = validate(model, train_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Pixel Acc: {val_acc:.4f}")
        
        # Learning rate
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'checkpoints/deeplab_epoch_{epoch+1}.pth'
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"✓ New best model saved! Acc: {best_acc:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Pixel Accuracy: {best_acc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, 
                       default='../archive/VOC2012_train_val/VOC2012_train_val/JPEGImages')
    parser.add_argument('--mask_dir', type=str, 
                       default='../enhanced_masks')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    main(args)