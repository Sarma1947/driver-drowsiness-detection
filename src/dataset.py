import pathlib
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DrowsinessDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = pathlib.Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() 
                               if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob('*.*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[class_dir.name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(split='train', img_size=224):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])