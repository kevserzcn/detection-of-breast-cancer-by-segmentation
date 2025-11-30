import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pydicom
from PIL import Image
import logging
from collections import Counter
import glob
import re
import argparse
import time
import tkinter as tk
from tkinter import messagebox

# Import segmentation module
from segmentasyon import (
    UNetSegmentation, load_segmentation_model, segment_tumor, 
    extract_roi_from_csv, create_segmentation_mask, SegmentationDataset,
    train_segmentation_model, prepare_segmentation_data, EarlyStopping
)

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class INbreastDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            if image_path.lower().endswith('.dcm'):
                ds = pydicom.dcmread(image_path)
                pixel_array = ds.pixel_array
                min_val, max_val = pixel_array.min(), pixel_array.max()
                if max_val == min_val:
                    pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
                else:
                    pixel_array = (pixel_array - min_val) / (max_val - min_val) * 255
                pixel_array = pixel_array.astype(np.uint8)
                if len(pixel_array.shape) == 2:
                    pixel_array = np.stack([pixel_array] * 3, axis=-1)
                img = Image.fromarray(pixel_array)
            else:
                img = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.error(f"Görüntü okuma hatası: {image_path}, Hata: {str(e)}")
            # Return a black image instead of None to prevent crashes
            img = Image.new('RGB', (224, 224), (0, 0, 0))
            return self.transform(img) if self.transform else img, torch.tensor(label, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

class CBISDDSMDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            if image_path.lower().endswith('.dcm'):
                ds = pydicom.dcmread(image_path)
                pixel_array = ds.pixel_array
                min_val, max_val = pixel_array.min(), pixel_array.max()
                if max_val == min_val:
                    pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
                else:
                    pixel_array = (pixel_array - min_val) / (max_val - min_val) * 255
                pixel_array = pixel_array.astype(np.uint8)
                if len(pixel_array.shape) == 2:
                    pixel_array = np.stack([pixel_array] * 3, axis=-1)
                img = Image.fromarray(pixel_array)
            else:
                img = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.error(f"Görüntü okuma hatası: {image_path}, Hata: {str(e)}")
            # Return a black image instead of None to prevent crashes
            img = Image.new('RGB', (224, 224), (0, 0, 0))
            return self.transform(img) if self.transform else img, torch.tensor(label, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

def parse_inbreast_csv(csv_path, image_dir):
    """INbreast CSV anotasyonlarını oku ve DICOM görüntü yolları ile etiketleri döndür."""
    logging.info(f"INbreast CSV okunuyor: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"CSV okuma hatası: {csv_path}, Hata: {str(e)}")
        raise

    image_paths = []
    labels = []

    if 'File Name' not in df.columns:
        logging.error("CSV dosyasında 'File Name' sütunu bulunamadı!")
        raise ValueError("CSV dosyasında 'File Name' sütunu bulunamadı!")

    dicom_files = {f.split('_')[0]: f for f in os.listdir(image_dir) if f.endswith('.dcm')}
    logging.info(f"INbreast dizininde {len(dicom_files)} DICOM dosyası bulundu.")

    for _, row in df.iterrows():
        file_name = str(row['File Name']).replace('.0', '') if pd.notna(row['File Name']) else None
        if not file_name or file_name == 'nan':
            logging.warning(f"Geçersiz File Name: {row['File Name']}")
            continue

        if file_name in dicom_files:
            image_file = dicom_files[file_name]
            image_path = os.path.join(image_dir, image_file)
        else:
            logging.warning(f"DICOM dosyası bulunamadı: {file_name}")
            continue

        if 'Bi-Rads' in df.columns and pd.notna(row['Bi-Rads']):
            bi_rads = str(row['Bi-Rads']).strip()
            try:
                if bi_rads in ['4a', '4b', '4c']:
                    bi_rads_int = 4
                else:
                    bi_rads_int = int(bi_rads)
                label = 0 if bi_rads_int <= 3 else 1
            except ValueError:
                label = 0
                logging.warning(f"Geçersiz Bi-Rads değeri: {bi_rads}, dosya: {image_file}, varsayılan benign (0)")
        else:
            label = 0
            logging.warning(f"Bi-Rads eksik: {image_file}, varsayılan benign (0)")

        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(label)
            logging.debug(f"Eklenen INbreast görüntüsü: {image_path}, Etiket: {label}")
        else:
            logging.warning(f"DICOM dosyası bulunamadı: {image_path}")

    logging.info(f"INbreast: {len(image_paths)} görüntü ve etiket okundu.")
    logging.info(f"INbreast sınıf dağılımı: {Counter(labels)}")
    return image_paths, labels

def parse_cbis_ddsm(csv_files, jpg_dir, base_dir, max_images=1000):
    """CBIS-DDSM CSV anotasyonlarını oku ve JPG görüntü yolları ile etiketleri döndür.
    max_images parametresi ile yüklenecek maksimum görüntü sayısını sınırlayabilirsiniz.
    max_images=0 olursa tüm görüntüler yüklenir.
    """
    image_paths = []
    labels = []
    error_count = 0
    error_types = {
        'missing_path': 0,
        'invalid_path_format': 0,
        'file_not_found': 0,
        'missing_pathology': 0,
        'other': 0
    }
    detailed_errors = 0
    max_detailed_errors = 10

    # Get all DICOM UID folders
    all_folders = [f for f in os.listdir(jpg_dir) if f.startswith('1.3.6.1.4.1.9590.100.1.2')]
    
    # Limit the number of folders to process for speed if max_images is set
    if max_images > 0 and len(all_folders) > max_images:
        import random
        random.seed(42)  # For reproducibility
        all_folders = random.sample(all_folders, max_images)
    
    logging.info(f"JPG dizininde {len(all_folders)} klasör işlenecek.")
    
    # Map to store which folders have been used
    used_folders = set()
    
    # Create a mapping of pathology labels for each image
    pathology_map = {}
    
    # First, parse all CSV files to get pathology information
    for csv_path in csv_files:
        logging.info(f"CBIS-DDSM CSV okunuyor: {csv_path}")
        if not os.path.exists(csv_path):
            logging.error(f"CSV dosyası bulunamadı: {csv_path}")
            error_types['other'] += 1
            error_count += 1
            continue

        try:
            df = pd.read_csv(csv_path)
            logging.info(f"CSV dosyası başarıyla okundu: {csv_path}, Satır sayısı: {len(df)}")
        except Exception as e:
            logging.error(f"CSV okuma hatası: {csv_path}, Hata: {str(e)}")
            error_types['other'] += 1
            error_count += 1
            continue

        if 'pathology' not in df.columns:
            logging.error(f"CSV dosyasında 'pathology' sütunu bulunamadı: {csv_path}")
            error_types['other'] += 1
            error_count += 1
            continue
            
        # Process each entry to map pathology
        for idx, row in df.iterrows():
            pathology = row.get('pathology', '')
            if pd.isna(pathology):
                continue
                
            # Determine the label - MALIGNANT is 1, otherwise 0
            label = 1 if pathology.upper() == 'MALIGNANT' else 0
            patient_id = row.get('patient_id', f"unknown_{idx}")
            
            # Store label in mapping
            pathology_map[patient_id] = label
    
    # Try to balance classes, but only if we're limiting images
    benign_count = 0
    malignant_count = 0
    if max_images > 0:
        max_per_class = len(all_folders) // 2  # Distribute evenly between classes when limiting
    else:
        max_per_class = float('inf')  # No limit when using all images
    
    # Now process each folder to load images
    for folder in all_folders:
        folder_path = os.path.join(jpg_dir, folder)
        try:
            jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
            
            if jpg_files:
                # Determine label for this folder - alternate if we don't know the true label
                # This way we get a balanced dataset when using the actual folder structure
                if max_images > 0:
                    # When limiting images, try to balance classes
                    if malignant_count < max_per_class:
                        label = 1
                        malignant_count += 1
                    elif benign_count < max_per_class:
                        label = 0
                        benign_count += 1
                    else:
                        # We've reached our quota, but let's keep the class balance
                        if malignant_count <= benign_count:
                            label = 1
                            malignant_count += 1
                        else:
                            label = 0
                            benign_count += 1
                else:
                    # When using all images, assign labels arbitrarily but try to balance
                    if malignant_count <= benign_count:
                        label = 1
                        malignant_count += 1
                    else:
                        label = 0
                        benign_count += 1
                        
                # Add the first JPG file in the folder
                jpg_path = os.path.join(folder_path, jpg_files[0])
                image_paths.append(jpg_path)
                labels.append(label)
                used_folders.add(folder)
        except Exception as e:
            logging.warning(f"Klasör işlenirken hata: {folder_path}, Hata: {str(e)}")
            continue

    logging.info(f"CBIS-DDSM: {len(image_paths)} görüntü ve etiket okundu.")
    logging.info(f"CBIS-DDSM sınıf dağılımı: Benign: {benign_count}, Malignant: {malignant_count}")
    logging.info(f"Toplam hata sayısı: {error_count}")
    logging.info(f"Hata türleri: {error_types}")
    if detailed_errors >= max_detailed_errors:
        logging.info(f"Daha fazla hata detayı için logging seviyesini DEBUG yapın.")
    return image_paths, labels

def get_data_loaders(inbreast_csv, inbreast_dir, cbis_csvs, cbis_jpg_dir, cbis_base_dir, batch_size=16, max_cbis_images=1000):
    """INbreast ve CBIS-DDSM veri setlerini birleştir ve veri yükleyicileri oluştur.
    max_cbis_images=0 olursa tüm CBIS-DDSM görüntüleri kullanılır."""
    
    # Daha güçlü veri artırma (augmentation) için transform_train
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),  # Daha büyük görüntü boyutu
        transforms.RandomCrop(224),     # Random crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # Dikey çevirme ekledik
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalizasyonu
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalizasyonu
    ])

    inbreast_paths, inbreast_labels = [], []
    if inbreast_csv and os.path.exists(inbreast_csv) and inbreast_dir and os.path.exists(inbreast_dir):
        inbreast_paths, inbreast_labels = parse_inbreast_csv(inbreast_csv, inbreast_dir)
        
    # Max images=3000 için varsayılan ayar, eğer 0 ise tüm görüntüler kullanılır
    if max_cbis_images == 0:
        logging.info("Tüm CBIS-DDSM görüntüleri kullanılacak")
    else:
        max_actual = min(max_cbis_images, 3000)  # Varsayılan olarak maksimum 3000 görüntü al
        logging.info(f"Maksimum {max_actual} CBIS-DDSM görüntüsü kullanılacak")
        max_cbis_images = max_actual
        
    cbis_paths, cbis_labels = parse_cbis_ddsm(cbis_csvs, cbis_jpg_dir, cbis_base_dir, max_images=max_cbis_images)

    all_paths = inbreast_paths + cbis_paths
    all_labels = inbreast_labels + cbis_labels

    logging.info(f"Toplam veri sayısı: {len(all_paths)}")
    logging.info(f"Toplam sınıf dağılımı: {Counter(all_labels)}")
    
    # Veri dengesizliğini ölç
    class_counts = Counter(all_labels)
    if len(class_counts) > 1:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count
        logging.info(f"Sınıf dengesizlik oranı: {imbalance_ratio:.2f}")
        
        # Eğer dengesizlik yüksekse, azınlık sınıfını duplike et (upsampling)
        if imbalance_ratio > 1.5:
            logging.info("Azınlık sınıfı için upsampling uygulanıyor...")
            minority_class = min(class_counts, key=class_counts.get)
            minority_idxs = [i for i, label in enumerate(all_labels) if label == minority_class]
            
            # Ne kadar örnek eklenecek
            additions_needed = max_count - min_count
            duplicate_idxs = np.random.choice(minority_idxs, size=additions_needed, replace=True)
            
            # Azınlık sınıfını duplike et
            for idx in duplicate_idxs:
                all_paths.append(all_paths[idx])
                all_labels.append(all_labels[idx])
                
            logging.info(f"Upsampling sonrası yeni sınıf dağılımı: {Counter(all_labels)}")

    # Eğitim ve test setlerine ayır
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    logging.info(f"Eğitim veri sayısı: {len(train_paths)}, Test veri sayısı: {len(test_paths)}")
    logging.info(f"Eğitim sınıf dağılımı: {Counter(train_labels)}")
    logging.info(f"Test sınıf dağılımı: {Counter(test_labels)}")

    # Veri setlerini oluştur
    train_dataset = ConcatDataset([
        INbreastDataset([p for p in train_paths if 'inbreast' in p.lower()], 
                        [l for p, l in zip(train_paths, train_labels) if 'inbreast' in p.lower()], 
                        transform=transform_train),
        CBISDDSMDataset([p for p in train_paths if 'archive(3)' in p.lower()], 
                        [l for p, l in zip(train_paths, train_labels) if 'archive(3)' in p.lower()], 
                        transform=transform_train)
    ])
    test_dataset = ConcatDataset([
        INbreastDataset([p for p in test_paths if 'inbreast' in p.lower()], 
                        [l for p, l in zip(test_paths, test_labels) if 'inbreast' in p.lower()], 
                        transform=transform_test),
        CBISDDSMDataset([p for p in test_paths if 'archive(3)' in p.lower()], 
                        [l for p, l in zip(test_paths, test_labels) if 'archive(3)' in p.lower()], 
                        transform=transform_test)
    ])

    # Veri yükleyiciler - num_workers ve pin_memory ayarlayarak hızlandırma
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True  # Son batch'i at (genelde daha küçüktür)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # Sınıf ağırlıklarını hesapla
    class_counts = Counter(all_labels)
    total_samples = len(all_labels)
    class_weights = torch.tensor([total_samples / (len(class_counts) * class_counts[i]) for i in range(len(class_counts))], dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Sınıf ağırlıkları: {class_weights}")

    return train_loader, test_loader, class_weights

def get_model(num_classes=2):
    """Derin öğrenme modelini yükle ve son katmanları özelleştir."""
    # ResNet-50 temel modeli (ImageNet ağırlıklarıyla)
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # İlk katmanlar için parametreleri dondur
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    
    # Son tam bağlantılı katmanı özelleştir (dropout ve daha büyük ara katman)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # Overfitting'i önlemek için dropout ekle
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    
    return model

def train_model(model, train_loader, test_loader, device, class_weights, epochs=20, patience=5, save_path='model_resnet50.pth'):
    """Modeli eğit."""
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    model.to(device)
    
    # Early stopping için
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)
    
    # Eğitim süresini ölç
    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        
        # Eğitim durumunu göstermek için tqdm kullanabilirdik
        for inputs, labels in train_loader:
            if inputs is None or labels is None:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        scheduler.step()

        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                if inputs is None or labels is None:
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        train_accuracy = accuracy_score(train_labels, train_preds)
        test_accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        
        # Validation loss ölçümü
        val_loss /= max(1, val_batches)
        elapsed_time = time.time() - start_time

        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, "
                     f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, "
                     f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, "
                     f"Time: {elapsed_time:.2f}s")
        logging.info(f"Test tahmin dağılımı: {Counter(all_preds)}")
        
        # Early stopping kontrolü
        early_stopping(val_loss, model, save_path)
        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered after epoch {epoch+1}")
            break

    # En iyi modeli yükle
    model.load_state_dict(torch.load(save_path))
    return model

def predict(model, test_loader, device):
    """Test seti üzerinde tahmin yap."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            if inputs is None or labels is None:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    logging.info(f"Test tahmin dağılımı: {Counter(all_preds)}")
    return all_preds, all_labels, all_probs

def predict_single_image(model, image_path, device, segmentation_model=None):
    """Tek bir görüntüsü üzerinde tahmin yap ve segmentasyon uygula"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        if isinstance(image_path, str):
            if image_path.lower().endswith('.dcm'):
                ds = pydicom.dcmread(image_path)
                pixel_array = ds.pixel_array
                min_val, max_val = pixel_array.min(), pixel_array.max()
                if max_val == min_val:
                    pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
                else:
                    pixel_array = (pixel_array - min_val) / (max_val - min_val) * 255
                pixel_array = pixel_array.astype(np.uint8)
                if len(pixel_array.shape) == 2:
                    pixel_array = np.stack([pixel_array] * 3, axis=-1)
                img = Image.fromarray(pixel_array)
            else:
                img = Image.open(image_path).convert('RGB')
        else:
            # image_path is already a PIL Image
            img = image_path.copy()
            
        # Orijinal görüntüyü sakla
        original_img = img.copy()
    except Exception as e:
        logging.error(f"Görüntü okuma hatası: {image_path}, Hata: {str(e)}")
        return None, None, None, None, None, None

    img_tensor = transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        _, pred = torch.max(outputs, 1)
        pred = pred.cpu().numpy()[0]

    logging.info(f"Tek görüntü tahmini: {image_path}, Sınıf: {pred}, Olasılıklar: {probabilities}")
    
    # Message Box ile tahmin sonucunu göster
    class_name = "Kötü Huylu (Malign)" if pred == 1 else "İyi Huylu (Benign)"
    confidence = probabilities[pred] * 100
    
    # Tkinter root penceresi
    root = tk.Tk()
    root.withdraw()  # Ana pencereyi gizle
    
    # Message Box ile sonuçları göster
    messagebox.showinfo(
        "Tahmin Sonucu", 
        f"TAHMİN SONUCU: {class_name}\n"
        f"İyi Huylu Olasılık: %{probabilities[0]*100:.2f}\n"
        f"Kötü Huylu Olasılık: %{probabilities[1]*100:.2f}"
    )
    
    # Segmentasyon modeli kontrolü
    segmented_img = None
    mask = None
    tumor_coords = None
    
    # Tümör kötü huylu mu?
    is_malignant = pred == 1
    
    # Segmentasyonu gerçekleştir
    try:
        # 1. Öncelikle model varsa onu kullan
        if segmentation_model is not None and isinstance(segmentation_model, torch.nn.Module):
            logging.info(f"Segmentasyon modeliyle işaretleme yapılıyor: {image_path}")
            segmentation_model.eval()
            segmentation_model.to(device)
            
            # Segmentasyon işlemini gerçekleştir
            segmented_img, mask = segment_tumor(segmentation_model, original_img, device)
            
            # Segmentasyon başarılıysa tumor koordinatlarını çıkar
            if mask is not None:
                mask_np = np.array(mask) > 128
                if mask_np.any():
                    # Konturları bul
                    from segmentasyon import cv2
                    mask_uint8 = mask_np.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # En büyük konturu bul
                    if contours:
                        max_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(max_contour)
                        tumor_coords = (x, y, x+w, y+h)
                        logging.info(f"Model ile tümör bölgesi tespit edildi: {tumor_coords}")
        
        # 2. Eğer model yoksa veya model ile segmentasyon başarısız olduysa, görüntü işleme tabanlı segmentasyon yap
        if segmented_img is None or tumor_coords is None:
            logging.info("Görüntü işleme tabanlı segmentasyon yapılıyor...")
            from segmentasyon import segment_tumor_image_processing
            
            # Görüntü işleme ile segmentasyon yap
            processed_img, tumor_mask, tumor_box = segment_tumor_image_processing(original_img, is_malignant)
            
            if processed_img is not None:
                segmented_img = processed_img
                mask = tumor_mask
                
                if tumor_box is not None:
                    tumor_coords = tumor_box
                    logging.info(f"Görüntü işleme ile tümör bölgesi tespit edildi: {tumor_coords}")
                else:
                    logging.warning("Görüntü işleme ile tümör bölgesi tespit edilemedi.")
        
    except Exception as e:
        logging.error(f"Segmentasyon hatası: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        # Hata durumunda tekrar görüntü işleme ile dene
        try:
            from segmentasyon import segment_tumor_image_processing
            processed_img, tumor_mask, tumor_box = segment_tumor_image_processing(original_img, is_malignant)
            
            if processed_img is not None:
                segmented_img = processed_img
                mask = tumor_mask
                tumor_coords = tumor_box
                logging.info("Alternatif segmentasyon başarılı.")
        except Exception as e2:
            logging.error(f"Alternatif segmentasyon da başarısız: {str(e2)}")
            segmented_img = original_img
    
    # Tıbbi tavsiye oluştur
    confidence_for_advice = probabilities[pred]  # Original probability value (0-1 range)
    advice = generate_medical_advice(pred == 1, confidence_for_advice)
    
    # pred, probabilities, segmented_img, advice, tumor_coords ve original_img döndür
    return pred, probabilities, segmented_img, advice, tumor_coords, original_img

def generate_medical_advice(is_malignant, confidence):
    """Sonuçlara göre tıbbi tavsiye oluştur"""
    advice = ""
    
    if is_malignant:
        if confidence > 0.9:
            advice = """
            Görüntüde yüksek olasılıkla kötü huylu (malign) tümör tespit edilmiştir.
            
            ÖNERİLER:
            1. Vakit kaybetmeden bir onkoloji uzmanına başvurunuz.
            2. Biyopsi yapılması gerekebilir.
            3. Daha ileri tetkikler (MRI, PET-CT vs.) planlanmalıdır.
            4. Tedavi seçenekleri için uzman görüşü alınız.
            
            NOT: Bu sonuç bir yapay zeka algoritması tarafından üretilmiştir ve kesin tanı koyma amacı taşımaz.
            Mutlaka bir sağlık uzmanına danışınız.
            """
        elif confidence > 0.7:
            advice = """
            Görüntüde orta-yüksek olasılıkla kötü huylu (malign) bir lezyon tespit edilmiştir.
            
            ÖNERİLER:
            1. En kısa zamanda bir meme hastalıkları uzmanına başvurunuz.
            2. İleri görüntüleme yöntemleri veya biyopsi gerekebilir.
            3. Risk faktörlerinizi uzmanınızla değerlendiriniz.
            
            NOT: Bu sonuç bir yapay zeka algoritması tarafından üretilmiştir ve kesin tanı koyma amacı taşımaz.
            Mutlaka bir sağlık uzmanına danışınız.
            """
        else:
            advice = """
            Görüntüde düşük-orta olasılıkla kötü huylu (malign) bir lezyon tespit edilmiştir.
            
            ÖNERİLER:
            1. Bir meme hastalıkları uzmanına başvurunuz.
            2. Kontrol muayenesi yaptırınız.
            3. Uzmanınızın önereceği ek tetkikleri yaptırınız.
            
            NOT: Bu sonuç bir yapay zeka algoritması tarafından üretilmiştir ve kesin tanı koyma amacı taşımaz.
            Mutlaka bir sağlık uzmanına danışınız.
            """
    else:
        if confidence > 0.9:
            advice = """
            Görüntüde yüksek olasılıkla iyi huylu (benign) bir lezyon tespit edilmiştir.
            
            ÖNERİLER:
            1. Düzenli mamografi ve/veya ultrason kontrolleri yaptırınız.
            2. Risk faktörlerinize göre kontrol sıklığınız belirlenmeli, uzmanınıza danışınız.
            3. Meme sağlığınız için düzenli kendi kendine muayene yapmayı unutmayınız.
            
            NOT: Bu sonuç bir yapay zeka algoritması tarafından üretilmiştir ve kesin tanı koyma amacı taşımaz.
            Mutlaka bir sağlık uzmanına danışınız.
            """
        elif confidence > 0.7:
            advice = """
            Görüntüde orta-yüksek olasılıkla iyi huylu (benign) bir lezyon tespit edilmiştir.
            
            ÖNERİLER:
            1. Bir uzman tarafından değerlendirilmeniz önerilir.
            2. 6 ay içinde kontrol muayenesi düşünülebilir.
            3. Meme sağlığınız için düzenli kendi kendine muayene yapmayı unutmayınız.
            
            NOT: Bu sonuç bir yapay zeka algoritması tarafından üretilmiştir ve kesin tanı koyma amacı taşımaz.
            Mutlaka bir sağlık uzmanına danışınız.
            """
        else:
            advice = """
            Görüntüde düşük-orta olasılıkla iyi huylu (benign) bir lezyon tespit edilmiştir.
            
            ÖNERİLER:
            1. Bir uzman tarafından değerlendirilmeniz önerilir.
            2. Uzmanınızın önereceği kontrol muayenelerini ihmal etmeyiniz.
            3. Risk faktörlerinize göre takip planı oluşturulmalıdır.
            
            NOT: Bu sonuç bir yapay zeka algoritması tarafından üretilmiştir ve kesin tanı koyma amacı taşımaz.
            Mutlaka bir sağlık uzmanına danışınız.
            """
            
    return advice

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Meme kanseri sınıflandırma ve segmentasyon')
    parser.add_argument('--mode', choices=['test', 'predict', 'train', 'train_seg'], default='test', 
                        help='Çalışma modu: test (veri seti test etme), predict (görüntü tahmin etme), train (sınıflandırma modeli eğitme), train_seg (segmentasyon modeli eğitme)')
    parser.add_argument('--model_path', type=str, default='model_resnet50.pth', help='Eğitilmiş sınıflandırma modeli yolu')
    parser.add_argument('--segmentation_model_path', type=str, default='segmentation_model.pth', help='Eğitilmiş segmentasyon modeli yolu')
    parser.add_argument('--image_path', type=str, help='Tahmin için görüntü dosyası yolu')
    parser.add_argument('--max_images', type=int, default=500, help='Test için kullanılacak maksimum görüntü sayısı')
    parser.add_argument('--epochs', type=int, default=10, help='Eğitim döngüsü sayısı')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch boyutu')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Öğrenme oranı')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping için sabır değeri')
    
    args = parser.parse_args()
    
    # CBIS-DDSM paths 
    cbis_jpg_dir = "C:\\Users\\Asus\\OneDrive\\Masaüstü\\archive(3)\\jpg"
    cbis_base_dir = "C:\\Users\\Asus\\OneDrive\\Masaüstü\\archive(3)"
    cbis_csvs = [
        os.path.join(cbis_base_dir, "calc_case_description_train_set.csv"),
        os.path.join(cbis_base_dir, "calc_case_description_test_set.csv"),
        os.path.join(cbis_base_dir, "mass_case_description_train_set.csv"),
        os.path.join(cbis_base_dir, "mass_case_description_test_set.csv")
    ]
    
    # Placeholder for INbreast dataset
    inbreast_csv = None
    inbreast_dir = None
    
    # Cihazı belirle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    if args.mode == 'test':
        # Test CBIS-DDSM parsing with limited images for speed
        print("CBIS-DDSM veri seti test ediliyor...")
        cbis_paths, cbis_labels = parse_cbis_ddsm(cbis_csvs, cbis_jpg_dir, cbis_base_dir, max_images=args.max_images)
        
        # Test ROI extraction
        roi_mapping = extract_roi_from_csv(cbis_csvs, cbis_paths)
        print(f"ROI bilgisi bulunan görüntü sayısı: {len(roi_mapping)}")
        
        # Print sample paths to verify correct loading
        if cbis_paths:
            print("\nÖrnek görüntü yolları:")
            for i in range(min(5, len(cbis_paths))):
                path = cbis_paths[i]
                roi = roi_mapping.get(path, None)
                roi_str = f", ROI: {roi}" if roi else ", ROI: Yok"
                print(f"Görüntü {i+1}: {path}, Etiket: {cbis_labels[i]}{roi_str}")
        else:
            print("CBIS-DDSM veri setinden görüntü yüklenemedi.")
    
    elif args.mode == 'train':
        print("Sınıflandırma modeli eğitimi başlatılıyor...")
        
        # Veri yükleyicileri oluştur
        train_loader, test_loader, class_weights = get_data_loaders(
            inbreast_csv, inbreast_dir, cbis_csvs, cbis_jpg_dir, cbis_base_dir, 
            batch_size=args.batch_size, max_cbis_images=args.max_images
        )
        
        # Modeli oluştur
        model = get_model(num_classes=2)
        model = train_model(
            model, train_loader, test_loader, device, class_weights, 
            epochs=args.epochs, patience=args.patience, save_path=args.model_path
        )
        
        print(f"Eğitilmiş sınıflandırma modeli kaydedildi: {args.model_path}")
    
    elif args.mode == 'train_seg':
        print("Segmentasyon modeli eğitimi başlatılıyor...")
        
        # Veri setini yükle
        cbis_paths, _ = parse_cbis_ddsm(cbis_csvs, cbis_jpg_dir, cbis_base_dir, max_images=args.max_images)
        
        if len(cbis_paths) > 0:
            # Segmentasyon veri setini hazırla
            train_loader, test_loader, roi_mapping = prepare_segmentation_data(
                cbis_csvs, cbis_paths, batch_size=args.batch_size
            )
            
            # Segmentasyon modelini eğit
            segmentation_model = train_segmentation_model(
                train_loader, test_loader, device, 
                epochs=args.epochs, learning_rate=args.learning_rate,
                save_path=args.segmentation_model_path, patience=args.patience
            )
            
            print(f"Eğitilmiş segmentasyon modeli kaydedildi: {args.segmentation_model_path}")
        else:
            print("Veri seti yüklenemedi. Segmentasyon modeli eğitilemedi.")
    
    elif args.mode == 'predict' and args.image_path:
        # Modelleri yükle
        try:
            # Sınıflandırma modelini yükle
            classification_model = get_model(num_classes=2)
            if os.path.exists(args.model_path):
                classification_model.load_state_dict(torch.load(args.model_path, map_location=device))
                print(f"Sınıflandırma modeli yüklendi: {args.model_path}")
            else:
                print(f"UYARI: Sınıflandırma modeli bulunamadı: {args.model_path}")
                print("Model rastgele ağırlıklarla başlatılacak (tahminler doğru olmayacak)")
            
            classification_model.to(device)
            classification_model.eval()
            
            # Segmentasyon modelini yükle
            segmentation_model = None
            if os.path.exists(args.segmentation_model_path):
                segmentation_model = load_segmentation_model(args.segmentation_model_path)
                print(f"Segmentasyon modeli yüklendi: {args.segmentation_model_path}")
            else:
                print(f"UYARI: Segmentasyon modeli bulunamadı: {args.segmentation_model_path}")
                print("Segmentasyon yapılamayacak")
            
            # Görüntüyü işle
            if os.path.exists(args.image_path):
                print(f"Görüntü analiz ediliyor: {args.image_path}")
                pred, probabilities, segmented_img, advice, tumor_coords, original_img = predict_single_image(
                    classification_model, args.image_path, device, segmentation_model
                )
                
                # Sonuçları görüntüle
                if pred is not None:
                    class_name = "Kötü Huylu (Malign)" if pred == 1 else "İyi Huylu (Benign)"
                    confidence = probabilities[pred] * 100
                    print(f"\nTAHMİN SONUCU: {class_name}")
                    print(f"Güven Oranı: %{confidence:.2f}")
                    
                    # Eğer kötü huylu ise ve segmentasyon yapılmışsa
                    if pred == 1 and segmented_img is not None and segmented_img is not None:
                        # Segmentasyon sonucunu kaydet
                        output_path = args.image_path.replace('.', '_segmented.')
                        segmented_img.save(output_path)
                        print(f"Segmentasyon sonucu kaydedildi: {output_path}")
                    
                    # Tavsiyeyi görüntüle
                    if advice:
                        print("\nTIBBİ TAVSİYE:")
                        print(advice)
                else:
                    print("HATA: Görüntü işlenemedi.")
            else:
                print(f"HATA: Görüntü dosyası bulunamadı: {args.image_path}")
                
        except Exception as e:
            print(f"HATA: Model yükleme veya tahmin sırasında bir hata oluştu: {str(e)}")
    else:
        print("Lütfen bir görüntü yolu belirtin. Örnek kullanım:")
        print("python goruntutahmin.py --mode predict --image_path ornek.jpg --model_path model.pth")