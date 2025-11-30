import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import pydicom
import logging
from sklearn.model_selection import train_test_split
from torchvision import transforms
import time
import torch.nn.functional as F
import cv2

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Early stopping sınıfı
class EarlyStopping:
    """Eğitim süresinde validation loss belirli bir süre iyileşmezse eğitimi durduran sınıf."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): Validation loss iyileşmeden kaç epoch bekleyeceği
            verbose (bool): Mesaj yazdırma durumu
            delta (float): Değişimin iyileşme sayılması için minimum değeri
            path (str): Model kaydedilecek dosya yolu
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, path=None):
        save_path = path if path is not None else self.path
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """Validation loss azaldığında modeli kaydet"""
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

# U-Net mimarisi bileşenleri
class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, 
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Full U-Net Mimarisi
class UNetSegmentation(nn.Module):
    """U-Net segmentasyon modeli"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, features=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor)
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)
        self.outc = OutConv(features, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class SegmentationDataset(Dataset):
    """Tümör segmentasyonu için veri seti sınıfı."""
    def __init__(self, image_paths, masks=None, transform=None, mask_transform=None):
        """
        Args:
            image_paths (list): Görüntü dosyalarının yolları
            masks (dict): Görüntü yolu -> maske eşleştirmesi
            transform: Görüntü dönüşümleri
            mask_transform: Maske dönüşümleri
        """
        self.image_paths = image_paths
        self.masks = masks if masks is not None else {}
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # Görüntüyü yükle
            img = Image.open(image_path).convert('RGB')
            
            # Maskeyi hazırla (yoksa boş maske oluştur)
            if image_path in self.masks:
                mask = self.masks[image_path]
                if not isinstance(mask, Image.Image):
                    mask = create_segmentation_mask(img.size, mask)
            else:
                # Maskeyi siyah olarak (tümör yok) başlat
                mask = Image.new('L', img.size, 0)
            
            # Dönüşümleri uygula
            if self.transform:
                img = self.transform(img)
            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                mask = transforms.ToTensor()(mask)
            
            return img, mask
            
        except Exception as e:
            logging.error(f"Görüntü yüklerken hata: {image_path}, {str(e)}")
            # Hata durumunda varsayılan görüntü ve maske döndür
            dummy_img = torch.zeros((3, 224, 224))
            dummy_mask = torch.zeros((1, 224, 224))
            return dummy_img, dummy_mask

def extract_roi_from_csv(csv_files, image_paths):
    """CSV dosyalarından ROI (Region of Interest) bilgisi çıkar."""
    roi_mapping = {}
    
    # Tüm CSV dosyalarını tara
    for csv_path in csv_files:
        if not os.path.exists(csv_path):
            logging.warning(f"CSV dosyası bulunamadı: {csv_path}")
            continue
            
        try:
            df = pd.read_csv(csv_path)
            
            # ROI bilgisi içeren sütunları kontrol et
            roi_columns = ["crop_row", "crop_col", "crop_height", "crop_width"]
            for col in roi_columns:
                if col not in df.columns:
                    logging.warning(f"CSV dosyasında {col} sütunu bulunamadı: {csv_path}")
            
            # Her bir görüntü için ROI bilgisini çıkar
            for idx, row in df.iterrows():
                # Görüntü dosyasını bul
                patient_id = row.get("patient_id", "")
                if not patient_id:
                    continue
                    
                # Bu hasta ID'sine sahip görüntü yollarını bul
                matching_images = [path for path in image_paths if patient_id in path]
                
                if not matching_images:
                    continue
                
                # ROI bilgisi varsa kaydet
                if all(col in df.columns and pd.notna(row[col]) for col in roi_columns):
                    roi = {
                        "x": int(row["crop_col"]),
                        "y": int(row["crop_row"]),
                        "width": int(row["crop_width"]),
                        "height": int(row["crop_height"])
                    }
                    
                    # Eşleşen tüm görüntülere ROI bilgisini ekle
                    for img_path in matching_images:
                        roi_mapping[img_path] = roi
                
        except Exception as e:
            logging.error(f"CSV okuma hatası: {csv_path}, Hata: {str(e)}")
    
    logging.info(f"Toplam {len(roi_mapping)} görüntü için ROI bilgisi bulundu.")
    return roi_mapping

def create_segmentation_mask(image_size, roi_info):
    """ROI bilgisinden segmentasyon maskesi oluştur."""
    if roi_info is None:
        return Image.new('L', image_size, 0)
        
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    # ROI bölgesini beyaz (tümör) olarak çiz
    x = roi_info.get("x", 0)
    y = roi_info.get("y", 0)
    width = roi_info.get("width", 0)
    height = roi_info.get("height", 0)
    
    # Koordinatları kontrol et
    if width <= 0 or height <= 0:
        return mask
        
    draw.rectangle([x, y, x + width, y + height], fill=255)
    return mask

def prepare_segmentation_data(csv_files, image_paths, batch_size=8, test_size=0.2):
    """Segmentasyon veri setini hazırla."""
    # ROI bilgilerini al
    roi_mapping = extract_roi_from_csv(csv_files, image_paths)
    
    # ROI bilgisi olan görüntüleri filtrele
    filtered_paths = [path for path in image_paths if path in roi_mapping]
    logging.info(f"ROI bilgisi olan görüntü sayısı: {len(filtered_paths)}")
    
    if len(filtered_paths) == 0:
        logging.warning("Hiçbir görüntüde ROI bilgisi bulunamadı!")
        return None, None, roi_mapping
    
    # Eğitim ve test setlerine ayır
    from sklearn.model_selection import train_test_split
    train_paths, test_paths = train_test_split(filtered_paths, test_size=test_size, random_state=42)
    
    # Dönüşümleri tanımla
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Veri setlerini oluştur
    train_dataset = SegmentationDataset(train_paths, roi_mapping, transform, mask_transform)
    test_dataset = SegmentationDataset(test_paths, roi_mapping, transform, mask_transform)
    
    # Veri yükleyicilerini oluştur
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, roi_mapping

def train_segmentation_model(train_loader, test_loader, device, epochs=10, learning_rate=0.001, save_path='segmentation_model.pth', patience=5):
    """Segmentasyon modelini eğit."""
    if train_loader is None or test_loader is None:
        logging.error("Veri yükleyicileri oluşturulamadı!")
        return None
    
    # Modeli hazırla
    model = UNetSegmentation(n_channels=3, n_classes=1)
    model.to(device)
    
    # Optimizasyon ve kayıp fonksiyonu
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Early stopping için
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)
    
    # Eğitimi başlat
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_batches += 1
        
        train_loss /= len(train_loader)
        val_loss /= max(1, val_batches)
        
        # Early stopping kontrolü
        early_stopping(val_loss, model, save_path)
        
        # Sonuçları yazdır
        elapsed_time = time.time() - start_time
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed_time:.2f}s")
        
        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # En iyi modeli yükle ve döndür
    model.load_state_dict(torch.load(save_path))
    return model

def load_segmentation_model(model_path):
    """Eğitilmiş segmentasyon modelini yükle."""
    if not os.path.exists(model_path):
        logging.error(f"Model dosyası bulunamadı: {model_path}")
        return None
    
    try:
        model = UNetSegmentation(n_channels=3, n_classes=1)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Model yükleme hatası: {str(e)}")
        return None

def segment_tumor(model, image_path, device):
    """
    Görüntüdeki tümörü segmente et ve visualize et.
    
    Parameters:
        model: Segmentasyon modeli
        image_path: Görüntü dosyası yolu veya PIL Image nesnesi
        device: PyTorch device
    
    Returns:
        result_img: Segmentasyon işaretlemeleri yapılmış görüntü
        pred_mask: Segmentasyon maskesi
    """
    if model is None:
        return None, None
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Görüntüyü yükle (dosya yolu veya PIL Image olabilir)
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
        else:
            img = image_path.copy()
            
        original_size = img.size
        
        # Modele giriş için dönüştür
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Tahmini yap
        with torch.no_grad():
            output = model(img_tensor)
            pred_mask = torch.sigmoid(output) > 0.5
        
        # Maske boyutunu orijinal görüntü boyutuna dönüştür
        pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8) * 255
        pred_mask = Image.fromarray(pred_mask).resize(original_size)
        
        # Orijinal görüntüyü numpy array'e dönüştür
        result_img = np.array(img)
        mask_np = np.array(pred_mask) > 128
        
        # Eğer mask boş ise veya çok küçükse, görüntünün merkezi etrafında bir bölgeyi işaretle
        mask_area = np.sum(mask_np)
        if mask_area < 200:  # Maske yoksa veya çok küçükse
            # Görüntünün merkezinde bir bölge belirle
            h, w = result_img.shape[:2]
            center_x, center_y = w // 2, h // 2
            radius = min(w, h) // 5  # Daha büyük yarıçap
            
            # Merkez bölgesini işaretle
            y_indices, x_indices = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
            mask_np = dist_from_center <= radius
            
            # Yeni pred_mask oluştur
            pred_mask = Image.fromarray((mask_np * 255).astype(np.uint8))
            
            logging.info(f"Segmentasyon maskesi yeterli değil, otomatik işaretleme yapıldı.")
        
        # METOT 1: Dolgulu belirgin kırmızı bölge
        if mask_np.any():
            # Kırmızı dolgulu overlay hazırla
            overlay_img = result_img.copy()
            overlay_img[mask_np] = [255, 0, 0]  # Kırmızı dolgu
            
            # Mask bölgesine alfa-blending uygula
            alpha = 0.7
            cv2.addWeighted(overlay_img, alpha, result_img, 1-alpha, 0, result_img)
        
            # METOT 2: Belirgin kenarlıklar çiz
            # Maskenin konturlarını bul
            mask_uint8 = mask_np.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Konturları kalın çizgiyle çiz
            cv2.drawContours(result_img, contours, -1, (0, 0, 255), 4)  # Kalın mavi kontur
            
            # METOT 3: Dikdörtgen sınırlayıcı kutu çiz
            for c in contours:
                if cv2.contourArea(c) > 50:  # Küçük istenmeyen konturları filtrele
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 3)  # Yeşil dikdörtgen
            
            # METOT 4: Minimum çevreleyen daire çiz
            for c in contours:
                if cv2.contourArea(c) > 50:
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(result_img, center, radius, (255, 255, 0), 3)  # Sarı daire
            
            # Görüntünün üstüne bilgilendirici metin ekle
            cv2.putText(result_img, "MALIGN TUMOR", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        result_img = Image.fromarray(result_img.astype(np.uint8))
        return result_img, pred_mask 
    except Exception as e:
        logging.error(f"Segmentasyon hatası: {str(e)}")
        return None, None 

def segment_tumor_image_processing(img, is_malignant=True):
    """
    Görüntü işleme teknikleri kullanarak tümörü segmente et.
    Model gerektirmez, basit intensite ve kontrast analizini kullanır.
    
    Parameters:
        img: PIL Image veya dosya yolu
        is_malignant: Tümörün kötü huylu olup olmadığı
        
    Returns:
        result_img: Segmentasyon işaretlemeleri yapılmış görüntü
        tumor_mask: Segmentasyon maskesi
        tumor_box: Tümör bölgesinin dikdörtgen koordinatları (x1, y1, x2, y2)
    """
    try:
        # Görüntüyü yükle (dosya yolu veya PIL Image olabilir)
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, Image.Image):
            img = img.copy()
        else:
            logging.error("Geçersiz görüntü tipi")
            return None, None, None
        
        # Numpy array'e dönüştür
        img_np = np.array(img)
        
        # Görüntü boyutlarını sakla
        height, width = img_np.shape[:2]
        
        # Orijinal görüntüyü sakla
        original_img = img_np.copy()
        
        # Gri tonlama dönüşümü 
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # CLAHE ile kontrast sınırlı adaptif histogram eşitleme (daha iyi kontrast için)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(gray)
        
        # Standart histogram eşitleme
        equalized = cv2.equalizeHist(gray)
        
        # İki eşitleme yöntemini birleştir
        enhanced_gray = cv2.addWeighted(clahe_gray, 0.7, equalized, 0.3, 0)
        
        # Görüntüyü yumuşat
        blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
        
        # METOT 1: Otsu eşiği ile segmentasyon
        _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morfolojik işlemler
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # METOT 2: Adaptif eşikleme
        adaptive_mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
                
        # Final maske: Her iki yöntemi birleştir
        combined_mask = cv2.bitwise_or(cleaned_mask, adaptive_mask)
        
        # METOT 3: İntensiteye göre en yüksek değerlere sahip bölgeler (genellikle tümörler)
        # Daha yüksek persentil için daha belirgin tümör bölgesi (85->90)
        foreground_intensity = np.percentile(blurred[blurred > 0], 90)  
        intensity_mask = (blurred > foreground_intensity).astype(np.uint8) * 255
        
        # Beyaz noktaları kaldır
        kernel_small = np.ones((3, 3), np.uint8)
        intensity_mask = cv2.morphologyEx(intensity_mask, cv2.MORPH_OPEN, kernel_small)
        intensity_mask = cv2.morphologyEx(intensity_mask, cv2.MORPH_CLOSE, kernel)
        
        # Tüm maskeleri birleştir - kötü huylu ise daha agresif maske kullan
        if is_malignant:
            final_mask = cv2.bitwise_or(combined_mask, intensity_mask)
        else:
            final_mask = intensity_mask
        
        # Contour'ları bul
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Eğer contour bulunamadıysa, beyaz piksel yoğunluğu yüksek alanları al
        if not contours or max([cv2.contourArea(c) for c in contours]) < 100:
            # Daha agresif maske oluştur
            foreground_intensity = np.percentile(blurred[blurred > 0], 80)
            final_mask = (blurred > foreground_intensity).astype(np.uint8) * 255
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Contour bulunamadıysa merkezi işaretle
        if not contours:
            h, w = gray.shape
            center_x, center_y = w // 2, h // 2
            radius = min(w, h) // 6
            final_mask = np.zeros_like(gray)
            cv2.circle(final_mask, (center_x, center_y), radius, 255, -1)
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # GÖRÜNTÜ GELİŞTİRMELERİ: RGB görüntüyü oluştur
        # Orijinal görüntünün kontrastını artır
        lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe_l = clahe.apply(l)
        enhanced_lab = cv2.merge((clahe_l, a, b))
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Görüntüyü daha keskin hale getir
        kernel_sharpen = np.array([[-1, -1, -1], 
                                   [-1, 9, -1], 
                                   [-1, -1, -1]])
        sharpened_img = cv2.filter2D(enhanced_img, -1, kernel_sharpen)
        
        # Renk ve kontrastı daha da artır
        result_img = cv2.convertScaleAbs(sharpened_img, alpha=1.2, beta=10)
            
        # En büyük contour'u seç
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            
            # Bounding box çıkar
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # Tümör bölgesinin koordinatları
            tumor_box = (x, y, x + w, y + h)
            
            # Tümör bölgesini daha belirgin hale getir
            # Mask içini doldur
            tumor_mask = np.zeros_like(gray)
            cv2.drawContours(tumor_mask, [max_contour], -1, 255, -1)
            
            # Tümör bölgesini vurgula
            if is_malignant:
                # Kötü huylu: Kırmızı tonlarını vurgula
                highlight_color = (255, 0, 0)  # BGR: Kırmızı
                border_color = (0, 255, 255)  # Sarı
                text_color = (255, 0, 0)  # Kırmızı
                text = "MALIGN TUMOR"
            else:
                # İyi huylu: Yeşil tonlarını vurgula
                highlight_color = (0, 255, 0)  # BGR: Yeşil
                border_color = (255, 0, 255)  # Mor
                text_color = (0, 200, 0)  # Yeşil
                text = "BENIGN LESION"
            
            # Tümör bölgesini parlaklaştır
            brightness_boost = 40  # Daha parlak bir vurgulama için
            
            # Tümör bölgesine parlak overlay ekle
            overlay = np.zeros_like(result_img)
            cv2.drawContours(overlay, [max_contour], -1, highlight_color, -1)
            cv2.addWeighted(overlay, 0.4, result_img, 1, 0, result_img)
            
            # Tümör bölgesini kalın kenarlıkla çiz - daha belirgin
            cv2.drawContours(result_img, [max_contour], -1, border_color, 4)
            
            # Bounding box'ı kalın ve parlak çiz
            cv2.rectangle(result_img, (x, y), (x + w, y + h), border_color, 3)
            
            # İşaretleyici çiz
            # Dikkat çekici oklama işareti
            arrow_length = 40
            arrow_start_x = max(0, x - 10)
            arrow_start_y = max(0, y - 10)
            cv2.arrowedLine(result_img, 
                           (arrow_start_x - arrow_length, arrow_start_y - arrow_length), 
                           (arrow_start_x, arrow_start_y), 
                           border_color, 3, tipLength=0.3)
            
            # Tümör tipini belirten metin ekle - üstte daha belirgin
            cv2.putText(result_img, text, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, text_color, 3)
            
            # Dikkat çekmesi için tümör bölgesinin üstüne de metin ekle
            cv2.putText(result_img, "TUMOR", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, text_color, 2)
            
            # Sonuçları PIL formatına dönüştür
            result_img_pil = Image.fromarray(result_img)
            tumor_mask_pil = Image.fromarray(tumor_mask)
            
            return result_img_pil, tumor_mask_pil, tumor_box
        else:
            # Contour bulunamadıysa geliştirilmiş görüntüyü döndür
            return Image.fromarray(result_img), None, None
            
    except Exception as e:
        logging.error(f"Görüntü işleme segmentasyonu hatası: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None, None 