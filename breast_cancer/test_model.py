#!/usr/bin/env python
import torch
from goruntutahmin import get_model
import torch.nn as nn
from segmentasyon import UNetSegmentation, EarlyStopping

def test_model_architecture():
    """Model mimarisini test et"""
    print("ResNet50 model mimarisi test ediliyor...")
    model = get_model(num_classes=2)
    
    # Model yapısını kontrol et
    print(f"Model tipi: {type(model).__name__}")
    print(f"Tam bağlantılı katman: {model.fc}")
    
    # Eğitilebilir parametreleri kontrol et
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Toplam parametre sayısı: {total_params:,}")
    print(f"Eğitilebilir parametre sayısı: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Early stopping özelliğini test et
    early_stopping = EarlyStopping(patience=5, verbose=True)
    print(f"Early stopping: {early_stopping}")
    
    print("\nModel mimarisi doğru şekilde yüklendi.")
    return model

def test_segmentation_model():
    """Segmentasyon modelini test et"""
    print("U-Net segmentasyon modeli test ediliyor...")
    model = UNetSegmentation(n_channels=3, n_classes=1)
    
    # Model yapısını kontrol et
    print(f"Model tipi: {type(model).__name__}")
    
    # Parametre sayısını kontrol et
    params = sum(p.numel() for p in model.parameters())
    print(f"Toplam parametre sayısı: {params:,}")
    
    # Giriş-çıkış boyutunu test et
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Giriş boyutu: {dummy_input.shape}")
    print(f"Çıkış boyutu: {output.shape}")
    
    print("\nSegmentasyon modeli doğru şekilde yüklendi.")
    return model

if __name__ == "__main__":
    print("==================================")
    print("    Model Mimarisi Test Aracı     ")
    print("==================================")
    
    # ResNet50 sınıflandırma modeli testi
    classifier = test_model_architecture()
    
    print("\n---------------------------------\n")
    
    # U-Net segmentasyon modeli testi
    segmenter = test_segmentation_model()
    
    print("\n==================================")
    print("      Tüm testler başarılı!       ")
    print("==================================") 