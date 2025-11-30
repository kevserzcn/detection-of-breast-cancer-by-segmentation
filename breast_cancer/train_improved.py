#!/usr/bin/env python
import os
import argparse
import torch
import logging
import time
import numpy as np
from goruntutahmin import (
    get_data_loaders, get_model, train_model, 
    parse_cbis_ddsm, predict
)
from segmentasyon import (
    prepare_segmentation_data, train_segmentation_model, 
    load_segmentation_model, UNetSegmentation
)

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_classification(args):
    """Sınıflandırma modelini eğit"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Cihaz: {device}")
    
    # Veri yükleyicileri oluştur
    train_loader, test_loader, class_weights = get_data_loaders(
        args.inbreast_csv, args.inbreast_dir,
        args.cbis_csvs, args.cbis_jpg_dir, args.cbis_base_dir,
        batch_size=args.batch_size, max_cbis_images=args.max_images
    )
    
    # Model oluştur
    model = get_model(num_classes=2)
    
    # Eğitim zamanlaması
    start_time = time.time()
    
    # Modeli eğit (early stopping özelliği ile)
    model = train_model(
        model, train_loader, test_loader, device, class_weights,
        epochs=args.epochs, patience=args.patience, save_path=args.model_path
    )
    
    training_time = time.time() - start_time
    logging.info(f"Eğitim tamamlandı. Toplam süre: {training_time/60:.2f} dakika")
    
    # Test et
    model.eval()
    preds, true_labels, probs = predict(model, test_loader, device)
    
    return model

def train_segmentation(args):
    """Segmentasyon modelini eğit"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Cihaz: {device}")
    
    # Veri setini yükle
    cbis_paths, _ = parse_cbis_ddsm(
        args.cbis_csvs, args.cbis_jpg_dir, args.cbis_base_dir, 
        max_images=args.max_images
    )
    
    if len(cbis_paths) > 0:
        # Segmentasyon veri setini hazırla
        train_loader, test_loader, roi_mapping = prepare_segmentation_data(
            args.cbis_csvs, cbis_paths, batch_size=args.batch_size
        )
        
        if train_loader is None or test_loader is None:
            logging.error("Segmentasyon veri seti oluşturulamadı.")
            return None
        
        # ROI bilgisi olan görüntülerin oranını göster
        roi_count = sum(1 for path in cbis_paths if path in roi_mapping)
        logging.info(f"ROI bilgisi bulunan görüntü oranı: {roi_count}/{len(cbis_paths)} ({roi_count/len(cbis_paths)*100:.1f}%)")
        
        # Segmentasyon modelini eğit
        start_time = time.time()
        
        segmentation_model = train_segmentation_model(
            train_loader, test_loader, device,
            epochs=args.epochs, learning_rate=args.learning_rate,
            save_path=args.model_path, patience=args.patience
        )
        
        training_time = time.time() - start_time
        logging.info(f"Eğitim tamamlandı. Toplam süre: {training_time/60:.2f} dakika")
        logging.info(f"Model kaydedildi: {args.model_path}")
        
        return segmentation_model
    else:
        logging.error("Veri seti yüklenemedi.")
        return None

def main():
    parser = argparse.ArgumentParser(description='Geliştirilmiş Meme Kanseri Modeli Eğitim Aracı')
    parser.add_argument('--mode', choices=['classification', 'segmentation'], default='classification',
                      help='Eğitilecek model türü: classification veya segmentation')
    parser.add_argument('--model_path', type=str, default='model_resnet50_improved.pth',
                      help='Eğitilmiş model kaydedilecek dosya yolu')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Maksimum eğitim döngüsü sayısı')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch boyutu')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                      help='Öğrenme oranı (segmentasyon için)')
    parser.add_argument('--patience', type=int, default=5,
                      help='Early stopping için sabır değeri')
    parser.add_argument('--max_images', type=int, default=3000,
                      help='Kullanılacak maksimum görüntü sayısı (0 = tüm görüntüler)')
    
    # Dosya yolları
    default_cbis_dir = os.path.join(os.path.expanduser("~"), "OneDrive", "Masaüstü", "archive(3)")
    default_inbreast_dir = os.path.join(os.path.expanduser("~"), "OneDrive", "Masaüstü", "archive(5)", "inbreast")
    
    parser.add_argument('--cbis_dir', type=str, default=default_cbis_dir,
                      help='CBIS-DDSM veri seti dizini')
    parser.add_argument('--inbreast_dir', type=str, default=default_inbreast_dir,
                      help='INbreast veri seti dizini')
    parser.add_argument('--inbreast_csv', type=str, 
                      default=os.path.join(default_inbreast_dir, "INbreast.csv"),
                      help='INbreast CSV dosyası yolu')
    
    args = parser.parse_args()
    
    # CBIS-DDSM dosya yolları
    args.cbis_jpg_dir = os.path.join(args.cbis_dir, "jpg")
    args.cbis_csvs = [
        os.path.join(args.cbis_dir, "calc_case_description_train_set.csv"),
        os.path.join(args.cbis_dir, "calc_case_description_test_set.csv"),
        os.path.join(args.cbis_dir, "mass_case_description_train_set.csv"),
        os.path.join(args.cbis_dir, "mass_case_description_test_set.csv")
    ]
    
    # Veri setlerinin varlığını kontrol et
    if not os.path.exists(args.cbis_jpg_dir):
        logging.error(f"CBIS-DDSM jpg dizini bulunamadı: {args.cbis_jpg_dir}")
        return
    
    # Kullanıcıya bilgi ver
    logging.info(f"Eğitim modu: {args.mode}")
    logging.info(f"Maksimum epoch sayısı: {args.epochs}")
    logging.info(f"Batch boyutu: {args.batch_size}")
    logging.info(f"Early stopping sabır değeri: {args.patience}")
    logging.info(f"Veri limiti: {args.max_images if args.max_images > 0 else 'Sınırsız'} görüntü")
    logging.info(f"Model kaydedilecek: {args.model_path}")
    
    # Eğitimi başlat
    if args.mode == 'classification':
        model = train_classification(args)
    else:
        model = train_segmentation(args)
    
    if model:
        logging.info("Eğitim başarıyla tamamlandı!")
    else:
        logging.error("Eğitim sırasında bir hata oluştu.")

if __name__ == "__main__":
    main() 