#!/usr/bin/env python
import os
import argparse
import torch
import logging
from goruntutahmin import (
    get_data_loaders, get_model, train_model, 
    parse_cbis_ddsm, predict
)
from segmentasyon import (
    prepare_segmentation_data, train_segmentation_model
)

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Meme kanseri modeli eğitim aracı')
    parser.add_argument('--mode', choices=['classification', 'segmentation'], default='classification',
                      help='Eğitilecek model türü: classification veya segmentation')
    parser.add_argument('--model_path', type=str, default='model_resnet50.pth',
                      help='Eğitilmiş model kaydedilecek dosya yolu')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Eğitim döngüsü sayısı')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch boyutu')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Öğrenme oranı (segmentasyon için)')
    parser.add_argument('--patience', type=int, default=5,
                      help='Early stopping için sabır değeri')
    parser.add_argument('--max_images', type=int, default=500,
                      help='Kullanılacak maksimum görüntü sayısı (0 = tüm görüntüler)')
    parser.add_argument('--cbis_dir', type=str, default=None,
                      help='CBIS-DDSM veri seti dizini')
    parser.add_argument('--inbreast_dir', type=str, default=None,
                      help='INbreast veri seti dizini')
    parser.add_argument('--inbreast_csv', type=str, default=None,
                      help='INbreast CSV dosyası yolu')
    parser.add_argument('--use_all_images', action='store_true',
                      help='Tüm görüntüleri kullanmak için bu bayrağı ayarlayın (--max_images=0 ile aynı)')
    
    args = parser.parse_args()
    
    # Use all images if flag is set
    if args.use_all_images:
        args.max_images = 0
    
    # Cihazı belirle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Varsayılan veri seti yolları
    if args.cbis_dir is None:
        args.cbis_dir = os.path.join(os.path.expanduser("~"), "OneDrive", "Masaüstü", "archive(3)")
    
    cbis_jpg_dir = os.path.join(args.cbis_dir, "jpg")
    cbis_csvs = [
        os.path.join(args.cbis_dir, "calc_case_description_train_set.csv"),
        os.path.join(args.cbis_dir, "calc_case_description_test_set.csv"),
        os.path.join(args.cbis_dir, "mass_case_description_train_set.csv"),
        os.path.join(args.cbis_dir, "mass_case_description_test_set.csv")
    ]
    
    if args.inbreast_dir is None and args.mode == 'classification':
        args.inbreast_dir = os.path.join(os.path.expanduser("~"), "OneDrive", "Masaüstü", "archive(5)", "inbreast")
        args.inbreast_csv = os.path.join(args.inbreast_dir, "INbreast.csv")
    
    # Veri setlerinin varlığını kontrol et
    if args.mode == 'classification' and (not os.path.exists(args.inbreast_dir) or not os.path.exists(args.inbreast_csv)):
        print(f"Uyarı: INbreast veri seti bulunamadı. Sadece CBIS-DDSM veri seti kullanılacak.")
        args.inbreast_dir = None
        args.inbreast_csv = None
    
    if not os.path.exists(cbis_jpg_dir):
        print(f"Hata: CBIS-DDSM veri seti bulunamadı: {cbis_jpg_dir}")
        return
    
    # Sınıflandırma modeli eğitimi
    if args.mode == 'classification':
        print(f"Sınıflandırma modeli eğitimi başlatılıyor...")
        print(f"- Epochs: {args.epochs}")
        print(f"- Batch size: {args.batch_size}")
        print(f"- Early stopping patience: {args.patience}")
        print(f"- Max images: {args.max_images} {'(tüm görüntüler)' if args.max_images == 0 else ''}")
        
        # Veri yükleyicileri oluştur
        train_loader, test_loader, class_weights = get_data_loaders(
            args.inbreast_csv, args.inbreast_dir, 
            cbis_csvs, cbis_jpg_dir, args.cbis_dir,
            batch_size=args.batch_size, max_cbis_images=args.max_images
        )
        
        # Modeli oluştur ve eğit
        model = get_model(num_classes=2)
        model = train_model(
            model, train_loader, test_loader, device, class_weights,
            epochs=args.epochs, patience=args.patience, save_path=args.model_path
        )
        
        # Test et
        preds, true_labels, probs = predict(model, test_loader, device)
        
        print(f"Eğitim tamamlandı. Model kaydedildi: {args.model_path}")
    
    # Segmentasyon modeli eğitimi
    elif args.mode == 'segmentation':
        print(f"Segmentasyon modeli eğitimi başlatılıyor...")
        print(f"- Epochs: {args.epochs}")
        print(f"- Batch size: {args.batch_size}")
        print(f"- Learning rate: {args.learning_rate}")
        print(f"- Early stopping patience: {args.patience}")
        print(f"- Max images: {args.max_images} {'(tüm görüntüler)' if args.max_images == 0 else ''}")
        
        # Veri setini yükle
        cbis_paths, _ = parse_cbis_ddsm(cbis_csvs, cbis_jpg_dir, args.cbis_dir, max_images=args.max_images)
        
        if len(cbis_paths) > 0:
            # Segmentasyon veri setini hazırla
            train_loader, test_loader, roi_mapping = prepare_segmentation_data(
                cbis_csvs, cbis_paths, batch_size=args.batch_size
            )
            
            # ROI bilgisi olan görüntülerin oranını göster
            roi_count = sum(1 for path in cbis_paths if path in roi_mapping)
            print(f"ROI bilgisi bulunan görüntü oranı: {roi_count}/{len(cbis_paths)} ({roi_count/len(cbis_paths)*100:.1f}%)")
            
            # Segmentasyon modelini eğit
            segmentation_model = train_segmentation_model(
                train_loader, test_loader, device,
                epochs=args.epochs, learning_rate=args.learning_rate,
                save_path=args.model_path, patience=args.patience
            )
            
            print(f"Eğitim tamamlandı. Model kaydedildi: {args.model_path}")
        else:
            print("Veri seti yüklenemedi. Segmentasyon modeli eğitilemedi.")

if __name__ == "__main__":
    main() 