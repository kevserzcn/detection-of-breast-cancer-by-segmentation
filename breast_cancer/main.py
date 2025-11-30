import os
import torch
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image
from goruntutahmin import parse_inbreast_csv, get_data_loaders, get_model, train_model, predict, predict_single_image
from arayuz import BreastCancerUI
from raportahmin import RandomForestModel
from sklearn.metrics import classification_report
import logging
import sys
from collections import Counter
import numpy as np

# Logging ayarları
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app_debug.log")
    ]
)

class App:
    def __init__(self, root):
        logging.debug("Initializing App...")
        self.root = root
        try:
            self.ui = BreastCancerUI(root)
            logging.debug("UI initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize UI: {str(e)}")
            raise

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.debug(f"Device: {self.device}")
        self.model = None

        # Veri seti ve model yolları
        self.inbreast_base_dir = os.path.join(os.path.expanduser("~"), "OneDrive", "Masaüstü", "archive(5)", "inbreast")
        self.inbreast_image_dir = os.path.join(self.inbreast_base_dir, "ALL-IMGS")
        self.inbreast_csv_path = os.path.join(self.inbreast_base_dir, "INbreast.csv")

        # CBIS-DDSM yolları
        self.cbis_base_dir = os.path.join(os.path.expanduser("~"), "OneDrive", "Masaüstü", "archive(3)")
        self.cbis_jpg_dir = os.path.join(self.cbis_base_dir, "jpg")
        self.cbis_csvs = [
            os.path.join(self.cbis_base_dir, "calc_case_description_train_set.csv"),
            os.path.join(self.cbis_base_dir, "calc_case_description_test_set.csv"),
            os.path.join(self.cbis_base_dir, "mass_case_description_train_set.csv"),
            os.path.join(self.cbis_base_dir, "mass_case_description_test_set.csv")
        ]
        self.model_path = os.path.join(self.inbreast_base_dir, "combined_model.pth")

        # Dosya yollarını doğrula
        self.validate_paths()

        logging.debug(f"INbreast base directory: {self.inbreast_base_dir}")
        logging.debug(f"CBIS-DDSM base directory: {self.cbis_base_dir}")
        logging.debug(f"CBIS-DDSM JPG directory: {self.cbis_jpg_dir}")
        logging.debug(f"CBIS-DDSM CSV files: {self.cbis_csvs}")

        # Arayüz butonlarına işlevleri bağla
        logging.debug("Setting callbacks...")
        try:
            self.ui.set_rf_callback(self.rf_predict)
            self.ui.set_cnn_train_callback(self.train_cnn)
            self.ui.set_upload_callback(self.upload_image)
            self.ui.set_cnn_predict_callback(self.cnn_predict)
            logging.debug("Callbacks set successfully.")
        except AttributeError as e:
            logging.error(f"Callback setup failed: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during callback setup: {str(e)}")
            raise

    def validate_paths(self):
        """Dosya yollarını doğrula ve hataları logla."""
        if not os.path.exists(self.inbreast_base_dir):
            logging.error(f"INbreast base directory bulunamadı: {self.inbreast_base_dir}")
        if not os.path.exists(self.inbreast_image_dir):
            logging.error(f"INbreast image directory bulunamadı: {self.inbreast_image_dir}")
        if not os.path.exists(self.inbreast_csv_path):
            logging.error(f"INbreast CSV dosyası bulunamadı: {self.inbreast_csv_path}")

        if not os.path.exists(self.cbis_base_dir):
            logging.error(f"CBIS-DDSM base directory bulunamadı: {self.cbis_base_dir}")
        if not os.path.exists(self.cbis_jpg_dir):
            logging.error(f"CBIS-DDSM JPG directory bulunamadı: {self.cbis_jpg_dir}")
        else:
            folder_count = len([f for f in os.listdir(self.cbis_jpg_dir) if os.path.isdir(os.path.join(self.cbis_jpg_dir, f))])
            logging.debug(f"CBIS-DDSM JPG dizininde {folder_count} klasör bulundu.")

        for csv_path in self.cbis_csvs:
            if not os.path.exists(csv_path):
                logging.error(f"CBIS-DDSM CSV dosyası bulunamadı: {csv_path}")
            else:
                logging.debug(f"CBIS-DDSM CSV dosyası bulundu: {csv_path}")
                # Ek kontrol: Okunabiliyor mu?
                try:
                    with open(csv_path, "r", encoding="utf-8") as f:
                        f.readline()
                    logging.debug(f"CBIS-DDSM CSV dosyası okunabiliyor: {csv_path}")
                except Exception as e:
                    logging.error(f"CBIS-DDSM CSV dosyası okunamıyor: {csv_path}, Hata: {str(e)}")

    def rf_predict(self):
        """Random Forest tahmini yap ve sonucu göster"""
        logging.debug("Running RF prediction...")
        features = self.ui.get_feature_values()
        if features is None:
            logging.warning("Feature values are None.")
            return

        try:
            rf_model = RandomForestModel()
            tahmin, olasilik, evre = rf_model.predict(features)
            result = "Kötü Huylu" if tahmin == 1 else "İyi Huylu"

            self.ui.display_prediction_graph([olasilik[0], olasilik[1]], evre)
            messagebox.showinfo("RF Tahmin Sonucu", f"Tahmin: {result}\n"
                                                  f"Klinik Evre: {evre}")
            messagebox.showinfo("Model Performansı", rf_model.performans_raporu)

        except Exception as e:
            logging.error(f"Random Forest tahmini sırasında hata: {str(e)}")
            messagebox.showerror("Hata", f"Random Forest tahmini sırasında hata: {str(e)}")

    def train_cnn(self):
        """CNN modelini eğit ve kaydet"""
        logging.debug("Starting CNN training...")
        progress_window, progress_label, progress_bar = self.ui.create_progress_window("CNN Eğitimi", "Model eğitiliyor...")
        
        # Early stopping flag
        self.stop_training = False
        
        # Add stop button
        stop_button = tk.Button(
            progress_window, 
            text="Eğitimi Durdur", 
            bg="#ff4d4d", 
            fg="white", 
            font=("Segoe UI", 10, "bold"),
            command=lambda: self._set_stop_flag()
        )
        stop_button.pack(pady=10)

        try:
            progress_label.config(text="Veriler yükleniyor...")
            self.root.update()
            if not os.path.exists(self.inbreast_csv_path) or not os.path.exists(self.inbreast_image_dir):
                raise FileNotFoundError(f"INbreast veri seti bulunamadı: {self.inbreast_csv_path} veya {self.inbreast_image_dir}")
            for csv in self.cbis_csvs:
                if not os.path.exists(csv):
                    logging.warning(f"CBIS-DDSM CSV bulunamadı, atlanıyor: {csv}")

            progress_label.config(text="Veri yükleyiciler hazırlanıyor...")
            self.root.update()
            train_loader, test_loader, class_weights = get_data_loaders(
                self.inbreast_csv_path, self.inbreast_image_dir,
                self.cbis_csvs, self.cbis_jpg_dir, self.cbis_base_dir, batch_size=8,
                max_cbis_images=0  # Use all images
            )

            progress_label.config(text="Model yükleniyor...")
            self.root.update()
            self.model = get_model(num_classes=2)
            self.model.to(self.device)

            progress_label.config(text="Eğitim başlıyor...")
            progress_bar['maximum'] = 20
            for epoch in range(20):
                if self.stop_training:
                    logging.info("Eğitim kullanıcı tarafından durduruldu.")
                    progress_label.config(text="Eğitim durduruldu!")
                    self.root.update()
                    break
                    
                self.model = train_model(self.model, train_loader, test_loader, self.device, class_weights, epochs=1)
                progress_bar['value'] = epoch + 1
                progress_label.config(text=f"Epoch {epoch+1}/20 tamamlandı")
                self.root.update()

            progress_label.config(text="Test ediliyor...")
            self.root.update()
            preds, true_labels, probs = predict(self.model, test_loader, self.device)
            report = classification_report(true_labels, preds, target_names=["Benign", "Malignant"])
            logging.info("Sınıflandırma Raporu:\n" + report)
            print("\nSınıflandırma Raporu:")
            print(report)
            logging.info(f"Test olasılık dağılımı: {Counter([np.argmax(p) for p in probs])}")

            torch.save(self.model.state_dict(), self.model_path)
            logging.info(f"Model kaydedildi: {self.model_path}")
            self.ui.enable_cnn_prediction()
            messagebox.showinfo("Başarılı", "CNN modeli başarıyla eğitildi ve kaydedildi!\n"
                                          f"Sınıflandırma Raporu:\n{report}")

        except Exception as e:
            logging.error(f"Eğitim hatası: {str(e)}")
            messagebox.showerror("Hata", f"Model eğitilirken bir hata oluştu: {str(e)}")
        finally:
            progress_window.destroy()
            
    def _set_stop_flag(self):
        """Set the flag to stop training"""
        self.stop_training = True
        logging.info("Eğitimi durdurma talebi alındı.")

    def upload_image(self):
        """DICOM, JPG veya PNG görüntüsünü yükle ve göster"""
        logging.debug("Uploading image...")
        file_path = filedialog.askopenfilename(
            initialdir=self.cbis_jpg_dir if os.path.exists(self.cbis_jpg_dir) else self.inbreast_image_dir,
            title="Görüntü Dosyası Seç",
            filetypes=[("Görüntü dosyaları", "*.dcm *.jpg *.jpeg *.png"), ("DICOM files", "*.dcm"),
                      ("JPG files", "*.jpg *.jpeg"), ("PNG files", "*.png")]
        )
        if file_path:
            logging.info(f"Seçilen dosya: {file_path}")
            if not os.path.exists(file_path):
                messagebox.showerror("Hata", f"Dosya bulunamadı: {file_path}")
                return
            try:
                if self.ui.display_image(file_path):
                    self.ui.loaded_image = file_path
                    self.ui.enable_cnn_prediction()
            except Exception as e:
                messagebox.showerror("Hata", f"Görüntü yüklenirken hata: {str(e)}")

    def cnn_predict(self):
        """Yüklenen görüntü üzerinde CNN tahmini yap"""
        logging.debug("Running CNN prediction...")
        if not self.ui.loaded_image:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü dosyası yükleyin!")
            return

        if self.model is None:
            try:
                self.model = get_model(num_classes=2)
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_path}")
                self.model.load_state_dict(torch.load(self.model_path))
                self.model.to(self.device)
            except Exception as e:
                logging.error(f"Model yüklenirken hata: {str(e)}")
                messagebox.showerror("Hata", f"Model yüklenirken hata: {str(e)}")
                return

        try:
            # Segmentasyon modeli yolu
            seg_model_path = os.path.join(os.path.dirname(self.model_path), "segmentation_model.pth")
            segmentation_model = None
            
            # Eğer segmentasyon modeli varsa yükle
            if os.path.exists(seg_model_path):
                from segmentasyon import load_segmentation_model, segment_tumor, cv2
                segmentation_model = load_segmentation_model(seg_model_path)
                segmentation_model.to(self.device)
                logging.info(f"Segmentasyon modeli yüklendi: {seg_model_path}")
            else:
                logging.warning(f"Segmentasyon modeli bulunamadı: {seg_model_path}")
            
            # Tahmin yap ve tüm değerleri al
            pred, probabilities, segmented_img, advice, tumor_coords, original_img = predict_single_image(
                self.model, self.ui.loaded_image, self.device, segmentation_model)
                
            if pred is None:
                raise ValueError("Görüntü işlenirken hata oluştu.")

            result = "Kötü Huylu (Malignant)" if pred == 1 else "İyi Huylu (Benign)"
            prob_malignant = probabilities[1] * 100
            prob_benign = probabilities[0] * 100

            tumor_size = float(self.ui.entries[0].get()) if self.ui.entries[0].get() else 10.0
            evre = self._estimate_stage(tumor_size, pred)

            # Tahmin grafiğini göster
            self.ui.display_prediction_graph([prob_benign / 100, prob_malignant / 100], evre)
            
            # Ayrıca pasta grafiğini ayrı pencerede göster
            self.ui.display_image_prediction_result([prob_benign / 100, prob_malignant / 100])
            
            # Segmentasyon görüntüsünü göster
            if original_img is not None:
                try:
                    # Her zaman görüntü işleme tabanlı segmentasyonu dene (daha belirgin görüntüler için)
                    from segmentasyon import segment_tumor_image_processing
                    processed_img, tumor_mask, processed_tumor_coords = segment_tumor_image_processing(original_img, is_malignant=(pred==1))
                    
                    if processed_tumor_coords:
                        # Eğer görüntü işleme ile bulduysa bu koordinatları kullan
                        tumor_coords = processed_tumor_coords
                        logging.info(f"Geliştirilmiş segmentasyon ile tümör bölgesi: {tumor_coords}")
                    
                    # İşlenmiş (kontrast arttırılmış) görüntüyü kullan
                    if processed_img is not None and tumor_coords:
                        self.ui.display_segmentation(processed_img, tumor_coords)
                        logging.info("Geliştirilmiş kontrast ile segmentasyon görüntüsü gösterildi")
                    elif tumor_coords:
                        # İşlenmiş görüntü yoksa ama tümör koordinatları varsa
                        self.ui.display_segmentation(original_img, tumor_coords)
                        logging.info("Normal segmentasyon görüntüsü gösterildi")
                    else:
                        # Koordinat yoksa veya iyi huylu ise açıklayıcı mesaj göster
                        message = "İyi huylu olduğu için sadece muhtemel lezyon bölgesi gösteriliyor."
                        if pred == 1:
                            message = "Tümör bölgesi tespit edilemedi. Lütfen farklı bir görüntü deneyin."
                        self.ui.display_segmentation(processed_img or original_img, error_message=message)
                        logging.info("Segmentasyon gösterimi tamamlandı (tümör tespit edilemedi).")
                
                except Exception as e:
                    # Eğer geliştirilmiş görüntü işleme başarısız olursa, normal yönteme geri dön
                    logging.error(f"Geliştirilmiş segmentasyon hatası: {str(e)}")
                    # Tümör koordinatları varsa göster
                    if tumor_coords and pred == 1:  # Kötü huylu ve tümör tespit edildi
                        self.ui.display_segmentation(original_img, tumor_coords)
                        logging.info(f"Segmentasyon görüntüsü başarıyla gösterildi. Tümör koordinatları: {tumor_coords}")
                    else:
                        # Koordinat yoksa veya iyi huylu ise açıklayıcı mesaj göster
                        message = "İyi huylu olduğu için sadece muhtemel lezyon bölgesi gösteriliyor."
                        if pred == 1:
                            message = "Tümör bölgesi tespit edilemedi. Lütfen farklı bir görüntü deneyin."
                        self.ui.display_segmentation(original_img, error_message=message)
                        logging.info("Segmentasyon gösterimi tamamlandı (tümör tespit edilemedi).")
            else:
                logging.warning("Orijinal görüntü bulunamadı, segmentasyon gösterilemiyor.")
                self.ui.display_segmentation(error_message="Görüntü işlenirken hata oluştu.")
            
            # Sonucu göster
            message = f"Sonuç: {result}\n" \
                      f"Klinik Evre (Tahmini): {evre}"
                      
            # Tıbbi tavsiye varsa ekle
            if advice:
                message += "\n\nTIBBİ DEĞERLENDİRME:\n" + advice.strip()
                
            messagebox.showinfo("CNN Tahmin Sonucu", message)

        except Exception as e:
            logging.error(f"CNN tahmini sırasında hata: {str(e)}")
            messagebox.showerror("Hata", f"CNN tahmini sırasında hata: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

    def _estimate_stage(self, tumor_size, pred):
        """Basit bir evre tahmini yap"""
        if pred == 0:
            return "Evre 0 - İyi huylu tümör"
        if tumor_size <= 20:
            return "Evre 1 - Erken evre, küçük tümör"
        elif tumor_size <= 50:
            return "Evre 2 - Orta büyüklükte tümör"
        else:
            return "Evre 3 - Büyük tümör"

def main():
    logging.debug("Starting application...")
    root = tk.Tk()
    try:
        app = App(root)
        logging.debug("App initialized, starting UI...")
        app.ui.run()
        logging.debug("Application closed.")
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()