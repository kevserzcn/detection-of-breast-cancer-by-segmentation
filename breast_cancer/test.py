import pandas as pd
import os

# CSV dosyalarını oku
csv_files = [
    "C:/Users/Asus/OneDrive/Masaüstü/archive(3)/calc_case_description_train_set.csv",
    "C:/Users/Asus/OneDrive/Masaüstü/archive(3)/calc_case_description_test_set.csv",
    "C:/Users/Asus/OneDrive/Masaüstü/archive(3)/mass_case_description_train_set.csv",
    "C:/Users/Asus/OneDrive/Masaüstü/archive(3)/mass_case_description_test_set.csv",
    "C:/Users/Asus/OneDrive/Masaüstü/archive(3)/dicom_info.csv"
]

jpg_dir = "C:/Users/Asus/OneDrive/Masaüstü/archive(3)/jpg"

# CSV dosyalarını kontrol et
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    print(f"Dosya: {csv_file}")
    print("Sütunlar:", df.columns.tolist())
    print("Eksik değerler:\n", df[['image file path', 'ROI mask file path', 'pathology']].isna().sum())
    print("Örnek image file path:", df['image file path'].head(5).tolist())
    print("Örnek ROI mask file path:", df['ROI mask file path'].head(5).tolist() if 'ROI mask file path' in df.columns else "Sütun bulunamadı")
    print("-" * 50)

# JPG dosyalarını listele
jpg_files = []
for root, _, files in os.walk(jpg_dir):
    for file in files:
        if file.endswith('.jpg'):
            jpg_files.append(os.path.join(root, file))
print(f"Toplam JPG dosyası: {len(jpg_files)}")
print("Örnek JPG yolları:", jpg_files[:5])