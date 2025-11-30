import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import logging

class BreastCancerUI:
    def __init__(self, root):
        """Initialize the breast cancer prediction UI"""
        self.root = root
        self.root.title("💖 Meme Kanseri Tahmin Aracı")
        
        # Ekran boyutunu al
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Arayüz için uygun bir boyut belirle (ekranın %80'i)
        app_width = min(int(screen_width * 0.8), 1200)
        app_height = min(int(screen_height * 0.8), 700)
        
        # Pencereyi ekranın ortasına yerleştir
        x_position = (screen_width - app_width) // 2
        y_position = (screen_height - app_height) // 2
        
        # Pencere boyutunu ve pozisyonunu ayarla
        self.root.geometry(f"{app_width}x{app_height}+{x_position}+{y_position}")
        
        # Kadınlara yönelik modern tema renkleri - daha yumuşak ve zarif
        self.colors = {
            "bg_primary": "#fdf4f9",      # Çok açık pembe arka plan
            "bg_secondary": "#fcebf5",    # Biraz daha koyu pembe arka plan
            "accent": "#e75a97",          # Orta ton pembe vurgu
            "accent_light": "#f7a9cc",    # Açık pembe vurgu
            "text_primary": "#65385d",    # Koyu mor tonda metin
            "text_secondary": "#9c6b9a",  # Orta mor tonda metin
            "button_primary": "#de6f9f",  # Ana düğme rengi (pembe)
            "button_secondary": "#b162a8"  # İkincil düğme rengi (mor)
        }
        
        # Başlık ve düğme fontları
        self.fonts = {
            "header": ("Segoe UI", 16, "bold"),
            "subheader": ("Segoe UI", 14, "bold"),
            "button": ("Segoe UI", 11, "bold"),
            "label": ("Segoe UI", 10),
            "small": ("Segoe UI", 9)
        }
        
        # Arayüz stillendirmesi
        self.root.config(bg=self.colors["bg_primary"], padx=20, pady=20)
        
        # Global variables
        self.canvas = None
        self.image_display = None
        self.segmentation_display = None
        self.loaded_image = None
        self.entries = []
        
        # Tüm arayüz içeriğini bir ana çerçeveye yerleştir (scroll desteği için)
        self.main_frame = tk.Frame(self.root, bg=self.colors["bg_primary"])
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Çerçeveyi iki sütuna böl
        self.left_frame = tk.Frame(self.main_frame, bg=self.colors["bg_primary"])
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.right_frame = tk.Frame(self.main_frame, bg=self.colors["bg_primary"])
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Build the UI
        self._create_header()
        self._create_feature_inputs()
        self._create_rf_prediction_button()
        self._create_cnn_section()
        self._create_graph_frame()
    
    def _create_header(self):
        """Create the header section of the UI"""
        header_frame = tk.Frame(self.main_frame, bg=self.colors["bg_primary"])
        header_frame.pack(pady=(0, 15), fill=tk.X)
        
        # Logo/İkon (kalp emoji) ve başlık yan yana kompakt şekilde
        header_content = tk.Frame(header_frame, bg=self.colors["bg_primary"])
        header_content.pack(pady=(0, 0), fill=tk.X, expand=True)
        
        # Logo (kalp emoji) - daha büyük
        logo_label = tk.Label(
            header_content,
            text="💖",
            font=("Segoe UI", 28),
            bg=self.colors["bg_primary"],
            fg=self.colors["accent"]
        )
        logo_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # Başlık metni - daha kompakt ve tek satır
        title_label = tk.Label(
            header_content,
            text="MEME KANSERİ ERKEN TEŞHİS SİSTEMİ",
            font=self.fonts["header"],
            bg=self.colors["bg_primary"],
            fg=self.colors["accent"]
        )
        title_label.pack(side=tk.LEFT, fill=tk.X)
    
    def _create_feature_inputs(self):
        """Create input fields for the features"""
        # Sol çerçeve içinde özellik girişleri
        feature_frame = tk.LabelFrame(
            self.left_frame, 
            text="Hasta Verileri", 
            font=self.fonts["subheader"], 
            bg=self.colors["bg_primary"],
            fg=self.colors["accent"],
            padx=15, pady=15
        )
        feature_frame.pack(fill=tk.X, pady=10)
        
        features = [
            "Tumor Boyutu (mm)", 
            "Metastatik Lenf Nodu Sayısı", 
            "Nükleer Grade", 
            "Östrojen Reseptörü (0/1)", 
            "Progesteron Reseptörü (0/1)",
            "Yaş"
        ]
        
        # Her özellik için tip ve minimum/maksimum değerler
        feature_info = {
            "Tumor Boyutu (mm)": {"tip": "sayı", "min": 0, "max": 100, "default": "15"},
            "Metastatik Lenf Nodu Sayısı": {"tip": "sayı", "min": 0, "max": 20, "default": "0"},
            "Nükleer Grade": {"tip": "seçim", "seçenekler": ["1", "2", "3"], "default": "2"},
            "Östrojen Reseptörü (0/1)": {"tip": "seçim", "seçenekler": ["0", "1"], "default": "1"},
            "Progesteron Reseptörü (0/1)": {"tip": "seçim", "seçenekler": ["0", "1"], "default": "1"},
            "Yaş": {"tip": "sayı", "min": 18, "max": 100, "default": "45"}
        }
        
        # İki sütun oluştur ve özellikleri dengelice dağıt
        left_col = tk.Frame(feature_frame, bg=self.colors["bg_primary"])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        right_col = tk.Frame(feature_frame, bg=self.colors["bg_primary"])
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        for i, feature in enumerate(features):
            # Sol/sağ sütunu belirle
            container = left_col if i < 3 else right_col
            
            frame = tk.Frame(container, bg=self.colors["bg_primary"])
            frame.pack(pady=8, fill=tk.X)
            
            label = tk.Label(
                frame, 
                text=feature, 
                font=self.fonts["label"], 
                bg=self.colors["bg_primary"], 
                fg=self.colors["text_primary"],
                width=22,
                anchor="w"
            )
            label.pack(side=tk.LEFT)
            
            info = feature_info[feature]
            
            if info["tip"] == "seçim":
                var = tk.StringVar(value=info["default"])
                entry = ttk.Combobox(frame, values=info["seçenekler"], textvariable=var, 
                                     state="readonly", width=10, font=self.fonts["small"])
                entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            else:
                var = tk.StringVar(value=info["default"])
                entry = tk.Entry(frame, font=self.fonts["label"], width=10, textvariable=var)
                entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
                
                # Bilgi etiketi
                info_text = f"({info['min']}-{info['max']})"
                info_label = tk.Label(
                    frame, 
                    text=info_text, 
                    font=self.fonts["small"], 
                    bg=self.colors["bg_primary"], 
                    fg=self.colors["text_secondary"]
                )
                info_label.pack(side=tk.LEFT)
            
            self.entries.append(entry)
    
    def _create_rf_prediction_button(self):
        """Create the Random Forest prediction button"""
        # Tahmin düğmesini bir çerçeve içine yerleştir
        rf_frame = tk.Frame(self.left_frame, bg=self.colors["bg_primary"])
        rf_frame.pack(pady=15, fill=tk.X)
        
        # Tahmin düğmesi
        self.rf_button = tk.Button(
            rf_frame, 
            text="💗 Klinik Verilere Göre Tahmin Et", 
            bg=self.colors["button_primary"], 
            fg="white", 
            font=self.fonts["button"], 
            relief="raised",
            padx=15,
            pady=8,
            cursor="hand2",
            activebackground="#ff3399",
            activeforeground="white"
        )
        self.rf_button.pack(fill=tk.X)
    
    def _create_cnn_section(self):
        """Create the CNN model section"""
        # CNN bölümü için etiketli çerçeve
        cnn_frame = tk.LabelFrame(
            self.right_frame, 
            text="Görüntü Analizi", 
            font=self.fonts["subheader"], 
            bg=self.colors["bg_primary"],
            fg=self.colors["accent"],
            padx=15, pady=15
        )
        cnn_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # CNN düğmeleri çerçevesi
        buttons_frame = tk.Frame(cnn_frame, bg=self.colors["bg_primary"])
        buttons_frame.pack(pady=10, fill=tk.X)
        
        # CNN Model Eğit düğmesi
        self.cnn_train_button = tk.Button(
            buttons_frame, 
            text="🧠 CNN Modeli Eğit", 
            bg="#9c27b0", 
            fg="white", 
            font=self.fonts["button"], 
            relief="raised",
            padx=15,
            pady=8,
            cursor="hand2",
            activebackground="#7b1fa2",
            activeforeground="white"
        )
        self.cnn_train_button.pack(side=tk.LEFT, padx=(0, 5), expand=True, fill=tk.X)
        
        # Görüntü yükleme düğmesi
        self.upload_button = tk.Button(
            buttons_frame, 
            text="📂 Görüntü Yükle", 
            bg="#03a9f4", 
            fg="white", 
            font=self.fonts["button"], 
            relief="raised",
            padx=15,
            pady=8,
            cursor="hand2",
            activebackground="#0288d1",
            activeforeground="white"
        )
        self.upload_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # CNN tahmin düğmesi
        self.cnn_predict_button = tk.Button(
            buttons_frame, 
            text="🔍 CNN ile Tahmin Et", 
            bg="#4caf50", 
            fg="white", 
            font=self.fonts["button"], 
            relief="raised",
            padx=15,
            pady=8,
            cursor="hand2",
            activebackground="#388e3c",
            activeforeground="white",
            state=tk.DISABLED
        )
        self.cnn_predict_button.pack(side=tk.LEFT, padx=(5, 0), expand=True, fill=tk.X)
        
        # Görüntü göstergeleri için frame
        self.image_frame = tk.Frame(cnn_frame, bg=self.colors["bg_primary"])
        self.image_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Görüntü etiketi ve gösterge çerçevelerini oluştur
        img_label_frame = tk.Frame(self.image_frame, bg=self.colors["bg_primary"])
        img_label_frame.pack(fill=tk.X)
        
        # Orijinal görüntü için etiket
        self.original_label = tk.Label(
            img_label_frame, 
            text="Orijinal Görüntü", 
            bg=self.colors["bg_primary"], 
            fg=self.colors["text_primary"],
            font=self.fonts["label"]
        )
        self.original_label.pack(side=tk.LEFT, expand=True)
        
        # Segmentasyon görüntüsü için etiket
        self.segmentation_label = tk.Label(
            img_label_frame, 
            text="Tümör Bölgesi", 
            bg=self.colors["bg_primary"], 
            fg=self.colors["text_primary"],
            font=self.fonts["label"]
        )
        self.segmentation_label.pack(side=tk.LEFT, expand=True)
        
        # Görüntü göstergeleri çerçevesi
        img_display_frame = tk.Frame(self.image_frame, bg=self.colors["bg_primary"])
        img_display_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Görüntü göstericilerin arka planları
        img_bg = tk.Frame(
            img_display_frame, 
            bg=self.colors["bg_secondary"], 
            relief=tk.GROOVE, 
            bd=1
        )
        img_bg.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5))
        
        seg_bg = tk.Frame(
            img_display_frame, 
            bg=self.colors["bg_secondary"], 
            relief=tk.GROOVE, 
            bd=1
        )
        seg_bg.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(5, 0))
        
        # Orijinal görüntü display
        self.image_display = tk.Label(
            img_bg, 
            bg=self.colors["bg_secondary"], 
            width=300, 
            height=240
        )
        self.image_display.pack(padx=5, pady=5, expand=True)
        
        # Segmentasyon görüntüsü display
        self.segmentation_display = tk.Label(
            seg_bg, 
            bg=self.colors["bg_secondary"], 
            width=300, 
            height=240
        )
        self.segmentation_display.pack(padx=5, pady=5, expand=True)
    
    def _create_graph_frame(self):
        """Create the frame for displaying graphs"""
        self.frame_grafik = tk.LabelFrame(
            self.left_frame, 
            text="Tahmin Sonuçları", 
            font=self.fonts["subheader"], 
            bg=self.colors["bg_primary"],
            fg=self.colors["accent"],
            padx=15, pady=15
        )
        self.frame_grafik.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Sonuç gösterme için bilgilendirme etiketi
        self.result_info = tk.Label(
            self.frame_grafik,
            text="Tahmin sonuçları burada gösterilecek",
            font=self.fonts["label"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_secondary"]
        )
        self.result_info.pack(pady=10)
    
    def get_feature_values(self):
        """Get the values entered in the feature input fields"""
        values = []
        try:
            # Tüm alanlar için gerekli kontrolleri yap
            feature_names = ["Tumor Boyutu (mm)", "Metastatik Lenf Nodu Sayısı", "Nükleer Grade", 
                            "Östrojen Reseptörü (0/1)", "Progesteron Reseptörü (0/1)", "Yaş"]
            
            for i, entry in enumerate(self.entries):
                val = entry.get().strip()
                
                # Boş kontrolü
                if not val:
                    messagebox.showerror("Hata", f"{feature_names[i]} alanı boş bırakılamaz.")
                    return None
                
                # Sayı kontrolü
                if i == 3 or i == 4:  # Östrojen ve Progesteron Reseptörü için
                    if val not in ["0", "1"]:
                        messagebox.showerror("Hata", f"{feature_names[i]} için 0 veya 1 değeri giriniz.")
                        return None
                    values.append(int(val))
                else:
                    try:
                        numerical_val = float(val)
                        
                        # Limit kontrolleri
                        if i == 0 and (numerical_val <= 0 or numerical_val > 100):  # Tumor boyutu
                            messagebox.showerror("Hata", "Tumor boyutu 0-100 mm arasında olmalı.")
                            return None
                        elif i == 1 and (numerical_val < 0 or numerical_val > 20):  # Lenf nodu
                            messagebox.showerror("Hata", "Lenf nodu sayısı 0-20 arasında olmalı.")
                            return None
                        elif i == 2 and (numerical_val < 1 or numerical_val > 3):  # Grade
                            messagebox.showerror("Hata", "Nükleer grade 1-3 arasında olmalı.")
                            return None
                        elif i == 5 and (numerical_val < 18 or numerical_val > 100):  # Yaş
                            messagebox.showerror("Hata", "Yaş 18-100 arasında olmalı.")
                            return None
                        
                        values.append(numerical_val)
                    except ValueError:
                        messagebox.showerror("Hata", f"{feature_names[i]} için geçerli bir sayı giriniz.")
                        return None
            
            return values
        except Exception as e:
            messagebox.showerror("Hata", f"Veriler alınırken bir hata oluştu: {str(e)}")
            logging.error(f"get_feature_values error: {str(e)}")
            return None
    
    def display_prediction_graph(self, probabilities, evre):
        """Display a pie chart for prediction probabilities with stage"""
        # Önceki grafik varsa temizle
        if hasattr(self, 'result_info'):
            self.result_info.destroy()
        
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        # Sonuç panelini oluştur
        result_frame = tk.Frame(self.frame_grafik, bg=self.colors["bg_primary"])
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Sol kısım: Pasta grafiği
        chart_frame = tk.Frame(result_frame, bg=self.colors["bg_primary"])
        chart_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.BOTH, expand=True)
        
        # Sağ kısım: Sonuç bilgileri
        info_frame = tk.Frame(result_frame, bg=self.colors["bg_primary"])
        info_frame.pack(side=tk.LEFT, padx=(10, 0), fill=tk.Y)
        
        # Pasta grafiği
        fig = plt.figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        labels = ["İyi Huylu", "Kötü Huylu"]
        colors = [self.colors["accent_light"], self.colors["accent"]]
        
        # Pasta dilimi özellikleri
        wedges, texts, autotexts = ax.pie(
            probabilities, 
            labels=None,
            autopct='%1.1f%%', 
            colors=colors, 
            startangle=90,
            textprops={'fontsize': 11, 'color': 'white', 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'alpha': 0.9}
        )
        
        # Efsaneyi grafiğin altına yerleştir
        ax.legend(
            wedges,
            [f"{lbl}: %{prob*100:.1f}" for lbl, prob in zip(labels, probabilities)],
            title="Sonuçlar",
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            frameon=True,
            shadow=True,
            ncol=2,
            facecolor=self.colors["bg_primary"]
        )
        
        # Ekstra boşluğu kaldır
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        
        # Evre başlığı
        evre_label = tk.Label(
            info_frame,
            text="Klinik Evre:",
            font=self.fonts["subheader"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_primary"],
            anchor="w"
        )
        evre_label.pack(fill=tk.X, pady=(0, 5))
        
        # Evre değeri - daha göze çarpıcı
        evre_value_bg = self.colors["accent"] if "IV" in evre else self.colors["accent_light"]
        evre_value = tk.Label(
            info_frame,
            text=evre,
            font=("Segoe UI", 12, "bold"),
            bg=evre_value_bg,
            fg="white",
            padx=15,
            pady=10,
            relief=tk.RIDGE
        )
        evre_value.pack(fill=tk.X, pady=(0, 15))
        
        # Tahmin sonucu başlığı
        result_text = "Kötü Huylu" if probabilities[1] > probabilities[0] else "İyi Huylu"
        result_label = tk.Label(
            info_frame,
            text=f"Tahmin: {result_text}",
            font=self.fonts["subheader"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_primary"],
            anchor="w"
        )
        result_label.pack(fill=tk.X, pady=(0, 15))
        
        # Grafiği çiz
        self.canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def display_image(self, filepath):
        """Display the loaded image in the UI"""
        try:
            image = Image.open(filepath)
            self.loaded_image = image  # Store the PIL Image object instead of just the path
            display_image = image.copy()
            display_image.thumbnail((300, 240))
            tk_image = ImageTk.PhotoImage(display_image)
            
            # Orijinal görüntüyü göster
            self.image_display.config(image=tk_image)
            self.image_display.image = tk_image
            
            # Segmentasyon görüntüsünü temizle
            self.segmentation_display.config(image='')
            
            return True
        except Exception as e:
            messagebox.showerror("Hata", f"Görüntü yüklenirken bir hata oluştu: {str(e)}")
            return False
    
    def display_segmentation(self, original_image=None, tumor_coords=None, error_message=None):
        """
        Display the segmentation image in the UI by zooming in on the tumor region.
        
        Parameters:
        - original_image: The original PIL Image object
        - tumor_coords: Tuple of (x1, y1, x2, y2) coordinates of the tumor region
        - error_message: Optional error message to display when segmentation fails
        """
        try:
            # Case 1: If we have both an original image and tumor coordinates
            if isinstance(original_image, Image.Image) and tumor_coords is not None:
                # Unpack tumor region coordinates
                x1, y1, x2, y2 = tumor_coords
                
                # Add padding to tumor region for better visibility
                padding = max(10, min(x2-x1, y2-y1) // 5)  # Minimum 10px or 20% of tumor size
                
                # Get image dimensions
                width, height = original_image.size
                
                # Apply padding with bounds checking
                padded_x1 = max(0, x1 - padding)
                padded_y1 = max(0, y1 - padding)
                padded_x2 = min(width, x2 + padding)
                padded_y2 = min(height, y2 + padding)
                
                # Create a copy of the original image and crop to the padded tumor region
                tumor_region = original_image.copy().crop((padded_x1, padded_y1, padded_x2, padded_y2))
                
                # Resize for display if needed while maintaining aspect ratio
                display_image = tumor_region.copy()
                display_image.thumbnail((300, 240))
                
                # Convert to Tkinter image
                tk_image = ImageTk.PhotoImage(display_image)
                
                # Update the segmentation display
                self.segmentation_display.config(image=tk_image)
                self.segmentation_display.image = tk_image
                
                # Enable CNN prediction button
                self.enable_cnn_prediction()
                
                return True
            
            # Case 2: If we have an original image but no tumor coordinates (fallback)
            elif isinstance(original_image, Image.Image) or isinstance(self.loaded_image, Image.Image):
                # Use provided image or fallback to loaded_image
                img = original_image if isinstance(original_image, Image.Image) else self.loaded_image
                
                # Create a placeholder image with text indicating no segmentation model
                placeholder = Image.new('RGB', (300, 240), color=self.colors["bg_secondary"])
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(placeholder)
                
                # Try to get a font, fall back to default if not available
                try:
                    font = ImageFont.truetype("arial.ttf", 14)
                except IOError:
                    font = ImageFont.load_default()
                
                # Draw text explaining the issue
                message = error_message or "Segmentasyon modeli bulunamadı.\nTümör bölgesi gösterilemiyor."
                
                # Handle newlines and text wrapping
                lines = message.split('\n')
                y_pos = 100
                for line in lines:
                    draw.text((20, y_pos), line, fill=self.colors["text_primary"], font=font)
                    y_pos += 25
                
                # Convert to Tkinter image
                tk_image = ImageTk.PhotoImage(placeholder)
                
                # Update the segmentation display
                self.segmentation_display.config(image=tk_image)
                self.segmentation_display.image = tk_image
                
                # Still enable CNN prediction as the original image is available
                self.enable_cnn_prediction()
                
                return False
            else:
                messagebox.showwarning("Uyarı", "Tümör bölgesi tespit edilemedi.")
                return False
        except Exception as e:
            messagebox.showerror("Hata", f"Segmentasyon görüntüsünü gösterirken bir hata oluştu: {str(e)}")
            return False
    
    def upload_image(self):
        """Handle image upload"""
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if filepath:
            self.display_image(filepath)
    
    def create_progress_window(self, title="İşlem Yapılıyor", initial_text="Lütfen bekleyin..."):
        """Create a progress window for various processes"""
        progress_window = tk.Toplevel(self.root)
        progress_window.title(title)
        
        # Pencere boyutu
        width, height = 400, 180
        
        # Ana pencerenin ortasında konumlandır
        x = self.root.winfo_x() + (self.root.winfo_width() - width) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - height) // 2
        
        progress_window.geometry(f"{width}x{height}+{x}+{y}")
        progress_window.config(bg=self.colors["bg_primary"])
        progress_window.resizable(False, False)
        
        # İç çerçeve
        inner_frame = tk.Frame(progress_window, bg=self.colors["bg_primary"], padx=20, pady=20)
        inner_frame.pack(fill=tk.BOTH, expand=True)
        
        # Başlık
        title_label = tk.Label(
            inner_frame, 
            text=title,
            bg=self.colors["bg_primary"],
            fg=self.colors["accent"],
            font=self.fonts["subheader"]
        )
        title_label.pack(pady=(0, 15))
        
        # Bilgi etiketi
        progress_label = tk.Label(
            inner_frame, 
            text=initial_text, 
            bg=self.colors["bg_primary"], 
            fg=self.colors["text_primary"],
            font=self.fonts["label"]
        )
        progress_label.pack(pady=10)
        
        # İlerleme çubuğu stili
        style = ttk.Style()
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor=self.colors["bg_secondary"],
            background=self.colors["accent"],
            thickness=20
        )
        
        # İlerleme çubuğu
        progress_bar = ttk.Progressbar(
            inner_frame, 
            orient="horizontal", 
            length=350, 
            mode="determinate",
            style="Custom.Horizontal.TProgressbar"
        )
        progress_bar.pack(pady=10, fill=tk.X)
        
        return progress_window, progress_label, progress_bar
    
    def set_rf_callback(self, callback):
        """Set callback for Random Forest prediction button"""
        self.rf_button.config(command=callback)
    
    def set_cnn_train_callback(self, callback):
        """Set callback for CNN training button"""
        def confirm_and_train():
            result = messagebox.askokcancel(
                "Onay", 
                "CNN modelini eğitmek istediğinize emin misiniz?\nBu işlem uzun sürebilir.",
                icon=messagebox.WARNING
            )
            if result:
                callback()
        
        self.cnn_train_button.config(command=confirm_and_train)
    
    def set_upload_callback(self, callback):
        """Set callback for image upload button"""
        self.upload_button.config(command=callback)
    
    def set_cnn_predict_callback(self, callback):
        """Set callback for CNN prediction button"""
        self.cnn_predict_button.config(command=callback)
    
    def enable_cnn_prediction(self):
        """Enable the CNN prediction button"""
        self.cnn_predict_button.config(state=tk.NORMAL)
    
    def show_results_window(self, title, charts, width=900, height=400):
        """Display a window with charts/results"""
        results_window = tk.Toplevel(self.root)
        results_window.title(title)
        
        # Ana pencerenin ortasında konumlandır
        x = self.root.winfo_x() + (self.root.winfo_width() - width) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - height) // 2
        
        results_window.geometry(f"{width}x{height}+{x}+{y}")
        results_window.config(bg=self.colors["bg_primary"])
        
        # İç çerçeve
        inner_frame = tk.Frame(results_window, bg=self.colors["bg_primary"], padx=20, pady=20)
        inner_frame.pack(fill=tk.BOTH, expand=True)
        
        # Başlık
        title_label = tk.Label(
            inner_frame, 
            text=title,
            bg=self.colors["bg_primary"],
            fg=self.colors["accent"],
            font=self.fonts["subheader"]
        )
        title_label.pack(pady=(0, 15))
        
        # Grafik
        chart_canvas = FigureCanvasTkAgg(charts, master=inner_frame)
        chart_canvas.draw()
        chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Kapat düğmesi
        close_button = tk.Button(
            inner_frame,
            text="Kapat",
            bg=self.colors["button_secondary"],
            fg="white",
            font=self.fonts["button"],
            padx=15,
            pady=5,
            command=results_window.destroy
        )
        close_button.pack(pady=(15, 0))
        
        return results_window
    
    def run(self):
        """Start the Tkinter main loop"""
        logging.debug("Starting Tkinter main loop...")
        self.root.mainloop()
        logging.debug("Tkinter main loop exited.")

    def display_image_prediction_result(self, probabilities):
        """Display a pie chart for CNN prediction probabilities"""
        # Görüntü tahmini için pasta grafiği penceresi oluştur
        fig = plt.figure(figsize=(8, 6), dpi=100)
        
        # Ana başlık
        fig.suptitle("Görüntü Analizi Sonuçları", fontsize=16, color=self.colors["text_primary"], y=0.95)
        
        # Pasta grafiği için bir alt grafik alanı oluştur
        ax = fig.add_subplot(111)
        
        labels = ["İyi Huylu", "Kötü Huylu"]
        colors = [self.colors["accent_light"], self.colors["accent"]]
        
        wedges, texts, autotexts = ax.pie(
            probabilities, 
            labels=None,
            autopct='%1.1f%%', 
            colors=colors, 
            startangle=90,
            textprops={'fontsize': 12, 'color': 'white', 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'alpha': 0.9}
        )
        
        # Efsaneyi ekle ve güzel hizala
        ax.legend(
            wedges, 
            [f"{lbl}: %{prob*100:.1f}" for lbl, prob in zip(labels, probabilities)],
            title="Sonuçlar",
            loc="center right",
            bbox_to_anchor=(0.9, 0.5),
            frameon=True,
            shadow=True,
            facecolor=self.colors["bg_primary"]
        )
        
        # Ekstra boşluğu kaldır
        plt.tight_layout()
        
        # Sonuç penceresini göster
        self.show_results_window("📊 Görüntü Analiz Sonuçları", fig)
        
        return True