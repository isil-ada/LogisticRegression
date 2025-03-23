# **Kredi Kartı Borç Ödeme Tahmini (Logistic Regression)**

## **📌 1. Problem Tanımı**
Bu proje, kredi kartı kullanıcılarının borçlarını ödeyip ödemeyeceğini tahmin etmeyi amaçlamaktadır. Bankalar ve finans kuruluşları, müşterilerinin kredi risklerini değerlendirmek için bu tür tahmin modellerine ihtiyaç duyar. Doğru tahminler, **mali kayıpları önlemeye** ve **daha bilinçli kredi kararları vermeye** yardımcı olur.

---

## **📌 2. Veri Seti**
Veri seti, **"Credit Card Defaulter Prediction.csv"** dosyasından alınmıştır ve aşağıdaki bilgileri içerir:

- **ID**: Müşteri kimliği (Kullanılmadı, modelden çıkarıldı).
- **Kişisel Bilgiler**: Yaş, gelir, kredi limitleri vb.
- **Ödeme Geçmişi**: Önceki ödemeler, gecikmeler vb.
- **Hedef Değişken (default_Y)**:
  - **1**: Müşteri borcunu ödeyemedi.
  - **0**: Müşteri borcunu ödedi.

🔹 **Ön İşleme Adımları:**
- **Eksik veri kontrolü yapıldı** (Boş değer bulunmadı).
- **Kategorik değişkenler One-Hot Encoding ile dönüştürüldü**.
- **Özellikler standardize edildi (StandardScaler kullanıldı)**.
- **Veri eğitim ve test setlerine ayrıldı (%80 eğitim - %20 test).**

---

## **📌 3. Kullanılan Yöntem**
Bu projede **lojistik regresyon** kullanılmıştır. İki farklı model oluşturulmuştur:

1️⃣ **Sklearn LogisticRegression Modeli** ([LogisticRegression.py](LogisticRegression.py))
   - **Sklearn kütüphanesinin LogisticRegression sınıfı** kullanılarak uygulanmıştır.
   - Model, **max_iter=500** ile eğitilmiştir.
   
2️⃣ **Manuel Logistic Regression Modeli** ([LogisticRegression2.py](LogisticRegression2.py))
   - **Gradient Descent algoritması kullanılarak sıfırdan** lojistik regresyon modeli oluşturulmuştur.
   - **Özelleştirilebilir öğrenme oranı ve iterasyon sayısı** ile optimize edilmiştir.
   - Model, **learning_rate=0.0099, epochs=450** ile eğitilmiştir.

📌 **Model Eğitimi:**
- **Özellikler ve hedef değişken ayrıldı.**
- **Veri seti eğitim ve test kümelerine ayrıldı.**
- **Veriler ölçeklendirildi (StandardScaler).**
- **Model eğitildi ve test edildi.**

---

## **📌 4. Sonuçlar ve Model Performansı**
Her iki modelin doğruluğu (accuracy) ve hata analizi için **confusion matrix** kullanılmıştır.

### **1️⃣ Sklearn Logistic Regression Modeli Sonuçları**
- **Eğitim Süresi:** ≈ **0.16921990003902465 saniye**
- **Tahmin Süresi:** ≈ **0.001525100029539317 saniye**
- **Doğruluk (Accuracy):** ≈ **%82**
- **Confusion Matrix:**
- *(Grafik olarak görselleştirildi)*
- ![image](https://github.com/user-attachments/assets/987897d5-7060-47b2-8940-9546422861c6)

- 

### **2️⃣ Manuel Oluşturulan Logistic Regression Modeli Sonuçları**
- **Eğitim Süresi:** ≈ **0.9786075999727473 saniye**
- **Tahmin Süresi:** ≈ **0.000534399994648993 saniye**
- **Doğruluk (Accuracy):** ≈ **%816**
- **Confusion Matrix:** *(Grafik olarak görselleştirildi)*

🔹 **Sonuç Yorumu:**
- **Sklearn modeli genellikle daha hızlıdır** çünkü optimize edilmiş bir kütüphane kullanır.
- **Manuel oluşturulmuş modelde öğrenme oranı ve epoch sayısı doğru ayarlandığında iyi sonuçlar alınmıştır.**
- **Accuracy yanıltıcı olabilir**, bu yüzden **F1-Score, Precision, Recall gibi metrikler de incelenmelidir.**

---

## **📌 5. Yorum/Tartışma**
- **Veri seti büyütülerek modelin genelleme yeteneği artırılabilir.**
- **Daha iyi hiperparametre ayarları için Grid Search gibi teknikler kullanılabilir.**
- **Daha güçlü modeller (Random Forest gibi) ile kıyaslama yapılabilir.**
- **Precision ve Recall hesaplanarak modelin dengesiz veri setlerinde performansı test edilebilir.**

**Bu proje, kredi risk analizlerinde lojistik regresyonun nasıl kullanılabileceğini göstermektedir!**

