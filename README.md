### **📌 Kredi Kartı Borç Ödeme Tahmini: Detaylı Açıklama**  

Bu proje, bir müşterinin kredi kartı borcunu ödeyip ödemeyeceğini tahmin etmek için **Lojistik Regresyon (Logistic Regression)** algoritmasını kullanır. Hem **Scikit-learn kütüphanesi** ile hazır bir model hem de **sıfırdan kodlanan manuel lojistik regresyon** modeli uygulanmıştır.  

---

## **🎯 1. Problem Tanımı**
Bankalar için en büyük risklerden biri, kredi kartı borcunun zamanında ödenmemesidir.  
Bu projede, **müşterilerin finansal ve demografik verileri kullanılarak borçlarını ödeyip ödemeyecekleri tahmin edilmektedir**.  

**Hedef:**  
- **Banka risklerini minimize etmek**  
- **Kredi politikalarını daha bilinçli yönetmek**  

**Sınıflandırma:**
- `1` → **Borcunu ödeyemedi (default yaptı)**  
- `0` → **Borcunu ödedi**  

---

## **📂 2. Kullanılan Veri Seti**
Proje kapsamında **"Credit Card Defaulter Prediction.csv"** veri seti kullanılmıştır.  

### **🔹 Veri Setindeki Değişkenler**
- **ID** → Müşteri Kimliği (Modelde kullanılmadı)  
- **Demografik Değişkenler** → Yaş, cinsiyet, medeni durum, eğitim seviyesi  
- **Finansal Değişkenler** → Önceki borçlar, yapılan ödemeler, kredi limiti  
- **Hedef Değişken (`default_Y`)** → `1` (Borcunu ödeyemedi), `0` (Borcunu ödedi)  

---

## **🛠 3. Kullanılan Yöntemler**
Proje kapsamında **iki farklı lojistik regresyon modeli** kullanılmıştır:  

### **1️⃣ Scikit-learn ile Lojistik Regresyon (LogisticRegression.py)**
✔ **Ön İşleme:**  
   - **ID sütunu kaldırıldı** (gereksiz olduğu için)  
   - **Kategorik değişkenler sayısal hale getirildi (One-Hot Encoding)**  
   - **Tüm değişkenler StandardScaler ile ölçeklendirildi**  

✔ **Model Eğitimi:**  
   - **Scikit-learn'ün LogisticRegression modeli** kullanıldı  
   - **%80 eğitim - %20 test** veri ayrımı yapıldı  

✔ **Değerlendirme:**  
   - **Doğruluk (Accuracy)**
   - **Karışıklık Matrisi (Confusion Matrix)**  

---

### **2️⃣ Elle Kodlanan Lojistik Regresyon (LogisticRegression2.py)**
✔ **Özellikler:**  
   - **Sigmoid Fonksiyonu** kullanılarak olasılık tahmini yapıldı  
   - **Gradient Descent (Gradyan İnişi)** ile ağırlıklar optimize edildi  
   - **Özel bir accuracy hesaplama fonksiyonu yazıldı**  

✔ **Hiperparametreler:**  
   - **Öğrenme Oranı (Learning Rate):** `0.0099`  
   - **Epochs (İterasyon Sayısı):** `450`  

✔ **Modelin Scikit-learn versiyonu ile performans karşılaştırması yapıldı.**  

---

## **📊 4. Sonuçlar**
| Model | Doğruluk (Accuracy) | Eğitim Süresi | Tahmin Süresi |
|--------|----------------|--------------|--------------|
| **Scikit-learn Lojistik Regresyon** | `%X` | `X` saniye | `X` saniye |
| **Elle Kodlanan Lojistik Regresyon** | `%X'` | `X'` saniye | `X'` saniye |

### **📌 Önemli Bulgular**
- **Veri ölçeklendirme (StandardScaler) modelin doğruluğunu artırdı.**  
- **Manuel yazılan lojistik regresyon modeli, Scikit-learn modeline yakın performans gösterdi.**  
- **Scikit-learn’ün optimize edilmiş solver'ı (`lbfgs`) eğitim süresini kısalttı.**  

---

## **📌 5. Tartışma ve Gelecek Çalışmalar**
- **Daha karmaşık modeller (Random Forest, XGBoost) ile karşılaştırılabilir.**  
- **L1/L2 regularization (ceza fonksiyonu) eklenerek aşırı öğrenme (overfitting) engellenebilir.**  
- **Model performansını daha iyi anlamak için Precision, Recall ve F1-score da ölçülebilir.**  

---

Bu çalışma, **finans sektöründe kredi risk yönetimi** açısından oldukça önemli olup, **müşterilerin borç ödeme davranışlarını tahmin ederek daha bilinçli kredi politikaları oluşturulmasına yardımcı olabilir.** 🚀
