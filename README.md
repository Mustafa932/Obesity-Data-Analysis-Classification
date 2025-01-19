# Obesity Data Analysis & Classification
Bu projede, obezite ile ilgili veriler üzerinde analiz yaparak, eksik verileri işliyor, veriyi dönüştürüyor ve üç farklı makine öğrenimi modeliyle sınıflandırma işlemi gerçekleştiriyoruz. Kullanılan modeller: Lojistik Regresyon, Karar Ağacı ve Rastgele Orman.

# Adımlar
# 1. Kütüphanelerin İçe Aktarılması
Projemizin başında, veri analizi, ön işleme ve makine öğrenimi için gerekli kütüphaneler içe aktarılmaktadır. Bu kütüphaneler arasında pandas, numpy, ve sklearn gibi araçlar yer almaktadır.

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Veri Setinin Yüklenmesi
Obezite veri seti, CSV dosyasından yüklenir. Bu veri seti, obeziteyle ilişkili çeşitli özellikleri içerir.

data = pd.read_csv('/content/ObesityDataSet_raw_and_data_sinthetic (1).csv')

# 3. Eksik Veri Analizi
Veri setindeki eksik değerler hesaplanır ve bu eksikliklerin sayısı ve yüzdesi belirlenir.

missing_data = data.isnull().sum()
missing_data_percentage = (missing_data / len(data)) * 100

print("Eksik Veri Sayısı:\n", missing_data)
print("\nEksik Veri Yüzdesi:\n", missing_data_percentage)

# 4. Eksik Verilerin Doldurulması
Eksik veriler şu şekilde doldurulur:

Sayısal sütunlar, ortalama değerle doldurulur.
Kategorik sütunlar, en sık görülen değerle doldurulur.

data['Weight'] = data['Weight'].fillna(data['Weight'].mean())
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])

# 5. Veri Dönüştürme
One-Hot Encoding: Kategorik değişkenler sayısal formata dönüştürülür.
Label Encoding: Kategorik sütunlar etiketlenir.

data = pd.get_dummies(data, columns=['Gender', 'family_history_with_overweight'], drop_first=True)
encoder = LabelEncoder()
data['FAVC'] = encoder.fit_transform(data['FAVC'])

# 6. Veri Normalizasyonu
Sayısal veriler Min-Max Normalizasyonu yöntemi ile ölçeklenir.

scaler = MinMaxScaler()
data[['Weight', 'Height']] = scaler.fit_transform(data[['Weight', 'Height']])

# 7. Makine Öğrenimi Modelleme

Kategorik sütunlar etiketlenir ve özellikler hedef değişkeninden ayrılır.
Veri seti eğitim ve test kümelerine ayrılır.
Üç farklı model eğitilir ve test edilir:
Lojistik Regresyon
Karar Ağacı
Rastgele Orman
Her modelin doğruluk oranları ve sınıflandırma raporları yazdırılır.

# Lojistik Regresyon
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
y_pred_logreg = logreg_model.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

# Karar Ağacı
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# Rastgele Orman
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
Sonuç Analizi
Çıktılar, her bir modelin eğitim ve test sürecinin sonuçlarını göstermektedir. Her model için doğruluk (accuracy), precision, recall, f1-score, ve support değerleri bulunmaktadır:

# 1. Lojistik Regresyon
Doğruluk (Accuracy): 0.65
Modelin genel doğruluğu %65. Bu, modelin sınıflandırmalarda doğru sonuç verme oranının düşük olduğunu gösteriyor.
Precision: Bazı sınıflarda düşük değerler var (örneğin, sınıf 4 ve 6).
Recall: Bazı sınıflarda oldukça düşük (örneğin, sınıf 6'da %35).
F1-Score: Genel olarak düşük değerler mevcut, özellikle sınıf 6 için %37 gibi düşük bir değer.
Lojistik Regresyon modeli, bu veri setinde zayıf performans göstermektedir, özellikle sınıflar arasında dengesizlik bulunmakta. Bu model, daha karmaşık ilişkileri yeterince iyi öğrenememiş görünüyor.

# 2. Karar Ağacı
Doğruluk (Accuracy): 0.94
Modelin genel doğruluğu %94, bu oldukça yüksek bir başarı oranıdır.
Precision ve Recall: Genellikle %90 ve üzeri değerler almış, bu da modelin tahminlerde oldukça başarılı olduğunu gösteriyor.
F1-Score: Yüksek F1-Skorları (çoğunlukla %94 ve üzeri) ile, hem hataları iyi yakalıyor hem de doğru sınıflandırmaları başarıyla yapıyor.
Karar Ağacı modeli, Lojistik Regresyona göre çok daha iyi performans göstermekte. Sınıflar arasında dengeli bir sonuç vermiş ve karmaşık ilişkileri daha iyi öğrenmiş.

# 3. Rastgele Orman
Doğruluk (Accuracy): 0.96
Rastgele Orman modeli %96 doğruluk oranı ile en iyi sonucu vermiş.
Precision ve Recall: Çoğu sınıfta %90 ve üzeri değerler alarak, doğru tahmin oranının yüksek olduğunu gösteriyor.
F1-Score: Neredeyse tüm sınıflar için %95 ve üzeri F1-Skorları mevcut, bu da hem yüksek hassasiyet hem de geri çağırma oranlarının dengeli olduğunu ortaya koyuyor.
Rastgele Orman modeli, diğer iki modele kıyasla en iyi performansı sergilemiş. Genellikle yüksek doğruluk, hassasiyet ve geri çağırma oranları ile bu model veri setindeki ilişkileri daha iyi öğrenmiş.
Genel Değerlendirme:
Rastgele Orman modeli, sınıflandırmada en iyi performansı göstermiştir, ardından Karar Ağacı gelmektedir.
Lojistik Regresyon modeli, doğruluğu ve diğer metrikleri açısından en düşük performansı sergilemiştir, bu da lineer olmayan ilişkileri öğrenmede yetersiz kaldığını gösterir.
Özetle, bu veri seti için Rastgele Orman en iyi seçimdir çünkü daha karmaşık ve doğrusal olmayan ilişkileri daha iyi yakalayabilir. Bu sonuçlar, daha güçlü ve daha karmaşık modellerin (Karar Ağacı ve Rastgele Orman) daha iyi performans gösterdiğini ortaya koymaktadır.

# Gereksinimler
Python 3.x
pandas
numpy
scikit-learn
Kurulum

# Gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanabilirsiniz:
pip install pandas numpy scikit-learn
