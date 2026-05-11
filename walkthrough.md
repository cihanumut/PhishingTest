# Oltalama Tespiti Karşılaştırma Raporu

Bu çalışma, sağlanan 88k satırlık oltalama veri seti üzerinde geleneksel ML ve Graf tabanlı yöntemleri kıyaslamaktadır.

## Kullanılan Yöntemler

### 1. Geleneksel ML (XGBoost)
- **Veri**: 111 yapısal özellik (URL uzunluğu, özel karakter sayıları, DNS bilgileri vb.).
- **Model**: Karar ağacı tabanlı XGBoost sınıflandırıcı.
- **Performans**: %96.50 doğruluk.

### 2. Graf Tabanlı (GNN)
- **Veri**: Örnekler arası kosinüs benzerliği %95 üzerinde olanlar arasında kenarlar oluşturularak bir graf yapısı kuruldu.
- **Model**: 2 katmanlı Graf Konvolüsyonel Ağ (GCN).
- **Performans**: %68.16 doğruluk.

## Karşılaştırma Grafiği
![Karşılaştırma Sonuçları](file:///c:/Users/hp/mthproje/comparison_results.png)

## Analiz ve Bulgular
- **Tablosal Veri Üstünlüğü**: XGBoost, bağımsız özellikler arasındaki doğrusal olmayan ilişkileri yakalamada çok daha başarılı oldu.
- **Graf İlişkileri**: Mevcut graf yapısı sadece özellik benzerliğine dayandığı için GNN modeli henüz XGBoost'un başarısına ulaşamadı. Graf yapısına alan adı sahipliği, IP blokları gibi "ilişkisel" veriler eklendiğinde bu skorun artması beklenmektedir.

## Yapılan Değişiklikler
- `data_loader.py`: Veri yükleme ve başlık yönetimi.
- `ml_models.py`: XGBoost eğitim hattı.
- `graph_models.py`: GNN (PyTorch Geometric) mimarisi.
- `main.py`: Karşılaştırma ve görselleştirme motoru.
