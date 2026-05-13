# Proje: Yapay Zeka ve Graf Tabanlı Oltalama (Phishing) Tespit Sistemi

## Proje Tanımı:
Bu projede, URL yapılarını ve bağlantı özelliklerini analiz ederek bir web sitesinin güvenilir mi yoksa oltalama (phishing) amaçlı mı olduğunu tahmin edebilen hibrit bir yapay zeka sistemi geliştirilmiştir. Sistem, kullanıcı tarafından girilen URL'yi makine öğrenmesi teknikleri ve graf tabanlı anomali tespiti algoritmaları ile eş zamanlı olarak analiz ederek sınıflandırma yapmaktadır. Model, URL'nin leksikal özelliklerini ve yapısal ağ (graf) ilişkilerini öğrenerek sitelerin güvenilirliğini değerlendirmektedir. Sonuçlar basit, anlaşılır ve kullanıcı dostu bir web arayüzünde gösterilmektedir.

Projenin en önemli özelliği graf yapısı ve modern makine öğrenmesi tekniklerini birleştirerek hybrid bir öğrenme ortamı sunmasıdır.

## Hedef:
Makine öğrenimi ve graf tabanlı analiz tekniklerini bir arada kullanarak bir URL sınıflandırma ve anomali tespit modeli geliştirmek; ayrıca bu modeli, kullanıcıların bir web sitesinin güvenilirliğini hızlı ve pratik bir şekilde test edebileceği bir web arayüzü ile entegre etmektir.

## Fonksiyonel Gereksinimler:
• Kullanıcı sisteme test etmek istediği bir web sitesi bağlantısını (URL) girebilmelidir.

• Sistem, URL üzerinde dinamik olarak özellik çıkarımı (feature extraction) adımlarını uygulamalıdır.

• Model, çıkarılan yapısal verileri analiz ederek web sitesinin güvenilir (Legitimate) veya sahte (Phishing) olduğu yönünde sınıflandırma yapmalıdır.

• Sistem, tahmin sonucunu kullanıcıya net bir metin ve görsel geribildirim olarak göstermelidir.

• Sonuçlar kullanıcı dostu bir web arayüzünde sunulmalıdır (URL giriş alanı, analiz butonu, tahmin sonucu vb.).

## Teknik Detaylar:
• URL verilerinden anında anlamlı özellikler çıkarılmalı ve modellerin işleyebileceği vektörel formata dönüştürülmelidir (`feature_extractor.py`).

• Model geliştirme aşamasında Scikit-Learn tabanlı klasik Makine Öğrenmesi algoritmaları (`ml_models.py`) ve Graf Tabanlı algoritmalar (`graph_models.py`, `graph_engine.py`) hibrit bir yapıda kullanılmalıdır.

• Modellerin performansı, farklı algoritmaların doğruluk oranlarını karşılaştıran grafiklerle değerlendirilmelidir (örn. `hybrid_comparison_results.png`).
• Arayüz geliştirme için HTML/CSS/JavaScript ve backend için Python (`app.py`) kullanılmalıdır.

• Sistem, kullanıcıdan gelen URL'yi backend'e gönderip, önceden eğitilmiş modeller (`trained_model` dizini) üzerinden geçirerek tahmini gerçek zamanlı olarak arayüze yansıtabilmelidir.

•Model geliştirme aşamasında, çıkarılan sayısal özellikler üzerinden yüksek doğruluk ve hızla sınıflandırma yapabilmek için gelişmiş bir ağaç tabanlı algoritma olan XGBoost (eXtreme Gradient Boosting) kullanılmalıdır.

## Kullanılan Datasetler
https://www.kaggle.com/datasets/arnavs19/phishing-websites-dataset
https://www.kaggle.com/datasets/akashkr/phishing-website-dataset
