# Proje: Yapay Zeka ve Graf Tabanlı Oltalama (Phishing) Tespit Sistemi

## Proje Linkleri

- **Site Linki:** [PhishGuard.ai](https://phishingtest-m00b.onrender.com/))
  
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

## Hangi özellikler kontrol ediliyor?

***1. Leksikal ve Yapısal URL Özellikleri (Makine Öğrenmesi İçin)***
Bu özellikler feature_extractor.py dosyasında, kullanıcının girdiği URL metni üzerinden anında hesaplanan sayısal verilerdir:

*URL Uzunluğu (URL Length)*: Oltalama siteleri, gerçek alan adını (domain) gizlemek için genellikle çok uzun URL'ler kullanır.

*Özel Karakter Sayıları*: URL içindeki @, - (tire), _, ?, = gibi karakterlerin sayısı. Örneğin, bir URL'de @ işareti varsa, tarayıcı @ işaretinden önceki kısmı yok sayar; bu sık kullanılan bir oltalama taktiğidir.

*IP Adresi Kullanımı*: URL'de bir alan adı (örn: google.com) yerine doğrudan bir IP adresi (örn: [http://192.168.1.1/login](http://192.168.1.1/login)) kullanılıyorsa, bu durum modeli yüksek risk konusunda uyarır.

*Nokta (.) Sayısı ve Alt Alan Adları (Subdomains)*: Çok fazla nokta içermek, çok sayıda alt alan adı kullanıldığı anlamına gelir (örn: login.update.secure.banka.com). Bu, sahtekarların meşru görünmek için kullandığı bir yöntemdir.

*Şüpheli Kelimelerin Varlığı*: URL içinde "login", "update", "secure", "verify", "account", "banking" gibi oltalama senaryolarında sıkça geçen anahtar kelimelerin bulunup bulunmadığı (1 veya 0 olarak kodlanır).

*Kısaltılmış URL Kullanımı*: Bit.ly, TinyURL gibi servislerin kullanılıp kullanılmadığı kontrol edilir.

***2. Graf Tabanlı Özellikler (Anomali Tespiti İçin)***
Bu özellikler, URL'nin internet üzerindeki diğer sitelerle olan bağlantılarını bir "Ağ" (Network) olarak modelleyen graph_engine.py tarafından çıkarılır:

*Gelen ve Giden Bağlantı Sayısı (In-Degree / Out-Degree)*: Bu sitenin kaç farklı siteye link verdiği veya kaç sitenin buraya link verdiği. Oltalama siteleri genellikle izole çalışır veya aniden ortaya çıktıkları için meşru sitelerden gelen "In-Degree" bağlantıları çok düşüktür.

*Komşuluk Analizi (Neighborhood Risk Score)*: Sitenin bağlantı kurduğu diğer sitelerin (düğümlerin/nodeların) güvenilirlik skoru. Eğer bir site sürekli "kötü niyetli" bilinen sitelerle iletişim halindeyse, kendisinin de oltalama olma ihtimali artar.

*Merkezilik Skoru (Centrality)*: Sitenin kendi ağı içindeki önemi. Meşru ve büyük web sitelerinin (örneğin popüler bir banka) merkezilik skoru çok yüksekken, oltalama sitelerinin skoru anomali gösterecek şekilde düşüktür.

***3. Dış Kaynaklı / İstatistiksel Özellikler (Opsiyonel)***
Sistemin internete çıkıp sorguladığı verilerdir:

*Domain Yaşı (Whois Data)*: Meşru siteler yıllardır yayındayken, phishing siteleri genellikle birkaç günlük veya haftalıktır.

*SSL (HTTPS) Kullanımı*: Sitenin geçerli bir güvenlik sertifikası olup olmadığı. (Not: Günümüzde saldırganlar da ücretsiz SSL alabildiği için tek başına yeterli değildir, ancak diğer özelliklerle birleştiğinde güçlü bir sinyaldir).


## Kullanılan Datasetler
-https://www.kaggle.com/datasets/arnavs19/phishing-websites-dataset

-https://www.kaggle.com/datasets/akashkr/phishing-website-dataset


## Öneriler
usom.gov.tr/adres linkindeki linkler denenerek sistemin çalışıp çalışmadığı kontrol edilebilir.Doğruluk ve yanlışlık göz önünde bulundurulurken sistemin bir yapay zeka olduğu ve hata yapabileceği unutulmamalıdır.
