from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import sys
from ml_models import MLModels
from feature_extractor import FeatureExtractor
from data_loader import DataLoader
import threading

base_dir = os.path.abspath(os.path.dirname(__file__))
website_dir = os.path.join(base_dir, 'website')

app = Flask(__name__, static_folder=website_dir)
CORS(app)

# Global variables
model = None
extractor = FeatureExtractor()
is_ready = False

        ml.load_trained()
        model = ml
        is_ready = True
        print("[OK] Server ready with pre-trained model!", flush=True)
    except FileNotFoundError:
        # No saved model - train from scratch
        print("[!] No pre-trained model found. Training from dataset...", flush=True)
        loader = DataLoader('dataset_full.csv')
        df = loader.load_data()
        
        X = df.drop('phishing', axis=1).values
        y = df['phishing'].values
        
        # Replace -1 with NaN for proper XGBoost handling
        X = np.where(X == -1, np.nan, X)
        
        ml.train(X, y)
        model = ml
        
        # Save for next time
        os.makedirs('trained_model', exist_ok=True)
        ml.save('trained_model/xgboost_phishguard.pkl')
        
        is_ready = True
        print("[OK] Model trained and saved. Server ready!", flush=True)
    except Exception as e:
        print(f"[ERROR] Model initialization failed: {e}", flush=True)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ready" if is_ready else "loading"})

@app.route('/analyze', methods=['POST'])
def analyze():
    if not is_ready:
        return jsonify({"error": "Models are still loading, please wait."}), 503
        
    data = request.json
    url = data.get('url', '').lower().strip()
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
        
    try:
        # 1. Extract Features (includes DNS + SSL + WHOIS checks)
        features = extractor.extract_features(url)
        
        # 2. Get prediction
        proba = model.predict_proba(features)[0][1]
        graph_score = proba * 100
        
        # Helper to safely read feature values
        def sf(val):
            return float(val) if not np.isnan(val) else None
        def si(val):
            return int(val) if not np.isnan(val) else 0

        # Extract values
        url_len      = si(features[0][18])
        has_ssl      = sf(features[0][106]) == 1
        has_spf      = sf(features[0][98]) == 1
        domain_age   = sf(features[0][100])
        expiry_days  = sf(features[0][101])
        mx_count     = si(features[0][104])
        ns_count     = si(features[0][103])
        asn          = sf(features[0][99])

        security_features = []
        security_features.append({"name": "SSL/TLS Sertifikası", "present": has_ssl, "detail": "HTTPS bağlantısı aktif" if has_ssl else "HTTPS bağlantısı yok"})
        security_features.append({"name": "SPF E-posta Doğrulaması", "present": has_spf, "detail": "SPF kaydı mevcut" if has_spf else "SPF kaydı bulunamadı"})
        security_features.append({"name": "Mail Sunucusu (MX)", "present": mx_count > 0, "detail": f"{mx_count} MX kaydı bulundu" if mx_count > 0 else "MX kaydı yok"})
        
        if domain_age is not None:
            age_safe = domain_age > 365
            age_text = f"{int(domain_age/365)} yıl" if domain_age > 365 else f"{int(domain_age)} gün"
            security_features.append({"name": "Domain Yaşı", "present": age_safe, "detail": f"{age_text} - {'köklü domain' if age_safe else 'yeni domain'}"})
        else:
            security_features.append({"name": "Domain Yaşı", "present": None, "detail": "WHOIS alınamadı"})

        security_features.append({"name": "DNS Altyapısı (NS)", "present": ns_count >= 2, "detail": f"{ns_count} name server"})
        security_features.append({"name": "ASN (Ağ Kimliği)", "present": asn is not None and asn > 0, "detail": f"AS{int(asn)}" if asn else "Bilinmiyor"})
        
        if expiry_days is not None:
            security_features.append({"name": "Domain Bitiş Süresi", "present": expiry_days > 90, "detail": f"{int(expiry_days)} gün kaldı"})

        security_features.append({"name": "URL Uzunluğu", "present": url_len < 50, "detail": f"{url_len} karakter - normal"})

        result = {
            "url": url,
            "overall_score": float(proba * 100),
            "xgboost_score": float(proba * 100),
            "graph_score": float(graph_score),
            "is_phishing": bool(proba > 0.5),
            "confidence": float(abs(proba - 0.5) * 200),
            "security_features": security_features
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Production deployment using Waitress
    from waitress import serve
    print("[OK] Server starting on http://0.0.0.0:5000")
    serve(app, host='0.0.0.0', port=5000)
