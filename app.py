import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ml_models import MLModels
from feature_extractor import FeatureExtractor

# Fix path for Render
base_dir = os.path.abspath(os.path.dirname(__file__))
website_dir = os.path.join(base_dir, 'website')

app = Flask(__name__, static_folder=website_dir)
CORS(app)

# Load model once at startup
print(f"[*] Loading pre-trained model...")
model = MLModels()
model.load_trained()
extractor = FeatureExtractor()

# --- Frontend Serving Routes ---
@app.route('/')
def serve_index():
    return send_from_directory(website_dir, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    # Try serving the file, if not found, fall back to index.html (for SPA-like behavior)
    if os.path.exists(os.path.join(website_dir, path)):
        return send_from_directory(website_dir, path)
    return send_from_directory(website_dir, 'index.html')

# --- API Routes ---
@app.route('/analyze', methods=['POST'])
def analyze_url():
    try:
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({"error": "URL missing"}), 400

        # 1. Extract features
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
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    print(f"[OK] Server starting on http://0.0.0.0:{port}")
    serve(app, host='0.0.0.0', port=port)
