import os
import numpy as np
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='website', static_url_path='')
CORS(app)

# Lazy loading for performance
model = None
extractor = None

def get_resources():
    global model, extractor
    if model is None:
        from ml_models import MLModels
        from feature_extractor import FeatureExtractor
        print("[*] Initializing ML models and Network Probes...")
        model = MLModels()
        model.load_trained()
        extractor = FeatureExtractor()
    return model, extractor

# --- Health Check ---
@app.route('/health')
def health():
    return "OK", 200

# --- Frontend Serving ---
@app.route('/')
def serve_index():
    return send_from_directory('website', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join('website', path)):
        return send_from_directory('website', path)
    return send_from_directory('website', 'index.html')

# --- API Routes ---
@app.route('/analyze', methods=['POST'])
def analyze_url():
    try:
        m, ex = get_resources()
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({"error": "URL missing"}), 400

        # 1. Extract features
        features = ex.extract_features(url)
        
        # 2. Site Existence Check
        def si(val): return int(val) if not np.isnan(val) else 0
        def sf(val): return float(val) if not np.isnan(val) else None

        ip_count = si(features[0][102])
        ns_count = si(features[0][103])
        if ip_count == 0 and ns_count == 0:
            return jsonify({"error": "Böyle bir site bulunamadı. Lütfen URL'yi kontrol edin."}), 404

        # 3. Prediction
        proba = m.predict_proba(features)[0][1]
        
        # 4. Feature Parsing
        url_len      = si(features[0][18])
        has_ssl      = sf(features[0][106]) == 1
        has_spf      = sf(features[0][98]) == 1
        domain_age   = sf(features[0][100])
        expiry_days  = sf(features[0][101])
        mx_count     = si(features[0][104])
        asn          = sf(features[0][99])

        security_features = []
        security_features.append({"name": "SSL/TLS Sertifikası", "present": has_ssl, "detail": "HTTPS aktif" if has_ssl else "HTTPS yok"})
        security_features.append({"name": "SPF E-posta Doğrulaması", "present": has_spf, "detail": "SPF kaydı mevcut" if has_spf else "SPF kaydı bulunamadı"})
        security_features.append({"name": "Mail Sunucusu (MX)", "present": mx_count > 0, "detail": f"{mx_count} MX kaydı" if mx_count > 0 else "MX kaydı yok"})
        
        if domain_age is not None:
            age_safe = domain_age > 365
            age_text = f"{int(domain_age/365)} yıl" if domain_age > 365 else f"{int(domain_age)} gün"
            security_features.append({"name": "Domain Yaşı", "present": age_safe, "detail": f"{age_text} - {'köklü' if age_safe else 'yeni'}"})
        else:
            security_features.append({"name": "Domain Yaşı", "present": None, "detail": "WHOIS alınamadı"})

        security_features.append({"name": "DNS Altyapısı (NS)", "present": ns_count >= 2, "detail": f"{ns_count} nameserver"})
        security_features.append({"name": "ASN (Ağ Kimliği)", "present": asn is not None, "detail": f"AS{int(asn)}" if asn else "Bilinmiyor"})
        
        if expiry_days is not None:
            security_features.append({"name": "Domain Bitiş Süresi", "present": expiry_days > 90, "detail": f"{int(expiry_days)} gün kaldı"})

        security_features.append({"name": "URL Uzunluğu", "present": url_len < 50, "detail": f"{url_len} karakter - normal"})

        return jsonify({
            "url": url,
            "overall_score": float(proba * 100),
            "xgboost_score": float(proba * 100),
            "graph_score": float(proba * 100),
            "is_phishing": bool(proba > 0.5),
            "confidence": float(abs(proba - 0.5) * 200),
            "security_features": security_features
        })
    except Exception as e:
        print("!!! ERROR DURING ANALYSIS !!!")
        traceback.print_exc() # Detaylı hata logu
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    from waitress import serve
    print(f"[OK] PhishGuard Live on port {port}")
    serve(app, host='0.0.0.0', port=port)
