import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='website', static_url_path='')
CORS(app)

# Lazy loading variables
model = None
extractor = None

def get_resources():
    global model, extractor
    if model is None:
        from ml_models import MLModels
        from feature_extractor import FeatureExtractor
        print("[*] Initializing models and extractor...")
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

        features = ex.extract_features(url)
        proba = m.predict_proba(features)[0][1]
        
        # Security feature logic
        def sf(val): return float(val) if not np.isnan(val) else None
        def si(val): return int(val) if not np.isnan(val) else 0

        has_ssl = sf(features[0][106]) == 1
        has_spf = sf(features[0][98]) == 1
        mx_count = si(features[0][104])
        ns_count = si(features[0][103])
        domain_age = sf(features[0][100])
        asn = sf(features[0][99])
        expiry = sf(features[0][101])

        security_features = [
            {"name": "SSL/TLS Sertifikası", "present": has_ssl, "detail": "HTTPS aktif" if has_ssl else "HTTPS yok"},
            {"name": "SPF E-posta Doğrulaması", "present": has_spf, "detail": "SPF mevcut" if has_spf else "SPF yok"},
            {"name": "Mail Sunucusu (MX)", "present": mx_count > 0, "detail": f"{mx_count} MX kaydı"},
            {"name": "DNS Altyapısı (NS)", "present": ns_count >= 2, "detail": f"{ns_count} NS kaydı"},
            {"name": "ASN (Ağ Kimliği)", "present": asn is not None, "detail": f"AS{int(asn)}" if asn else "Bilinmiyor"}
        ]
        
        if domain_age is not None:
            security_features.append({"name": "Domain Yaşı", "present": domain_age > 365, "detail": f"{int(domain_age)} gün"})

        return jsonify({
            "url": url,
            "overall_score": float(proba * 100),
            "xgboost_score": float(proba * 100),
            "graph_score": float(proba * 100),
            "is_phishing": bool(proba > 0.5),
            "security_features": security_features
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    from waitress import serve
    serve(app, host='0.0.0.0', port=port)
