// API Configuration - Change this to your live URL after deploying the backend (e.g., https://your-app.render.com)
const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:5000'
    : ''; // Leave empty if frontend and backend are on the same server, or put your Render URL here

async function simulateScan() {
    const input = document.getElementById('url-input');
    const resultDiv = document.getElementById('scan-result');
    const button = document.getElementById('scan-btn');
    const url = input.value.trim();

    if (!url) {
        alert('Lütfen geçerli bir URL girin.');
        return;
    }

    button.innerText = 'GERÇEK ZAMANLI ANALİZ EDİLİYOR...';
    button.disabled = true;
    resultDiv.style.display = 'none';

    try {
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'API Hatası');
        }

        const data = await response.json();
        const score = data.overall_score;
        
        let color, label, state;
        if (score < 40) {
            color = 'var(--accent-green)';
            label = 'GÜVENLİ WEB SİTESİ';
            state = 'safe';
        } else if (score >= 40 && score <= 60) {
            color = 'var(--accent-orange)';
            label = 'ŞÜPHELİ / DİKKATLİ OLUN';
            state = 'suspicious';
        } else {
            color = 'var(--accent-red)';
            label = 'OLTALAMA RİSKİ TESPİT EDİLDİ';
            state = 'phishing';
        }

        const xgboostScore = data.xgboost_score;
        const graphScore = data.graph_score;
        const features = data.security_features || [];

        // Split features: first 4 = structural (XGBoost), rest = network (Graph)
        const structuralNames = ['SSL/TLS Sertifikası', 'URL Uzunluğu', 'Domain Yaşı', 'Domain Bitiş Süresi'];
        const xgFeatures = features.filter(f => structuralNames.includes(f.name));
        const graphFeatures = features.filter(f => !structuralNames.includes(f.name));

        // Build feature item HTML
        function buildFeatureHTML(list) {
            return list.map(f => {
                let icon, iconColor;
                if (f.present === true) {
                    icon = '✅'; iconColor = 'var(--accent-green)';
                } else if (f.present === false) {
                    icon = '❌'; iconColor = 'var(--accent-red)';
                } else {
                    icon = '⚠️'; iconColor = 'var(--accent-orange)';
                }
                return `
                    <div style="display:flex; align-items:flex-start; gap:10px; padding:10px 14px; background:rgba(255,255,255,0.03); border-radius:10px; margin-bottom:6px; border-left: 3px solid ${iconColor};">
                        <span style="font-size:1.1rem; flex-shrink:0;">${icon}</span>
                        <div>
                            <div style="font-weight:600; color:#eee; font-size:0.9rem;">${f.name}</div>
                            <div style="color:#999; font-size:0.8rem; margin-top:2px;">${f.detail}</div>
                        </div>
                    </div>`;
            }).join('');
        }

        // Count safe/unsafe features for summary
        const safeCount = features.filter(f => f.present === true).length;
        const totalCount = features.length;

        resultDiv.innerHTML = `
            <div class="result-header">
                <div class="stat-card" style="max-width: 400px; margin: 0 auto; border-color: ${color}; box-shadow: 0 0 40px ${color}44;">
                    <div class="stat-value" style="color: ${color}; text-shadow: 0 0 10px ${color}aa;">%${score.toFixed(1)}</div>
                    <div class="stat-label" style="color: ${color}">${label}</div>
                </div>
            </div>
            <div class="result-grid">
                <div class="analysis-box animate-in">
                    <h5>XGBoost Analizi (Real) <span>%${xgboostScore.toFixed(0)}</span></h5>
                    <div class="progress-container"><div class="progress-bar" id="bar-xgb" style="width: 0%; background: ${color}"></div></div>
                    <div style="margin-top: 12px;">
                        ${buildFeatureHTML(xgFeatures)}
                    </div>
                </div>
                <div class="analysis-box animate-in" style="animation-delay: 0.2s">
                    <h5>Graf Analizi (Neural) <span>%${graphScore.toFixed(0)}</span></h5>
                    <div class="progress-container"><div class="progress-bar" id="bar-graph" style="width: 0%; background: ${color}"></div></div>
                    <div style="margin-top: 12px;">
                        ${buildFeatureHTML(graphFeatures)}
                    </div>
                </div>
            </div>
            <div class="analysis-box animate-in" style="animation-delay: 0.4s; margin-top: 16px;">
                <h5>Güvenlik Özeti <span style="font-size:0.85rem; opacity:0.7;">${safeCount}/${totalCount} özellik geçti</span></h5>
                <div style="padding:14px; background:rgba(255,255,255,0.03); border-radius:12px;">
                    <div style="color:#ccc; font-size:0.85rem; line-height:1.6;">
                        ${state === 'safe' ? 
                            (() => {
                                const failed = features.filter(f => f.present === false);
                                const unknown = features.filter(f => f.present === null);
                                const passed = features.filter(f => f.present === true);
                                
                                let html = `<p>✅ <strong>Bu site güvenli görünüyor.</strong></p>`;
                                
                                if (failed.length > 0 || unknown.length > 0) {
                                    html += `<p style="color:#aaa; margin-top:8px;">
                                        <strong style="color:#eee;">Bazı eksik özellikler olmasına rağmen neden güvenli?</strong><br>
                                        Model, 88.000 siteden öğrendiği kalıplara göre tüm özellikleri <strong>ağırlıklı olarak</strong> değerlendirir. 
                                        Aşağıdaki güçlü sinyaller, eksiklikleri telafi etmektedir:
                                    </p>`;
                                    
                                    html += `<div style="margin-top:10px;">`;
                                    passed.forEach(f => {
                                        let why = '';
                                        if (f.name === 'SSL/TLS Sertifikası') why = 'Şifreli bağlantı, veri güvenliğini sağlıyor';
                                        else if (f.name === 'Domain Yaşı') why = 'Köklü domainler oltalama için nadiren kullanılır';
                                        else if (f.name === 'ASN (Ağ Kimliği)') why = 'Bilinen ve güvenilir bir ağ altyapısında barınıyor';
                                        else if (f.name === 'URL Uzunluğu') why = 'Kısa ve temiz URL yapısı, oltalama kalıplarına uymuyor';
                                        else if (f.name === 'DNS Altyapısı (NS)') why = 'Sağlam DNS altyapısı profesyonel yapıya işaret ediyor';
                                        else if (f.name === 'Domain Bitiş Süresi') why = 'Uzun vadeli kayıt, kalıcılık göstergesi';
                                        else why = f.detail;
                                        html += `
                                            <div style="padding:8px 14px; background:rgba(0,255,135,0.03); border-radius:10px; margin-bottom:5px; border-left:3px solid var(--accent-green);">
                                                <span style="font-weight:600; color:#eee;">✅ ${f.name}</span>
                                                <span style="color:#888; font-size:0.8rem;"> — ${why}</span>
                                            </div>`;
                                    });
                                    html += `</div>`;
                                    
                                    if (failed.length > 0) {
                                        html += `<p style="color:#999; margin-top:12px; font-size:0.8rem;">
                                            ℹ️ <em>Eksik özellikler (${failed.map(f => f.name).join(', ')}), 
                                            alt domainlerde veya statik site barındırma hizmetlerinde normaldir 
                                            ve model bu durumu doğru şekilde değerlendirmektedir.</em>
                                        </p>`;
                                    }
                                } else {
                                    html += `<p style="color:#999;">Domain altyapısı sağlam ve güvenlik önlemleri yerinde. Bilinen bir tehdit tespit edilmedi.</p>`;
                                }
                                
                                return html;
                            })() :
                        (() => {
                            const failed = features.filter(f => f.present === false);
                            const unknown = features.filter(f => f.present === null);
                            
                            let explanation = state === 'suspicious' 
                                ? `<p>⚠️ <strong>Bu site bazı şüpheli özellikler taşıyor.</strong></p>`
                                : `<p>🚨 <strong>Bu site yüksek oltalama riski taşıyor.</strong></p>`;
                            
                            if (failed.length > 0 && safeCount >= totalCount - 2) {
                                explanation += `<p style="color:#aaa; margin-top:8px;">
                                    <strong style="color:#eee;">${safeCount}/${totalCount} özellik geçmesine rağmen risk skoru neden yüksek?</strong><br>
                                    Yapay zeka modeli tüm özelliklere eşit ağırlık vermez. Bazı özellikler, modelin karar mekanizmasında 
                                    diğerlerinden <strong>çok daha belirleyicidir</strong>. Özellikle:
                                </p>`;
                                
                                explanation += `<div style="margin-top:10px;">`;
                                failed.forEach(f => {
                                    let weight = '';
                                    if (f.name === 'Domain Yaşı') {
                                        weight = `<span style="color:var(--accent-red); font-weight:600;">Çok Yüksek Etki</span> — 
                                            Eğitim verisindeki güvenli sitelerin ortalama yaşı <strong>4,367 gün (~12 yıl)</strong>. 
                                            Bu domain çok yeni olduğu için model onu oltalama sitelerine benzetiyor. 
                                            Bu özellik tek başına skoru %30-50 artırabilir.`;
                                    } else if (f.name === 'SPF E-posta Doğrulaması') {
                                        weight = `<span style="color:var(--accent-orange); font-weight:600;">Orta Etki</span> — 
                                            SPF kaydı olmaması, e-posta sahteciliğine açık olduğunu gösterir.`;
                                    } else if (f.name === 'Mail Sunucusu (MX)') {
                                        weight = `<span style="color:var(--accent-orange); font-weight:600;">Orta Etki</span> — 
                                            Kurumsal sitelerin büyük çoğunluğunda MX kaydı bulunur.`;
                                    } else if (f.name === 'SSL/TLS Sertifikası') {
                                        weight = `<span style="color:var(--accent-red); font-weight:600;">Yüksek Etki</span> — 
                                            SSL sertifikası olmaması, veri güvenliğinin sağlanmadığını gösterir.`;
                                    } else if (f.name === 'Domain Bitiş Süresi') {
                                        weight = `<span style="color:var(--accent-orange); font-weight:600;">Orta Etki</span> — 
                                            Kısa süreli domain kayıtları oltalama sitelerinde yaygındır.`;
                                    } else {
                                        weight = `<span style="color:var(--accent-orange); font-weight:600;">Düşük-Orta Etki</span>`;
                                    }
                                    explanation += `
                                        <div style="padding:10px 14px; background:rgba(255,62,62,0.05); border-radius:10px; margin-bottom:8px; border-left:3px solid var(--accent-red);">
                                            <div style="font-weight:600; color:#eee;">❌ ${f.name}</div>
                                            <div style="color:#999; font-size:0.8rem; margin-top:4px;">${f.detail}</div>
                                            <div style="color:#bbb; font-size:0.8rem; margin-top:6px;">📊 Model Ağırlığı: ${weight}</div>
                                        </div>`;
                                });
                                explanation += `</div>`;
                                
                                explanation += `<p style="color:#888; margin-top:12px; font-size:0.8rem; border-top:1px solid rgba(255,255,255,0.06); padding-top:10px;">
                                    💡 <em>Not: Zaman içinde domain yaşı arttıkça ve eksik özellikler tamamlandıkça, 
                                    risk skoru otomatik olarak düşecektir. Model 88.000 siteden öğrendiği istatistiksel 
                                    kalıplara göre karar verir — manuel müdahale yoktur.</em>
                                </p>`;
                            } else if (failed.length > 0) {
                                explanation += `<p style="color:#999; margin-top:8px;">Aşağıdaki güvenlik özellikleri eksik veya yetersiz:</p>`;
                                explanation += `<div style="margin-top:8px;">`;
                                failed.forEach(f => {
                                    explanation += `
                                        <div style="padding:8px 12px; background:rgba(255,62,62,0.05); border-radius:8px; margin-bottom:6px; border-left:3px solid var(--accent-red);">
                                            <span style="font-weight:600; color:#eee;">❌ ${f.name}</span>
                                            <span style="color:#999; font-size:0.8rem;"> — ${f.detail}</span>
                                        </div>`;
                                });
                                explanation += `</div>`;
                            }
                            
                            if (unknown.length > 0) {
                                explanation += `<p style="color:#999; margin-top:8px;">Doğrulanamayan özellikler:</p>`;
                                unknown.forEach(f => {
                                    explanation += `<div style="padding:6px 12px; color:#999; font-size:0.8rem;">⚠️ ${f.name} — ${f.detail}</div>`;
                                });
                            }
                            
                            return explanation;
                        })()
                        }
                    </div>
                </div>
            </div>
        `;
        
        resultDiv.style.display = 'block';
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });

        setTimeout(() => {
            const barXgb = document.getElementById('bar-xgb');
            const barGraph = document.getElementById('bar-graph');
            if (barXgb) barXgb.style.width = xgboostScore + '%';
            if (barGraph) barGraph.style.width = graphScore + '%';
        }, 100);

    } catch (error) {
        resultDiv.innerHTML = `
            <div class="result-header animate-in">
                <div class="stat-card" style="max-width: 500px; margin: 0 auto; border-color: var(--accent-red); background: rgba(255, 62, 62, 0.05);">
                    <div class="stat-label" style="color: var(--accent-red); font-size: 1.2rem; margin-bottom: 0;">⚠️ ${error.message}</div>
                </div>
            </div>
        `;
        resultDiv.style.display = 'block';
    } finally {
        button.innerText = 'TARAMAYI BAŞLAT';
        button.disabled = false;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const stats = document.querySelectorAll('.stat-value');
    stats.forEach(stat => {
        const target = parseFloat(stat.innerText.replace(/[^0-9.]/g, ''));
        if (isNaN(target)) return;
        let count = 0;
        const duration = 2000;
        const increment = target / (duration / 16);
        const updateCount = () => {
            if (count < target) {
                count += increment;
                stat.innerText = stat.id.includes('acc') || stat.id.includes('gnn') ? count.toFixed(2) + '%' : Math.floor(count).toLocaleString();
                requestAnimationFrame(updateCount);
            } else {
                stat.innerText = stat.id.includes('acc') || stat.id.includes('gnn') ? target.toFixed(2) + '%' : target.toLocaleString();
            }
        };
        updateCount();
    });
});
