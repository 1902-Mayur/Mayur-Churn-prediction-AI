"""
Customer Churn Prediction System — Flask Web Application
Advanced interactive dashboard with batch prediction, insights & retention strategies
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import pandas as pd
import numpy as np
import joblib, json, io, os, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'churn_secret_2024'

BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, 'models')

# ── Load artifacts ─────────────────────────────────────────────────────────
model        = joblib.load(f'{MODELS}/best_model.pkl')
scaler       = joblib.load(f'{MODELS}/scaler.pkl')
imputer      = joblib.load(f'{MODELS}/imputer.pkl')
feature_names = joblib.load(f'{MODELS}/feature_names.pkl')
with open(f'{MODELS}/metadata.json') as f:
    metadata = json.load(f)

DARK   = '#0d0d14'
ACCENT = '#e94560'
TEAL   = '#00d4aa'
GOLD   = '#f5a623'
TEXT   = '#e2e8f0'

# ── Helper: build feature vector ───────────────────────────────────────────
INTERNET_MAP  = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
CONTRACT_MAP  = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
PAYMENT_MAP   = {
    'Bank transfer (automatic)': 0,
    'Credit card (automatic)': 1,
    'Electronic check': 2,
    'Mailed check': 3
}
YESNO_MAP = {'Yes': 1, 'No': 0}
SERVICE_MAP = {'Yes': 1, 'No': 0, 'No internet service': 2, 'No phone service': 2}

def build_features(form_data):
    tenure         = float(form_data.get('tenure', 12))
    monthly        = float(form_data.get('monthly_charges', 65))
    tc_raw = form_data.get('total_charges', '').strip() if hasattr(form_data.get('total_charges',''), 'strip') else ''
    total = float(tc_raw) if tc_raw else monthly * tenure
    gender         = 1 if form_data.get('gender','Male') == 'Male' else 0
    senior         = int(form_data.get('senior_citizen', 0))
    partner        = YESNO_MAP.get(form_data.get('partner','No'), 0)
    dependents     = YESNO_MAP.get(form_data.get('dependents','No'), 0)
    phone          = YESNO_MAP.get(form_data.get('phone_service','Yes'), 1)
    multi_lines    = SERVICE_MAP.get(form_data.get('multiple_lines','No'), 0)
    internet       = INTERNET_MAP.get(form_data.get('internet_service','Fiber optic'), 1)
    online_sec     = SERVICE_MAP.get(form_data.get('online_security','No'), 0)
    online_bk      = SERVICE_MAP.get(form_data.get('online_backup','No'), 0)
    dev_prot       = SERVICE_MAP.get(form_data.get('device_protection','No'), 0)
    tech_sup       = SERVICE_MAP.get(form_data.get('tech_support','No'), 0)
    stream_tv      = SERVICE_MAP.get(form_data.get('streaming_tv','No'), 0)
    stream_mv      = SERVICE_MAP.get(form_data.get('streaming_movies','No'), 0)
    contract       = CONTRACT_MAP.get(form_data.get('contract','Month-to-month'), 0)
    paperless      = YESNO_MAP.get(form_data.get('paperless_billing','Yes'), 1)
    payment        = PAYMENT_MAP.get(form_data.get('payment_method','Electronic check'), 2)

    # Engineered
    avg_monthly    = total / (tenure + 1)
    charge_per_srv = monthly / (phone + max(internet,1) + 1)
    tenure_group   = 0 if tenure<=12 else (1 if tenure<=24 else (2 if tenure<=48 else 3))
    is_long_term   = int(tenure > 24)
    num_services   = sum([int(online_sec > 0), int(online_bk > 0), int(dev_prot > 0),
                          int(tech_sup > 0), int(stream_tv > 0), int(stream_mv > 0)])
    engagement     = (num_services*0.4 + is_long_term*0.3 + (1-senior)*0.15 + partner*0.075 + dependents*0.075)
    high_spender   = int(monthly > 79.5)
    low_sup_risk   = int(tech_sup == 0 and online_sec == 0)

    row = [gender, senior, partner, dependents, tenure, phone, multi_lines,
           internet, online_sec, online_bk, dev_prot, tech_sup, stream_tv,
           stream_mv, contract, paperless, payment, monthly, total,
           avg_monthly, charge_per_srv, tenure_group, is_long_term, num_services,
           engagement, high_spender, low_sup_risk]
    return np.array(row).reshape(1, -1)


def predict_churn(features):
    features_imp = imputer.transform(features)
    features_sc  = scaler.transform(features_imp)
    prob   = model.predict_proba(features_sc)[0][1]
    pred   = int(prob >= 0.5)
    risk   = 'High' if prob >= 0.6 else ('Medium' if prob >= 0.3 else 'Low')
    return pred, float(prob), risk


def get_retention_tips(form_data, prob):
    tips = []
    contract = form_data.get('contract', 'Month-to-month')
    monthly  = float(form_data.get('monthly_charges', 65))
    tech_sup = form_data.get('tech_support', 'No')
    online_sec = form_data.get('online_security', 'No')
    tenure   = float(form_data.get('tenure', 12))

    if contract == 'Month-to-month':
        tips.append({'icon': '📋', 'title': 'Upgrade Contract',
                     'desc': 'Offer a 20% discount to switch to Annual plan. Annual customers churn 3× less.'})
    if monthly > 80:
        tips.append({'icon': '💰', 'title': 'Loyalty Discount',
                     'desc': f'Apply a 15% loyalty discount (${monthly*0.15:.0f}/mo savings) to retain high-value customer.'})
    if tech_sup == 'No':
        tips.append({'icon': '🛠️', 'title': 'Free Tech Support Trial',
                     'desc': 'Offer 3 months of free Tech Support. Customers with support churn 40% less.'})
    if online_sec == 'No':
        tips.append({'icon': '🔒', 'title': 'Security Bundle',
                     'desc': 'Bundle Online Security at no charge for 6 months. High impact on satisfaction.'})
    if tenure < 12:
        tips.append({'icon': '🎁', 'title': 'New Customer Reward',
                     'desc': 'Enroll in First-Year Loyalty Program: bonus data + priority support.'})
    if prob > 0.75:
        tips.append({'icon': '📞', 'title': 'Personal Outreach',
                     'desc': 'Assign dedicated account manager. High-risk customers respond well to personal contact.'})
    if not tips:
        tips.append({'icon': '⭐', 'title': 'Engagement Program',
                     'desc': 'Customer is low-risk. Enroll in referral program to boost loyalty further.'})
    return tips


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor=DARK)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', metadata=metadata)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    form_vals = {}
    if request.method == 'POST':
        form_vals = request.form.to_dict()
        features = build_features(form_vals)
        pred, prob, risk = predict_churn(features)
        tips = get_retention_tips(form_vals, prob)

        # Gauge chart
        fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor(DARK)
        theta  = np.linspace(np.pi, 0, 200)
        colors = [TEAL if t > np.pi*(1-prob) else ACCENT for t in theta]
        for i in range(len(theta)-1):
            ax.plot([theta[i], theta[i+1]], [1, 1], color=colors[i], lw=10)
        ax.plot([np.pi*(1-prob)], [1], 'o', color='white', ms=10, zorder=5)
        ax.set_ylim(0, 1.3)
        ax.axis('off')
        ax.set_title(f'{prob*100:.1f}%', fontsize=22, fontweight='bold',
                     color=ACCENT if prob>0.5 else TEAL, pad=5)
        gauge_img = fig_to_base64(fig)

        result = {
            'prediction': pred, 'probability': prob,
            'confidence': f'{prob*100:.1f}%',
            'risk': risk, 'tips': tips,
            'gauge_img': gauge_img,
            'label': 'LIKELY TO CHURN' if pred else 'LIKELY TO RETAIN',
        }
    return render_template('predict.html', result=result, form_vals=form_vals)


@app.route('/batch', methods=['GET', 'POST'])
def batch():
    results_df = None
    chart_img = None
    error = None
    if request.method == 'POST':
        try:
            file = request.files.get('csv_file')
            if not file:
                raise ValueError("No file uploaded.")
            df = pd.read_csv(file)

            # Map CSV columns to model features
            rows = []
            for _, row in df.iterrows():
                form_like = {
                    'tenure':           str(row.get('tenure', 12)),
                    'monthly_charges':  str(row.get('MonthlyCharges', 65)),
                    'total_charges':    str(row.get('TotalCharges', '')),
                    'gender':           str(row.get('gender', 'Male')),
                    'senior_citizen':   str(int(row.get('SeniorCitizen', 0))),
                    'partner':          str(row.get('Partner', 'No')),
                    'dependents':       str(row.get('Dependents', 'No')),
                    'phone_service':    str(row.get('PhoneService', 'Yes')),
                    'multiple_lines':   str(row.get('MultipleLines', 'No')),
                    'internet_service': str(row.get('InternetService', 'Fiber optic')),
                    'online_security':  str(row.get('OnlineSecurity', 'No')),
                    'online_backup':    str(row.get('OnlineBackup', 'No')),
                    'device_protection':str(row.get('DeviceProtection', 'No')),
                    'tech_support':     str(row.get('TechSupport', 'No')),
                    'streaming_tv':     str(row.get('StreamingTV', 'No')),
                    'streaming_movies': str(row.get('StreamingMovies', 'No')),
                    'contract':         str(row.get('Contract', 'Month-to-month')),
                    'paperless_billing':str(row.get('PaperlessBilling', 'Yes')),
                    'payment_method':   str(row.get('PaymentMethod', 'Electronic check')),
                }
                feat = build_features(form_like)
                pred, prob, risk = predict_churn(feat)
                rows.append({'ChurnPrediction': 'Yes' if pred else 'No',
                             'ChurnProbability': round(prob, 4),
                             'RiskSegment': risk})

            result_df = pd.concat([df.reset_index(drop=True),
                                   pd.DataFrame(rows)], axis=1)

            # Summary chart
            risk_counts = pd.DataFrame(rows)['RiskSegment'].value_counts()
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.patch.set_facecolor(DARK)
            seg_colors = {'Low': TEAL, 'Medium': GOLD, 'High': ACCENT}
            axes[0].bar(risk_counts.index,
                        risk_counts.values,
                        color=[seg_colors.get(r, TEAL) for r in risk_counts.index],
                        edgecolor=DARK)
            axes[0].set_title('Risk Segment Distribution', color=TEXT, fontweight='bold')
            axes[0].set_ylabel('Count', color=TEXT)
            for spine in axes[0].spines.values(): spine.set_visible(False)

            probs = [r['ChurnProbability'] for r in rows]
            axes[1].hist(probs, bins=20, color=ACCENT, alpha=0.8, edgecolor=DARK)
            axes[1].axvline(0.5, color=GOLD, ls='--', lw=1.5, label='Threshold')
            axes[1].set_title('Churn Probability Distribution', color=TEXT, fontweight='bold')
            axes[1].set_xlabel('Probability', color=TEXT)
            axes[1].legend()
            for spine in axes[1].spines.values(): spine.set_visible(False)
            plt.tight_layout()
            chart_img = fig_to_base64(fig)

            results_df = result_df.head(50).to_html(
                classes='results-table', index=False,
                border=0, justify='left')

        except Exception as e:
            error = str(e)

    return render_template('batch.html', results_df=results_df,
                           chart_img=chart_img, error=error)


@app.route('/dashboard')
def dashboard():
    # Load visuals
    visual_files = {
        'eda':         f'{BASE}/visuals/01_eda_overview.png',
        'heatmap':     f'{BASE}/visuals/02_correlation_heatmap.png',
        'engineered':  f'{BASE}/visuals/03_engineered_features.png',
        'comparison':  f'{BASE}/visuals/04_model_comparison.png',
        'evaluation':  f'{BASE}/visuals/05_best_model_evaluation.png',
        'importance':  f'{BASE}/visuals/06_feature_importance.png',
        'business':    f'{BASE}/visuals/07_business_dashboard.png',
    }
    images = {}
    for key, path in visual_files.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                images[key] = base64.b64encode(f.read()).decode('utf-8')
    return render_template('dashboard.html', images=images, metadata=metadata)


@app.route('/api/quick_predict', methods=['POST'])
def api_quick_predict():
    data = request.json or {}
    features = build_features(data)
    pred, prob, risk = predict_churn(features)
    return jsonify({
        'churn': bool(pred), 'probability': prob,
        'confidence': f'{prob*100:.1f}%', 'risk_segment': risk
    })


if __name__ == '__main__':
    app.run(debug=True, port=5050)
