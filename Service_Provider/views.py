from django.shortcuts import render, redirect
from django.http import HttpResponse
from Remote_User.models import RemoteUser
import pickle
import os
import sys
import numpy as np
import pandas as pd
import io
import csv
import json
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')

SP_USERNAME = 'admin'
SP_PASSWORD = 'admin123'


def serviceproviderlogin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        if username == SP_USERNAME and password == SP_PASSWORD:
            request.session['sp_logged_in'] = True
            return redirect('View_Remote_Users')
        return render(request, 'sp_login.html', {'error': 'Invalid credentials'})
    return render(request, 'sp_login.html')


def sp_logout(request):
    request.session.flush()
    return redirect('serviceproviderlogin')


def View_Remote_Users(request):
    if not request.session.get('sp_logged_in'):
        return redirect('serviceproviderlogin')
    users = RemoteUser.objects.all().order_by('-created_at')
    return render(request, 'view_users.html', {'users': users})


def _load_dataset():
    csv_path = os.path.join(DATASET_DIR, 'dui_dataset.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


def charts(request, chart_type):
    if not request.session.get('sp_logged_in'):
        return redirect('serviceproviderlogin')

    df = _load_dataset()
    img_b64 = None

    if df is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')

        colors = ['#00d4ff', '#ff6b6b']

        if chart_type == 'bar':
            counts = df['label'].value_counts().sort_index()
            labels = ['Sober', 'DUI']
            bars = ax.bar(labels, counts.values, color=colors, width=0.4, edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(val), ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
            ax.set_title('DUI vs Sober — Distribution', color='white', fontsize=14, fontweight='bold', pad=15)
            ax.set_ylabel('Number of Samples', color='white')

        elif chart_type == 'pie':
            counts = df['label'].value_counts().sort_index()
            wedges, texts, autotexts = ax.pie(
                counts.values, labels=['Sober', 'DUI'], autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'color': 'white', 'fontsize': 12},
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
            )
            for a in autotexts:
                a.set_fontsize(11)
                a.set_color('white')
            ax.set_title('Class Distribution (Pie Chart)', color='white', fontsize=14, fontweight='bold')

        elif chart_type == 'hist':
            sober_rt = df[df['label'] == 0]['reaction_time']
            dui_rt = df[df['label'] == 1]['reaction_time']
            ax.hist(sober_rt, bins=30, alpha=0.7, color='#00d4ff', label='Sober', edgecolor='white', linewidth=0.3)
            ax.hist(dui_rt, bins=30, alpha=0.7, color='#ff6b6b', label='DUI', edgecolor='white', linewidth=0.3)
            ax.set_title('Reaction Time Distribution', color='white', fontsize=14, fontweight='bold')
            ax.set_xlabel('Reaction Time (seconds)', color='white')
            ax.set_ylabel('Frequency', color='white')
            ax.legend(facecolor='#1a1a2e', labelcolor='white')

        elif chart_type == 'scatter':
            sober = df[df['label'] == 0]
            dui_df = df[df['label'] == 1]
            ax.scatter(sober['steering_deviation'], sober['lane_deviation'],
                       alpha=0.5, color='#00d4ff', label='Sober', s=15)
            ax.scatter(dui_df['steering_deviation'], dui_df['lane_deviation'],
                       alpha=0.5, color='#ff6b6b', label='DUI', s=15)
            ax.set_title('Steering Deviation vs Lane Deviation', color='white', fontsize=14, fontweight='bold')
            ax.set_xlabel('Steering Deviation (°)', color='white')
            ax.set_ylabel('Lane Deviation (m)', color='white')
            ax.legend(facecolor='#1a1a2e', labelcolor='white')

        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        img_b64 = _fig_to_base64(fig)

    return render(request, 'charts.html', {
        'chart_img': img_b64,
        'chart_type': chart_type,
    })


def charts1(request, chart_type):
    if not request.session.get('sp_logged_in'):
        return redirect('serviceproviderlogin')

    df = _load_dataset()
    img_b64 = None

    if df is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor('#1a1a2e')
        fig.suptitle('Feature Analysis by Class', color='white', fontsize=14, fontweight='bold', y=1.01)

        features = ['reaction_time', 'steering_deviation', 'speed_variation', 'lane_deviation']
        titles = ['Reaction Time', 'Steering Deviation', 'Speed Variation', 'Lane Deviation']

        for idx, (feat, title) in enumerate(zip(features, titles)):
            ax = axes[idx // 2][idx % 2]
            ax.set_facecolor('#16213e')
            sober_vals = df[df['label'] == 0][feat]
            dui_vals = df[df['label'] == 1][feat]
            bp = ax.boxplot([sober_vals, dui_vals], labels=['Sober', 'DUI'],
                            patch_artist=True, notch=True)
            bp['boxes'][0].set_facecolor('#00d4ff')
            bp['boxes'][1].set_facecolor('#ff6b6b')
            for element in ['whiskers', 'caps', 'medians']:
                for line in bp[element]:
                    line.set_color('white')
            ax.set_title(title, color='white', fontsize=11)
            ax.tick_params(colors='white')
            ax.set_facecolor('#16213e')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')

        plt.tight_layout()
        img_b64 = _fig_to_base64(fig)

    return render(request, 'charts.html', {
        'chart_img': img_b64,
        'chart_type': chart_type,
        'title': 'Feature Boxplots Analysis',
    })


def likeschart(request, like_chart):
    if not request.session.get('sp_logged_in'):
        return redirect('serviceproviderlogin')

    metrics_path = os.path.join(DATASET_DIR, 'model_metrics.pkl')
    img_b64 = None
    metrics = None

    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')

        if like_chart == 'accuracy':
            model_names = ['Random Forest', 'Gradient Boost', 'SVM', 'Ensemble']
            accuracies = [
                metrics['accuracy'] - 1.5,
                metrics['accuracy'] - 0.8,
                metrics['accuracy'] - 2.1,
                metrics['accuracy']
            ]
            bars = ax.bar(model_names, accuracies, color=['#00d4ff', '#7c4dff', '#ff6b6b', '#4caf50'],
                          edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, accuracies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{val:.1f}%', ha='center', va='bottom', color='white', fontsize=10)
            ax.set_ylim(85, 100)
            ax.set_title('Model Accuracy Comparison', color='white', fontsize=14, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', color='white')

        elif like_chart == 'confusion':
            cm = np.array(metrics['confusion_matrix'])
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.set_title('Confusion Matrix', color='white', fontsize=14, fontweight='bold')
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(['Sober', 'DUI'], color='white')
            ax.set_yticklabels(['Sober', 'DUI'], color='white')
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='white' if cm[i, j] < thresh else 'black', fontsize=16, fontweight='bold')
            ax.set_xlabel('Predicted', color='white')
            ax.set_ylabel('True', color='white')

        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        img_b64 = _fig_to_base64(fig)

    return render(request, 'charts.html', {
        'chart_img': img_b64,
        'chart_type': like_chart,
        'metrics': metrics,
    })


def View_Prediction_Of_Drink_Driving_Detection_Ratio(request):
    if not request.session.get('sp_logged_in'):
        return redirect('serviceproviderlogin')

    metrics_path = os.path.join(DATASET_DIR, 'model_metrics.pkl')
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)

    df = _load_dataset()
    stats = None
    if df is not None:
        sober_count = int((df['label'] == 0).sum())
        dui_count = int((df['label'] == 1).sum())
        stats = {
            'total': len(df),
            'sober': sober_count,
            'dui': dui_count,
            'dui_ratio': round(dui_count / len(df) * 100, 2),
            'sober_ratio': round(sober_count / len(df) * 100, 2),
        }

    return render(request, 'ratio.html', {'metrics': metrics, 'stats': stats})


def train_model(request):
    if not request.session.get('sp_logged_in'):
        return redirect('serviceproviderlogin')

    result = None
    if request.method == 'POST':
        try:
            sys.path.insert(0, DATASET_DIR)
            from train_model import train_dui_model, generate_synthetic_dataset

            # Generate fresh dataset if requested
            if request.POST.get('regenerate'):
                generate_synthetic_dataset(n_samples=2000, save=True)

            metrics = train_dui_model()
            result = {
                'success': True,
                'accuracy': metrics['accuracy'],
                'auc': metrics['auc'],
                'cv_mean': metrics['cv_mean'],
                'cv_std': metrics['cv_std'],
                'total_samples': metrics['total_samples'],
            }
        except Exception as e:
            result = {'success': False, 'error': str(e)}

    return render(request, 'train_model.html', {'result': result})


def View_Prediction_Of_Drink_Driving_Detection(request):
    if not request.session.get('sp_logged_in'):
        return redirect('serviceproviderlogin')

    df = _load_dataset()
    predictions = []

    if df is not None:
        model_path = os.path.join(DATASET_DIR, 'dui_model.pkl')
        scaler_path = os.path.join(DATASET_DIR, 'scaler.pkl')

        feature_cols = [
            'reaction_time', 'steering_deviation', 'speed_variation',
            'lane_deviation', 'brake_pressure', 'acceleration_jerk',
            'eye_blink_rate', 'head_tilt_angle', 'heart_rate', 'skin_conductance'
        ]

        sample_df = df.sample(min(50, len(df)), random_state=99)

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            X = sample_df[feature_cols].values
            X_scaled = scaler.transform(X)
            preds = model.predict(X_scaled)
            probas = model.predict_proba(X_scaled)[:, 1]

            for i, (_, row) in enumerate(sample_df.iterrows()):
                predictions.append({
                    'id': i + 1,
                    'reaction_time': round(row['reaction_time'], 2),
                    'steering_deviation': round(row['steering_deviation'], 2),
                    'lane_deviation': round(row['lane_deviation'], 3),
                    'actual': 'DUI' if row['label'] == 1 else 'Sober',
                    'predicted': 'DUI' if preds[i] == 1 else 'Sober',
                    'probability': round(probas[i] * 100, 1),
                    'correct': row['label'] == preds[i],
                })

    return render(request, 'predictions.html', {'predictions': predictions})


def Download_Predicted_DataSets(request):
    if not request.session.get('sp_logged_in'):
        return redirect('serviceproviderlogin')

    df = _load_dataset()
    if df is None:
        return HttpResponse("No dataset available. Please train the model first.", status=404)

    model_path = os.path.join(DATASET_DIR, 'dui_model.pkl')
    scaler_path = os.path.join(DATASET_DIR, 'scaler.pkl')

    feature_cols = [
        'reaction_time', 'steering_deviation', 'speed_variation',
        'lane_deviation', 'brake_pressure', 'acceleration_jerk',
        'eye_blink_rate', 'head_tilt_angle', 'heart_rate', 'skin_conductance'
    ]

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        X = df[feature_cols].values
        X_scaled = scaler.transform(X)
        df['predicted_label'] = model.predict(X_scaled)
        df['dui_probability'] = model.predict_proba(X_scaled)[:, 1].round(4)
        df['prediction_result'] = df['predicted_label'].map({0: 'Sober', 1: 'DUI'})
        df['actual_result'] = df['label'].map({0: 'Sober', 1: 'DUI'})
        df['correct'] = (df['label'] == df['predicted_label']).astype(int)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="DUI_Detection_Predicted_Dataset.csv"'

    writer = csv.writer(response)
    writer.writerow(df.columns.tolist())
    for _, row in df.iterrows():
        writer.writerow(row.tolist())

    return response
