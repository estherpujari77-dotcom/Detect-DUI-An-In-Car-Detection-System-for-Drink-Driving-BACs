from django.shortcuts import render, redirect
from django.contrib import messages as django_messages
from .models import RemoteUser
import pickle
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'Dataset', 'dui_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'Dataset', 'scaler.pkl')


def index(request):
    return render(request, 'index.html')


def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            user = RemoteUser.objects.get(username=username, password=password)
            request.session['user_id'] = user.id
            request.session['username'] = user.username
            return redirect('Predict_Drink_Driving_Detection')
        except RemoteUser.DoesNotExist:
            django_messages.error(request, 'Invalid username or password.')
            return render(request, 'login.html', {'error': 'Invalid credentials'})
    return render(request, 'login.html')


def Register1(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        email = request.POST.get('email')
        phone = request.POST.get('phone', '')
        address = request.POST.get('address', '')

        if RemoteUser.objects.filter(username=username).exists():
            return render(request, 'register.html', {'error': 'Username already exists'})
        if RemoteUser.objects.filter(email=email).exists():
            return render(request, 'register.html', {'error': 'Email already registered'})

        user = RemoteUser.objects.create(
            username=username,
            password=password,
            email=email,
            phone=phone,
            address=address
        )
        return redirect('login')
    return render(request, 'register.html')


def Predict_Drink_Driving_Detection(request):
    if 'user_id' not in request.session:
        return redirect('login')

    result = None
    prediction_label = None
    risk_level = None
    feature_values = None

    if request.method == 'POST':
        try:
            # Collect all sensor/behavioral features
            # Blood Alcohol Content related features
            reaction_time = float(request.POST.get('reaction_time', 0))       # seconds
            steering_deviation = float(request.POST.get('steering_deviation', 0))  # degrees
            speed_variation = float(request.POST.get('speed_variation', 0))    # km/h std
            lane_deviation = float(request.POST.get('lane_deviation', 0))      # meters
            brake_pressure = float(request.POST.get('brake_pressure', 0))     # 0-100
            acceleration_jerk = float(request.POST.get('acceleration_jerk', 0)) # m/s^3
            eye_blink_rate = float(request.POST.get('eye_blink_rate', 0))      # blinks/min
            head_tilt_angle = float(request.POST.get('head_tilt_angle', 0))    # degrees
            heart_rate = float(request.POST.get('heart_rate', 0))              # bpm
            skin_conductance = float(request.POST.get('skin_conductance', 0))  # μS

            feature_values = {
                'Reaction Time (s)': reaction_time,
                'Steering Deviation (°)': steering_deviation,
                'Speed Variation (km/h)': speed_variation,
                'Lane Deviation (m)': lane_deviation,
                'Brake Pressure': brake_pressure,
                'Acceleration Jerk': acceleration_jerk,
                'Eye Blink Rate (bpm)': eye_blink_rate,
                'Head Tilt Angle (°)': head_tilt_angle,
                'Heart Rate (bpm)': heart_rate,
                'Skin Conductance (μS)': skin_conductance,
            }

            features = np.array([[
                reaction_time, steering_deviation, speed_variation,
                lane_deviation, brake_pressure, acceleration_jerk,
                eye_blink_rate, head_tilt_angle, heart_rate, skin_conductance
            ]])

            # Load model and predict
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                with open(SCALER_PATH, 'rb') as f:
                    scaler = pickle.load(f)

                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]
                proba = model.predict_proba(features_scaled)[0]

                dui_probability = float(proba[1]) * 100

                if prediction == 1:
                    result = 'DUI DETECTED'
                    prediction_label = 'danger'
                    if dui_probability >= 80:
                        risk_level = 'HIGH RISK'
                    else:
                        risk_level = 'MODERATE RISK'
                else:
                    result = 'SOBER / SAFE'
                    prediction_label = 'success'
                    risk_level = 'LOW RISK'
                    dui_probability = 100 - dui_probability

            else:
                # Rule-based fallback if model not trained yet
                score = 0
                if reaction_time > 1.5: score += 2
                if steering_deviation > 15: score += 2
                if speed_variation > 20: score += 1
                if lane_deviation > 0.5: score += 2
                if eye_blink_rate < 10 or eye_blink_rate > 30: score += 1
                if head_tilt_angle > 20: score += 1

                if score >= 5:
                    result = 'DUI DETECTED (Rule-Based)'
                    prediction_label = 'danger'
                    risk_level = 'HIGH RISK'
                    dui_probability = min(score * 15, 95)
                else:
                    result = 'SOBER / SAFE (Rule-Based)'
                    prediction_label = 'success'
                    risk_level = 'LOW RISK'
                    dui_probability = score * 10

        except Exception as e:
            result = f'Error: {str(e)}'
            prediction_label = 'warning'
            risk_level = 'UNKNOWN'
            dui_probability = 0

        return render(request, 'predict.html', {
            'result': result,
            'prediction_label': prediction_label,
            'risk_level': risk_level,
            'feature_values': feature_values,
            'dui_probability': round(dui_probability, 2),
            'username': request.session.get('username'),
        })

    return render(request, 'predict.html', {'username': request.session.get('username')})


def ViewYourProfile(request):
    if 'user_id' not in request.session:
        return redirect('login')
    user = RemoteUser.objects.get(id=request.session['user_id'])
    return render(request, 'profile.html', {'user': user})


def logout_view(request):
    request.session.flush()
    return redirect('index')
