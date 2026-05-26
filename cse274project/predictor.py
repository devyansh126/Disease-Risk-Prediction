import pickle
import numpy as np
import os

class Predictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self._models   = {}
        self._scalers  = {}
        self._meta     = {}
        self._load_all()

    def _load_pkl(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_all(self):
        diseases = ['diabetes', 'heart', 'obesity', 'sleep_apnea']
        scaler_names = {
            'diabetes':   'diabetes_scaler.pkl',
            'heart':      'heart_scaler.pkl',
            'obesity':    'obesity_scaler.pkl',
            'sleep_apnea':'sleep_apnea_scaler.pkl',
        }
        for d in diseases:
            model_path  = os.path.join(self.model_dir, f'{d}.pkl')
            scaler_path = os.path.join(self.model_dir, scaler_names[d])
            if os.path.exists(model_path):
                self._models[d]  = self._load_pkl(model_path)
                self._scalers[d] = self._load_pkl(scaler_path)
                print(f'  [predictor] loaded {d}')
            else:
                print(f'  [predictor] WARNING: {model_path} not found — run train.py first')

        # Obesity needs label encoders
        enc_path = os.path.join(self.model_dir, 'obesity_encoders.pkl')
        if os.path.exists(enc_path):
            self._meta['obesity_encoders'] = self._load_pkl(enc_path)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _risk_level(self, prob):
        if prob < 0.30:  return 'low',      '#00e5a0'
        if prob < 0.60:  return 'moderate',  '#ff9500'
        return               'high',     '#ff3b5c'

    def _build_result(self, model_pkg, prob_array, is_multiclass):
        model     = model_pkg['model']
        classes   = model_pkg['classes']
        threshold = model_pkg.get('threshold', 0.5)

        if is_multiclass:
            pred_idx   = int(np.argmax(prob_array))
            pred_label = classes[pred_idx]
            prob_pct   = round(float(prob_array[pred_idx]) * 100, 1)
            risk, color = self._risk_level(prob_array[pred_idx])
            all_probs  = {classes[i]: round(float(prob_array[i]) * 100, 1)
                          for i in range(len(classes))}
            return {
                'prediction': pred_label,
                'probability': prob_pct,
                'risk_level':  risk,
                'color':       color,
                'all_probs':   all_probs,
                'multiclass':  True,
            }
        else:
            prob_pos   = float(prob_array[1])
            pred       = int(prob_pos >= threshold)
            pred_label = classes[pred]
            prob_pct   = round(prob_pos * 100, 1)
            risk, color = self._risk_level(prob_pos)
            return {
                'prediction': pred_label,
                'probability': prob_pct,
                'risk_level':  risk,
                'color':       color,
                'multiclass':  False,
            }

    # ── per-disease preprocessing ──────────────────────────────────────────────

    def _prep_diabetes(self, raw):
        # DiabetesPedigreeFunction was dropped by VarianceThreshold in train.py
        # Scaler was fitted on exactly these 7 features — order must match
        cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'Age']
        return np.array([[float(raw[c]) for c in cols]])

    def _prep_heart(self, raw):
        cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
                'thalach','exang','oldpeak','slope','ca','thal']
        return np.array([[float(raw[c]) for c in cols]])

    def _prep_obesity(self, raw):
        import pandas as pd
        cat_cols = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',
                    'family_history_with_overweight', 'CAEC', 'MTRANS']
        num_cols = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        encoders = self._meta.get('obesity_encoders', {})

        row = {}
        for c in num_cols:
            row[c] = float(raw[c])
        for c in cat_cols:
            le = encoders.get(c)
            val = str(raw[c])
            if le is not None and val in le.classes_:
                row[c] = int(le.transform([val])[0])
            else:
                row[c] = -1

        # Order must match training feature order
        feature_order = ['Age', 'Gender', 'CALC', 'FAVC', 'FCVC', 'NCP', 'SCC',
                         'SMOKE', 'CH2O', 'family_history_with_overweight',
                         'FAF', 'TUE', 'CAEC', 'MTRANS']
        return np.array([[row[c] for c in feature_order]])

    def _prep_sleep(self, raw):
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder

        cat_map = {
            'Gender':       {'Female': 0, 'Male': 1},
            'BMI Category': {'Normal': 0, 'Obese': 1, 'Overweight': 2},
        }
        # Occupation — simple alphabetical label encoding matching train
        occupations = sorted([
            'Accountant', 'Doctor', 'Engineer', 'Lawyer', 'Manager',
            'Nurse', 'Sales Representative', 'Salesperson', 'Scientist',
            'Software Engineer', 'Teacher'
        ])
        occ_map = {v: i for i, v in enumerate(occupations)}

        row = {
            'Gender':                  cat_map['Gender'].get(raw['Gender'], 0),
            'Age':                     float(raw['Age']),
            'Occupation':              occ_map.get(raw['Occupation'], 0),
            'Sleep Duration':          float(raw['Sleep Duration']),
            'Quality of Sleep':        float(raw['Quality of Sleep']),
            'Physical Activity Level': float(raw['Physical Activity Level']),
            'Stress Level':            float(raw['Stress Level']),
            'BMI Category':            cat_map['BMI Category'].get(raw['BMI Category'], 0),
            'Heart Rate':              float(raw['Heart Rate']),
            'Daily Steps':             float(raw['Daily Steps']),
            'Systolic':                float(raw['Systolic']),
            'Diastolic':               float(raw['Diastolic']),
        }
        feature_order = ['Gender', 'Age', 'Occupation', 'Sleep Duration',
                         'Quality of Sleep', 'Physical Activity Level',
                         'Stress Level', 'BMI Category', 'Heart Rate',
                         'Daily Steps', 'Systolic', 'Diastolic']
        return np.array([[row[c] for c in feature_order]])

    # ── public API ─────────────────────────────────────────────────────────────

    def predict(self, disease, raw):
        # Map route key → internal key
        key_map = {'sleep': 'sleep_apnea'}
        internal = key_map.get(disease, disease)

        if internal not in self._models:
            raise ValueError(f'Model for {disease} not loaded. Run train.py first.')

        model_pkg = self._models[internal]
        scaler    = self._scalers[internal]
        model     = model_pkg['model']

        prep = {
            'diabetes':   self._prep_diabetes,
            'heart':      self._prep_heart,
            'obesity':    self._prep_obesity,
            'sleep_apnea':self._prep_sleep,
        }
        X = prep[internal](raw)
        X_scaled = scaler.transform(X)

        prob_array   = model.predict_proba(X_scaled)[0]
        is_multiclass = len(model_pkg['classes']) > 2

        return self._build_result(model_pkg, prob_array, is_multiclass)
