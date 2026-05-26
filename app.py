import json, os
from flask import Flask, render_template, request, jsonify
from predictor import Predictor

app = Flask(__name__)
predictor = Predictor()

# Load training results once at startup
_RESULTS_PATH = os.path.join('results', 'all_results.json')
try:
    with open(_RESULTS_PATH) as f:
        ALL_RESULTS = json.load(f)
except FileNotFoundError:
    ALL_RESULTS = {}

# ── Per-disease form field config ──────────────────────────────────────────────
DISEASE_CONFIG = {
    'diabetes': {
        'title':   'Diabetes',
        'subtitle':'Enter your latest blood test and health report values',
        'color':   'cyan',
        'icon':    'drop',
        'fields': [
            {'name':'Pregnancies',              'label':'Pregnancies',                 'type':'number','min':0,  'max':20,  'step':1,    'placeholder':'0–17'},
            {'name':'Glucose',                  'label':'Glucose (mg/dL)',             'type':'number','min':0,  'max':300, 'step':1,    'placeholder':'70–200'},
            {'name':'BloodPressure',            'label':'Blood Pressure (mm Hg)',      'type':'number','min':0,  'max':200, 'step':1,    'placeholder':'60–140'},
            {'name':'SkinThickness',            'label':'Skin Thickness (mm)',         'type':'number','min':0,  'max':100, 'step':1,    'placeholder':'10–60'},
            {'name':'Insulin',                  'label':'Insulin (µU/mL)',             'type':'number','min':0,  'max':900, 'step':1,    'placeholder':'0–500'},
            {'name':'BMI',                      'label':'BMI',                         'type':'number','min':0,  'max':70,  'step':0.1,  'placeholder':'18.5–45'},
            {'name':'Age',                      'label':'Age (years)',                 'type':'number','min':1,  'max':120, 'step':1,    'placeholder':'21–81'},
        ],
        'classes': ['Non-Diabetic', 'Diabetic'],
    },
    'heart': {
        'title':   'Heart Disease',
        'subtitle':'Enter your cardiac test and clinical report values',
        'color':   'pink',
        'icon':    'heart',
        'fields': [
            {'name':'age',     'label':'Age (years)',                     'type':'number','min':1,  'max':120,'step':1,  'placeholder':'29–77'},
            {'name':'sex',     'label':'Sex',                             'type':'select','options':[('1','Male'),('0','Female')]},
            {'name':'cp',      'label':'Chest Pain Type',                 'type':'select','options':[('0','Typical Angina'),('1','Atypical Angina'),('2','Non-Anginal'),('3','Asymptomatic')]},
            {'name':'trestbps','label':'Resting Blood Pressure',          'type':'number','min':80, 'max':220,'step':1,  'placeholder':'94–200 mm Hg'},
            {'name':'chol',    'label':'Serum Cholesterol',               'type':'number','min':100,'max':600,'step':1,  'placeholder':'126–564 mg/dL'},
            {'name':'fbs',     'label':'Fasting Blood Sugar > 120 mg/dL','type':'select','options':[('0','No'),('1','Yes')]},
            {'name':'restecg', 'label':'Resting ECG Results',             'type':'select','options':[('0','Normal'),('1','ST-T Abnormality'),('2','Left Ventricular Hypertrophy')]},
            {'name':'thalach', 'label':'Maximum Heart Rate',              'type':'number','min':60, 'max':220,'step':1,  'placeholder':'71–202 bpm'},
            {'name':'exang',   'label':'Exercise-Induced Angina',         'type':'select','options':[('0','No'),('1','Yes')]},
            {'name':'oldpeak', 'label':'ST Depression (Oldpeak)',         'type':'number','min':0,  'max':7,  'step':0.1,'placeholder':'0–6.2'},
            {'name':'slope',   'label':'Slope of Peak ST Segment',        'type':'select','options':[('0','Upsloping'),('1','Flat'),('2','Downsloping')]},
            {'name':'ca',      'label':'Major Vessels (Fluoroscopy)',      'type':'select','options':[('0','0'),('1','1'),('2','2'),('3','3')]},
            {'name':'thal',    'label':'Thalassemia',                     'type':'select','options':[('1','Normal'),('2','Fixed Defect'),('3','Reversible Defect')]},
        ],
        'classes': ['No Heart Disease', 'Heart Disease'],
    },
    'obesity': {
        'title':   'Obesity',
        'subtitle':'Enter your lifestyle and dietary habits',
        'color':   'green',
        'icon':    'activity',
        'fields': [
            {'name':'Age',   'label':'Age (years)','type':'number','min':1, 'max':100,'step':1,  'placeholder':'14–61'},
            {'name':'Gender','label':'Gender',      'type':'select','options':[('Female','Female'),('Male','Male')]},
            {'name':'CALC',  'label':'Alcohol Consumption','type':'select','options':[('no','Never'),('Sometimes','Sometimes'),('Frequently','Frequently'),('Always','Always')]},
            {'name':'FAVC',  'label':'High-Calorie Food Frequently','type':'select','options':[('no','No'),('yes','Yes')]},
            {'name':'FCVC',  'label':'Vegetables in Meals (1–3)','type':'number','min':1,'max':3,'step':0.1,'placeholder':'1=rarely, 3=always'},
            {'name':'NCP',   'label':'Main Meals per Day',       'type':'number','min':1,'max':4,'step':0.1,'placeholder':'1–4'},
            {'name':'SCC',   'label':'Monitor Calorie Intake',   'type':'select','options':[('no','No'),('yes','Yes')]},
            {'name':'SMOKE', 'label':'Smoker',                   'type':'select','options':[('no','No'),('yes','Yes')]},
            {'name':'CH2O',  'label':'Daily Water Intake (1–3)', 'type':'number','min':1,'max':3,'step':0.1,'placeholder':'1=<1L, 3=>2L'},
            {'name':'family_history_with_overweight','label':'Family History of Overweight','type':'select','options':[('no','No'),('yes','Yes')]},
            {'name':'FAF',   'label':'Physical Activity Freq (0–3)','type':'number','min':0,'max':3,'step':0.1,'placeholder':'0=none, 3=daily'},
            {'name':'TUE',   'label':'Tech Device Usage (0–2)',  'type':'number','min':0,'max':2,'step':0.1,'placeholder':'0–2 hrs/day'},
            {'name':'CAEC',  'label':'Eating Between Meals',     'type':'select','options':[('no','Never'),('Sometimes','Sometimes'),('Frequently','Frequently'),('Always','Always')]},
            {'name':'MTRANS','label':'Transportation Mode',      'type':'select','options':[('Automobile','Automobile'),('Bike','Bike'),('Motorbike','Motorbike'),('Public_Transportation','Public Transport'),('Walking','Walking')]},
        ],
        'classes': ['Underweight','Normal','Overweight I','Overweight II','Obese I','Obese II','Obese III'],
    },
    'sleep': {
        'title':   'Sleep Apnea',
        'subtitle':'Enter your sleep and lifestyle information',
        'color':   'purple',
        'icon':    'moon',
        'fields': [
            {'name':'Gender',                 'label':'Gender',                     'type':'select','options':[('Female','Female'),('Male','Male')]},
            {'name':'Age',                    'label':'Age (years)',                'type':'number','min':18,'max':100,'step':1,  'placeholder':'27–59'},
            {'name':'Occupation',             'label':'Occupation',                 'type':'select','options':[
                ('Accountant','Accountant'),('Doctor','Doctor'),('Engineer','Engineer'),
                ('Lawyer','Lawyer'),('Manager','Manager'),('Nurse','Nurse'),
                ('Sales Representative','Sales Representative'),('Salesperson','Salesperson'),
                ('Scientist','Scientist'),('Software Engineer','Software Engineer'),('Teacher','Teacher'),
            ]},
            {'name':'Sleep Duration',         'label':'Sleep Duration (hrs/night)','type':'number','min':4, 'max':10, 'step':0.1,'placeholder':'5.8–8.5'},
            {'name':'Quality of Sleep',       'label':'Sleep Quality (1–10)',      'type':'number','min':1, 'max':10, 'step':1,  'placeholder':'1=poor, 10=excellent'},
            {'name':'Physical Activity Level','label':'Physical Activity (min/day)','type':'number','min':0,'max':120,'step':1,  'placeholder':'30–90 mins'},
            {'name':'Stress Level',           'label':'Stress Level (1–10)',       'type':'number','min':1, 'max':10, 'step':1,  'placeholder':'1=low, 10=high'},
            {'name':'BMI Category',           'label':'BMI Category',              'type':'select','options':[('Normal','Normal'),('Overweight','Overweight'),('Obese','Obese')]},
            {'name':'Heart Rate',             'label':'Resting Heart Rate (bpm)',  'type':'number','min':40,'max':120,'step':1,  'placeholder':'65–86'},
            {'name':'Daily Steps',            'label':'Daily Steps',               'type':'number','min':0,'max':20000,'step':100,'placeholder':'3000–10000'},
            {'name':'Systolic',               'label':'Systolic BP (mm Hg)',       'type':'number','min':80,'max':200,'step':1,  'placeholder':'115–142'},
            {'name':'Diastolic',              'label':'Diastolic BP (mm Hg)',      'type':'number','min':50,'max':130,'step':1,  'placeholder':'75–95'},
        ],
        'classes': ['No Disorder','Sleep Apnea','Insomnia'],
    },
}

HOME_CARDS = [
    {'key':'diabetes','title':'Diabetes',    'color':'cyan',  'icon':'drop',    'desc':'Blood glucose, insulin and metabolic risk assessment'},
    {'key':'heart',   'title':'Heart Disease','color':'pink', 'icon':'heart',   'desc':'Cardiac risk based on ECG and clinical blood markers'},
    {'key':'obesity', 'title':'Obesity',     'color':'green', 'icon':'activity','desc':'Weight classification from lifestyle and dietary habits'},
    {'key':'sleep',   'title':'Sleep Apnea', 'color':'purple','icon':'moon',    'desc':'Sleep disorder detection from sleep and stress patterns'},
]

# Internal key mapping (route → results key)
_ROUTE_TO_RESULTS = {'sleep': 'sleep_apnea'}


@app.route('/')
def index():
    return render_template('index.html', cards=HOME_CARDS)


@app.route('/<disease>')
def form(disease):
    if disease not in DISEASE_CONFIG:
        return render_template('index.html', cards=HOME_CARDS), 404
    results_key = _ROUTE_TO_RESULTS.get(disease, disease)
    stats = ALL_RESULTS.get(results_key)
    return render_template('form.html',
                           disease=disease,
                           cfg=DISEASE_CONFIG[disease],
                           stats=stats)


@app.route('/predict/<disease>', methods=['POST'])
def predict(disease):
    if disease not in DISEASE_CONFIG:
        return jsonify({'error': 'Unknown disease'}), 404
    try:
        raw = request.get_json()
        result = predictor.predict(disease, raw)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
