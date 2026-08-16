import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request
import warnings
warnings.filterwarnings('ignore')

# Create Flask app
app = Flask(__name__)

# =========================================
# LOAD MODEL AND SCALER
# =========================================

def load_model():
    """Load the trained FS-SVM model and scaler"""
    try:
        with open('models/FS_SVM_UnivariateMI.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler_fssvm_umi.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Model and scaler loaded successfully!")
        return model, scaler
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None, None

# Load the models
model, scaler = load_model()

# =========================================
# DEFINE ALL 45 FEATURES (in correct order)
# =========================================

ALL_FEATURES = [
    'matage_30_39_vs_15_19', 'househeadage_more_than_35_vs_35_and_less',
    'matage_20_29_vs_15_19', 'contraceptiveuse_using_any_method_vs_not_using',
    'birthorder_fifth_and_more_vs_first', 'country_Madagascar_vs_Burundi',
    'country_Zimbabwe_vs_Burundi', 'husbedu_higher_vs_no_education',
    'workinghusb_working_vs_not_working', 'sexofthehousehead_female_vs_male',
    'anc_4_plus_times_vs_no_anc', 'twin_yes_vs_no', 'country_Zambia_vs_Burundi',
    'childgender_female_vs_male', 'birthorder_second_upto_fourth_vs_first',
    'placeresid_rural_vs_urban', 'wealthindex_middle_vs_poorest',
    'immediate_bf_Yes_vs_No', 'sanitation_open_defecation_vs_improved',
    'parity_multiparous_vs_primiparous', 'matedu_primary_vs_no_education',
    'matedu_secondary_vs_no_education', 'maritalstatus_married_vs_not_married',
    'country_Mozambique_vs_Burundi', 'placeofdelivery_home_vs_institutional',
    'watersource_unimproved_vs_improved', 'watersource_surface_water_vs_improved',
    'country_Comoros_vs_Burundi', 'country_Ethiopia_vs_Burundi',
    'country_Uganda_vs_Burundi', 'country_Kenya_vs_Burundi',
    'teenagepx_yes_vs_no', 'sanitation_unimproved_vs_improved',
    'country_Rwanda_vs_Burundi', 'country_Tanzania_vs_Burundi',
    'country_Malawi_vs_Burundi', 'mediaexposure_any_exposure_vs_no_exposure',
    'householdsize_above_5_members_vs_1_5_members', 'anc_1_3_times_vs_no_anc',
    'preceedbirthinterval_4_and_more_years_vs_less_than_2_years',
    'wealthindex_poorer_vs_poorest', 'parity_grandparous_vs_primiparous',
    'matage_40_49_vs_15_19', 'deliverybycs_yes_vs_no', 'pnc_6weeks_yes_vs_no'
]

# =========================================
# DUMMY VARIABLE MAPPING (Internal Use)
# =========================================

DUMMY_MAPPING = {
    'country': {
        'Burundi': {},
        'Comoros': {'country_Comoros_vs_Burundi': 1},
        'Ethiopia': {'country_Ethiopia_vs_Burundi': 1},
        'Kenya': {'country_Kenya_vs_Burundi': 1},
        'Madagascar': {'country_Madagascar_vs_Burundi': 1},
        'Malawi': {'country_Malawi_vs_Burundi': 1},
        'Mozambique': {'country_Mozambique_vs_Burundi': 1},
        'Rwanda': {'country_Rwanda_vs_Burundi': 1},
        'Tanzania': {'country_Tanzania_vs_Burundi': 1},
        'Uganda': {'country_Uganda_vs_Burundi': 1},
        'Zambia': {'country_Zambia_vs_Burundi': 1},
        'Zimbabwe': {'country_Zimbabwe_vs_Burundi': 1}
    },
    'matage': {
        '15-19 years': {},
        '20-29 years': {'matage_20_29_vs_15_19': 1},
        '30-39 years': {'matage_30_39_vs_15_19': 1},
        '40-49 years': {'matage_40_49_vs_15_19': 1}
    },
    'househeadage': {
        '35 years or less': {},
        'More than 35 years': {'househeadage_more_than_35_vs_35_and_less': 1}
    },
    'contraceptiveuse': {
        'Not using': {},
        'Using any method': {'contraceptiveuse_using_any_method_vs_not_using': 1}
    },
    'birthorder': {
        'First': {},
        'Second to Fourth': {'birthorder_second_upto_fourth_vs_first': 1},
        'Fifth and more': {'birthorder_fifth_and_more_vs_first': 1}
    },
    'husbedu': {
        'No education': {},
        'Higher education': {'husbedu_higher_vs_no_education': 1}
    },
    'workinghusb': {
        'Not working': {},
        'Working': {'workinghusb_working_vs_not_working': 1}
    },
    'sexhousehead': {
        'Male': {},
        'Female': {'sexofthehousehead_female_vs_male': 1}
    },
    'anc_visits': {
        'No ANC': {},
        '1-3 visits': {'anc_1_3_times_vs_no_anc': 1},
        '4+ visits': {'anc_4_plus_times_vs_no_anc': 1}
    },
    'twin': {
        'No': {},
        'Yes': {'twin_yes_vs_no': 1}
    },
    'childgender': {
        'Male': {},
        'Female': {'childgender_female_vs_male': 1}
    },
    'placeresid': {
        'Urban': {},
        'Rural': {'placeresid_rural_vs_urban': 1}
    },
    'wealthindex': {
        'Poorest': {},
        'Poorer': {'wealthindex_poorer_vs_poorest': 1},
        'Middle': {'wealthindex_middle_vs_poorest': 1}
    },
    'immediate_bf': {
        'No': {},
        'Yes': {'immediate_bf_Yes_vs_No': 1}
    },
    'sanitation': {
        'Improved': {},
        'Unimproved': {'sanitation_unimproved_vs_improved': 1},
        'Open defecation': {'sanitation_open_defecation_vs_improved': 1}
    },
    'parity': {
        'Primiparous': {},
        'Multiparous': {'parity_multiparous_vs_primiparous': 1},
        'Grandparous': {'parity_grandparous_vs_primiparous': 1}
    },
    'matedu': {
        'No education': {},
        'Primary': {'matedu_primary_vs_no_education': 1},
        'Secondary': {'matedu_secondary_vs_no_education': 1}
    },
    'maritalstatus': {
        'Not married': {},
        'Married': {'maritalstatus_married_vs_not_married': 1}
    },
    'placeofdelivery': {
        'Institutional': {},
        'Home': {'placeofdelivery_home_vs_institutional': 1}
    },
    'watersource': {
        'Improved': {},
        'Unimproved': {'watersource_unimproved_vs_improved': 1},
        'Surface water': {'watersource_surface_water_vs_improved': 1}
    },
    'teenagepx': {
        'No': {},
        'Yes': {'teenagepx_yes_vs_no': 1}
    },
    'mediaexposure': {
        'No exposure': {},
        'Any exposure': {'mediaexposure_any_exposure_vs_no_exposure': 1}
    },
    'householdsize': {
        '1-5 members': {},
        'Above 5 members': {'householdsize_above_5_members_vs_1_5_members': 1}
    },
    'birthinterval': {
        'Less than 2 years': {},
        '4+ years': {'preceedbirthinterval_4_and_more_years_vs_less_than_2_years': 1}
    },
    'deliverybycs': {
        'No': {},
        'Yes': {'deliverybycs_yes_vs_no': 1}
    },
    'pnc': {
        'No': {},
        'Yes': {'pnc_6weeks_yes_vs_no': 1}
    }
}

# =========================================
# DEFINE VARIABLES FOR DROPDOWN MENUS
# =========================================

VARIABLES = {
    'country': {
        'label': 'Country',
        'options': ['Burundi', 'Comoros', 'Ethiopia', 'Kenya', 'Madagascar', 
                   'Malawi', 'Mozambique', 'Rwanda', 'Tanzania', 'Uganda', 
                   'Zambia', 'Zimbabwe']
    },
    'matage': {
        'label': "Mother's Age Group",
        'options': ['15-19 years', '20-29 years', '30-39 years', '40-49 years']
    },
    'househeadage': {
        'label': "Household Head Age",
        'options': ['35 years or less', 'More than 35 years']
    },
    'contraceptiveuse': {
        'label': "Contraceptive Use",
        'options': ['Not using', 'Using any method']
    },
    'birthorder': {
        'label': "Birth Order",
        'options': ['First', 'Second to Fourth', 'Fifth and more']
    },
    'husbedu': {
        'label': "Husband's Education",
        'options': ['No education', 'Higher education']
    },
    'workinghusb': {
        'label': "Husband Working Status",
        'options': ['Not working', 'Working']
    },
    'sexhousehead': {
        'label': "Sex of Household Head",
        'options': ['Male', 'Female']
    },
    'anc_visits': {
        'label': "Antenatal Care Visits",
        'options': ['No ANC', '1-3 visits', '4+ visits']
    },
    'twin': {
        'label': "Twin Birth",
        'options': ['No', 'Yes']
    },
    'childgender': {
        'label': "Child Gender",
        'options': ['Male', 'Female']
    },
    'placeresid': {
        'label': "Place of Residence",
        'options': ['Urban', 'Rural']
    },
    'wealthindex': {
        'label': "Wealth Index",
        'options': ['Poorest', 'Poorer', 'Middle']
    },
    'immediate_bf': {
        'label': "Immediate Breastfeeding",
        'options': ['No', 'Yes']
    },
    'sanitation': {
        'label': "Sanitation Type",
        'options': ['Improved', 'Unimproved', 'Open defecation']
    },
    'parity': {
        'label': "Parity",
        'options': ['Primiparous', 'Multiparous', 'Grandparous']
    },
    'matedu': {
        'label': "Mother's Education",
        'options': ['No education', 'Primary', 'Secondary']
    },
    'maritalstatus': {
        'label': "Marital Status",
        'options': ['Not married', 'Married']
    },
    'placeofdelivery': {
        'label': "Place of Delivery",
        'options': ['Institutional', 'Home']
    },
    'watersource': {
        'label': "Water Source",
        'options': ['Improved', 'Unimproved', 'Surface water']
    },
    'teenagepx': {
        'label': "Teenage Pregnancy",
        'options': ['No', 'Yes']
    },
    'mediaexposure': {
        'label': "Media Exposure",
        'options': ['No exposure', 'Any exposure']
    },
    'householdsize': {
        'label': "Household Size",
        'options': ['1-5 members', 'Above 5 members']
    },
    'birthinterval': {
        'label': "Preceding Birth Interval",
        'options': ['Less than 2 years', '4+ years']
    },
    'deliverybycs': {
        'label': "Delivery by C-Section",
        'options': ['No', 'Yes']
    },
    'pnc': {
        'label': "Postnatal Care (6 weeks)",
        'options': ['No', 'Yes']
    }
}

# =========================================
# PREDICTION FUNCTIONS
# =========================================

def create_feature_vector(form_data):
    """Convert dropdown selections to dummy variable vector"""
    
    # Initialize all features to 0
    feature_dict = {f: 0 for f in ALL_FEATURES}
    
    # Map user selections to dummy variables
    for var_key in DUMMY_MAPPING.keys():
        selected = form_data.get(var_key)
        if selected and selected in DUMMY_MAPPING[var_key]:
            dummy_map = DUMMY_MAPPING[var_key][selected]
            for dummy_col, value in dummy_map.items():
                if dummy_col in feature_dict:
                    feature_dict[dummy_col] = value
    
    # Create DataFrame with proper feature order
    df = pd.DataFrame([feature_dict])
    return df[ALL_FEATURES]

def predict_risk(form_data):
    """Predict risk score from form data"""
    if model is None or scaler is None:
        return None
    
    # Create feature vector
    feature_df = create_feature_vector(form_data)
    
    # Scale features
    feature_scaled = scaler.transform(feature_df)
    
    # Get risk score
    risk_score = model.predict(feature_scaled)[0]
    
    return risk_score

def get_survival_probability(risk_score, time_points=[1, 6, 12, 24, 36, 48, 60]):
    """Calculate survival probability at specified time points"""
    survival = []
    for t in time_points:
        # Using the risk score to estimate survival
        # Higher risk = lower survival
        prob = np.exp(-np.exp(risk_score * 0.5) * t * 0.008)
        survival.append(max(0, min(1, prob)))
    return survival

# =========================================
# FLASK ROUTES
# =========================================

@app.route('/')
def index():
    """Home page with form"""
    return render_template('index.html', variables=VARIABLES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and return predictions"""
    
    # Get form data
    form_data = request.form.to_dict()
    
    # Calculate risk score
    risk_score = predict_risk(form_data)
    
    if risk_score is None:
        return "Error: Model not loaded", 500
    
    # Calculate survival probabilities
    time_points = [1, 6, 12, 24, 36, 48, 60]
    survival_probs = get_survival_probability(risk_score, time_points)
    
    # Prepare results
    results = {
        'risk_score': float(risk_score),
        'time_points': time_points,
        'survival': survival_probs
    }
    
    return render_template('result.html', 
                          results=results,
                          selected_values=form_data)

# =========================================
# RUN THE APP
# =========================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Child Survival Predictor Web App")
    print("="*50)
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press CTRL+C to stop the server")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)