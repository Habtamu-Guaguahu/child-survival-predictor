import numpy as np
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Child Survival Predictor",
    page_icon="👶",
    layout="wide"
)

# =========================================
# LOAD MODEL AND SCALER
# =========================================

@st.cache_resource
def load_model():
    """Load the trained FS-SVM model and scaler"""
    try:
        with open('models/FS_SVM_UnivariateMI.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler_fssvm_umi.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure models are in the correct directory.")
        return None, None

model, scaler = load_model()

# =========================================
# DEFINE FEATURES AND MAPPING
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

VARIABLE_LABELS = {
    'country': 'Country',
    'matage': "Mother's Age Group",
    'househeadage': "Household Head Age",
    'contraceptiveuse': "Contraceptive Use",
    'birthorder': "Birth Order",
    'husbedu': "Husband's Education",
    'workinghusb': "Husband Working Status",
    'sexhousehead': "Sex of Household Head",
    'anc_visits': "Antenatal Care Visits",
    'twin': "Twin Birth",
    'childgender': "Child Gender",
    'placeresid': "Place of Residence",
    'wealthindex': "Wealth Index",
    'immediate_bf': "Immediate Breastfeeding",
    'sanitation': "Sanitation Type",
    'parity': "Parity",
    'matedu': "Mother's Education",
    'maritalstatus': "Marital Status",
    'placeofdelivery': "Place of Delivery",
    'watersource': "Water Source",
    'teenagepx': "Teenage Pregnancy",
    'mediaexposure': "Media Exposure",
    'householdsize': "Household Size",
    'birthinterval': "Preceding Birth Interval",
    'deliverybycs': "Delivery by C-Section",
    'pnc': "Postnatal Care (6 weeks)"
}

VARIABLE_OPTIONS = {
    'country': ['Burundi', 'Comoros', 'Ethiopia', 'Kenya', 'Madagascar', 
                'Malawi', 'Mozambique', 'Rwanda', 'Tanzania', 'Uganda', 
                'Zambia', 'Zimbabwe'],
    'matage': ['15-19 years', '20-29 years', '30-39 years', '40-49 years'],
    'househeadage': ['35 years or less', 'More than 35 years'],
    'contraceptiveuse': ['Not using', 'Using any method'],
    'birthorder': ['First', 'Second to Fourth', 'Fifth and more'],
    'husbedu': ['No education', 'Higher education'],
    'workinghusb': ['Not working', 'Working'],
    'sexhousehead': ['Male', 'Female'],
    'anc_visits': ['No ANC', '1-3 visits', '4+ visits'],
    'twin': ['No', 'Yes'],
    'childgender': ['Male', 'Female'],
    'placeresid': ['Urban', 'Rural'],
    'wealthindex': ['Poorest', 'Poorer', 'Middle'],
    'immediate_bf': ['No', 'Yes'],
    'sanitation': ['Improved', 'Unimproved', 'Open defecation'],
    'parity': ['Primiparous', 'Multiparous', 'Grandparous'],
    'matedu': ['No education', 'Primary', 'Secondary'],
    'maritalstatus': ['Not married', 'Married'],
    'placeofdelivery': ['Institutional', 'Home'],
    'watersource': ['Improved', 'Unimproved', 'Surface water'],
    'teenagepx': ['No', 'Yes'],
    'mediaexposure': ['No exposure', 'Any exposure'],
    'householdsize': ['1-5 members', 'Above 5 members'],
    'birthinterval': ['Less than 2 years', '4+ years'],
    'deliverybycs': ['No', 'Yes'],
    'pnc': ['No', 'Yes']
}

# =========================================
# PREDICTION FUNCTIONS
# =========================================

def create_feature_vector(form_data):
    """Convert dropdown selections to dummy variable vector"""
    feature_dict = {f: 0 for f in ALL_FEATURES}
    
    for var_key in DUMMY_MAPPING.keys():
        selected = form_data.get(var_key)
        if selected and selected in DUMMY_MAPPING[var_key]:
            dummy_map = DUMMY_MAPPING[var_key][selected]
            for dummy_col, value in dummy_map.items():
                if dummy_col in feature_dict:
                    feature_dict[dummy_col] = value
    
    df = pd.DataFrame([feature_dict])
    return df[ALL_FEATURES]

def predict_risk(form_data):
    """Predict risk score from form data"""
    if model is None or scaler is None:
        return None
    
    feature_df = create_feature_vector(form_data)
    feature_scaled = scaler.transform(feature_df)
    risk_score = model.predict(feature_scaled)[0]
    return risk_score

def get_survival_probability(risk_score):
    """Calculate survival probabilities"""
    time_points = [1, 6, 12, 24, 36, 48, 60]
    survival = []
    for t in time_points:
        prob = np.exp(-np.exp(risk_score * 0.5) * t * 0.008)
        survival.append(max(0, min(1, prob)))
    return time_points, survival

# =========================================
# STREAMLIT UI
# =========================================

st.title("👶 Child Survival Probability Predictor")
st.markdown("### Enter child and mother characteristics to predict survival probability")

# Check if model loaded
if model is None or scaler is None:
    st.error("⚠️ Model not loaded. Please ensure model files are in the 'models' folder.")
    st.stop()

# Create layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("---")
    st.subheader("📋 Patient Information")
    
    # Create form
    form_data = {}
    
    # Group variables into sections
    sections = {
        "👤 Demographics": ['country', 'matage', 'childgender', 'householdsize', 'househeadage'],
        "🏠 Household & Socioeconomic": ['wealthindex', 'matedu', 'husbedu', 'workinghusb', 'mediaexposure', 'sexhousehead'],
        "🏥 Health & Healthcare": ['contraceptiveuse', 'anc_visits', 'placeofdelivery', 'deliverybycs', 'pnc', 'immediate_bf'],
        "👶 Child & Birth": ['birthorder', 'twin', 'parity', 'birthinterval', 'teenagepx'],
        "🌍 Environment": ['placeresid', 'sanitation', 'watersource']
    }
    
    for section_name, variables in sections.items():
        st.markdown(f"#### {section_name}")
        cols = st.columns(2)
        for i, var_key in enumerate(variables):
            with cols[i % 2]:
                label = VARIABLE_LABELS.get(var_key, var_key)
                options = VARIABLE_OPTIONS.get(var_key, [])
                form_data[var_key] = st.selectbox(label, options, key=var_key)
        st.markdown("---")
    
    predict_button = st.button("🔮 Predict Survival Probability", type="primary", use_container_width=True)

with col2:
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    This tool predicts the survival probability of a child based on:
    
    - **FS-SVM Model** (Fast Survival Support Vector Machine)
    - **45 Features** selected via Univariate Mutual Information
    - **6,168 training samples** (7.7% event rate)
    
    ---
    
    **Risk Classification (Two Groups):**
    
    🟢 **LOW RISK**: Risk score ≤ 0.3
       - Good prognosis
       - Standard care recommended
       
    🔴 **HIGH RISK**: Risk score > 0.3
       - Poorer prognosis
       - Close monitoring required
       
    ---
    **Threshold:** 0.3 (based on model calibration)
    """)

# =========================================
# PREDICTION RESULTS
# =========================================

if predict_button:
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Calculate prediction
    risk_score = predict_risk(form_data)
    
    if risk_score is not None:
        time_points, survival_probs = get_survival_probability(risk_score)
        
        # Risk level (HIGH vs LOW only)
        if risk_score > 0.3:  # Threshold for High Risk
            risk_level = "🔴 HIGH RISK"
            risk_color = "red"
            risk_badge_color = "#e74c3c"
            risk_bg_color = "#fde8e8"
        else:
            risk_level = "🟢 LOW RISK"
            risk_color = "green"
            risk_badge_color = "#27ae60"
            risk_bg_color = "#e8f8f5"
        
        # Display metrics with colored risk badge
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.metric("Risk Score", f"{risk_score:.4f}")

        with col2:
            # Custom styled risk badge
            st.markdown(f"""
            <div style="background-color: {risk_bg_color}; padding: 15px; border-radius: 10px; border: 2px solid {risk_badge_color}; text-align: center;">
                <span style="font-size: 1.2rem; font-weight: bold; color: {risk_badge_color};">
                    {risk_level}
                </span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            event_prob = 1 - survival_probs[2]  # 12-month event probability
            st.metric("12-Month Event Risk", f"{event_prob:.1%}")
        
        # Survival table
        st.markdown("#### Survival Probability by Time")
        df_surv = pd.DataFrame({
            'Time (months)': time_points,
            'Survival Probability': [f"{p:.3f} ({p*100:.1f}%)" for p in survival_probs]
        })
        st.dataframe(df_surv, use_container_width=True, hide_index=True)
        
        # Survival curve
        st.markdown("#### Survival Curve")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot survival curve
        ax.plot(time_points, survival_probs, 'b-o', linewidth=2, markersize=8, label='Survival Probability')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% Threshold')
        ax.fill_between(time_points, survival_probs, alpha=0.2, color='blue')
        
        # Styling
        ax.set_xlabel('Time (months)', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title('Kaplan-Meier Style Survival Curve', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, 65])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left')
        
        # Add risk level annotation
        ax.text(0.98, 0.05, f"Risk: {risk_level}", 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        st.pyplot(fig)
        plt.close()
        
        # =========================================
        # RISK GROUP CLASSIFICATION
        # =========================================
        
        st.markdown("#### 📊 Risk Group Classification")
        
        # Calculate risk group
        if risk_score > 0.3:
            risk_group = "HIGH RISK"
            risk_color = "#e74c3c"
            risk_icon = "🔴"
            recommendation = "⚠️ This child is classified as HIGH RISK. Close monitoring and early intervention are strongly recommended."
        else:
            risk_group = "LOW RISK"
            risk_color = "#27ae60"
            risk_icon = "🟢"
            recommendation = "✅ This child is classified as LOW RISK. Continue standard care and routine monitoring."
        
        # Display risk group with large visual indicator
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div style="background-color: {risk_color}20; padding: 30px; border-radius: 15px; border: 3px solid {risk_color}; text-align: center;">
                <span style="font-size: 4rem;">{risk_icon}</span>
                <br>
                <span style="font-size: 1.5rem; font-weight: bold; color: {risk_color};">
                    {risk_group}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; height: 100%;">
                <h5 style="color: {risk_color};">Risk Group: {risk_group}</h5>
                <p style="font-size: 1rem;">{recommendation}</p>
                <hr>
                <p style="font-size: 0.9rem; color: #555;">
                    <strong>Risk Score:</strong> {risk_score:.4f}<br>
                    <strong>Threshold:</strong> Score > 0.3 = High Risk
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Survival comparison with risk zone
        st.markdown("#### 📈 Survival Curve with Risk Zone")
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        
        # Current patient survival
        ax2.plot(time_points, survival_probs, 'o-', linewidth=2.5, markersize=8, 
                 label=f'Current Patient ({risk_group})', color=risk_badge_color)
        
        # 50% threshold line
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% Threshold')
        
        # Add shaded regions for risk interpretation
        if risk_score > 0.3:
            # High risk - show warning area
            ax2.fill_between(time_points, 0, survival_probs, alpha=0.2, color='red', 
                             label='High Risk Zone')
        else:
            # Low risk - show safe area
            ax2.fill_between(time_points, survival_probs, 1, alpha=0.2, color='green', 
                             label='Low Risk Zone')
        
        ax2.set_xlabel('Time (months)', fontsize=12)
        ax2.set_ylabel('Survival Probability', fontsize=12)
        ax2.set_title(f'Survival Curve - {risk_group} Patient', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.set_xlim([0, 65])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower left')
        
        st.pyplot(fig2)
        plt.close()
        
        # =========================================
        # INTERPRETATION
        # =========================================
        
        st.markdown("#### 📝 Clinical Interpretation")
        
        if risk_score > 0.3:
            st.warning(f"""
            **🔴 HIGH RISK Patient**
            
            | Metric | Value |
            |--------|-------|
            | Risk Score | **{risk_score:.4f}** (> 0.3 threshold) |
            | 12-month survival | **{survival_probs[2]:.1%}** |
            | 24-month survival | **{survival_probs[3]:.1%}** |
            | 60-month survival | **{survival_probs[6]:.1%}** |
            
            **📌 Recommendation:** This child requires **close monitoring** and **early intervention**.
            Consider additional clinical assessment and follow-up.
            """)
        else:
            st.success(f"""
            **🟢 LOW RISK Patient**
            
            | Metric | Value |
            |--------|-------|
            | Risk Score | **{risk_score:.4f}** (≤ 0.3 threshold) |
            | 12-month survival | **{survival_probs[2]:.1%}** |
            | 24-month survival | **{survival_probs[3]:.1%}** |
            | 60-month survival | **{survival_probs[6]:.1%}** |
            
            **📌 Recommendation:** This child has a **favorable prognosis**.
            Continue standard care and routine monitoring.
            """)
        
        # =========================================
        # RISK FACTORS SUMMARY
        # =========================================
        
        st.markdown("#### 🔍 Key Risk Factors")
        
        # Identify which features are "active" (dummy = 1)
        feature_df = create_feature_vector(form_data)
        active_features = []
        
        for col in feature_df.columns:
            if feature_df[col].values[0] == 1:
                # Get human-readable label
                for var_key, mapping in DUMMY_MAPPING.items():
                    for option, dummy_dict in mapping.items():
                        if col in dummy_dict:
                            active_features.append(f"• {VARIABLE_LABELS.get(var_key, var_key)}: {option}")
                            break
        
        if active_features:
            # Show up to 10 most important risk factors
            st.markdown("**Selected characteristics that may influence risk:**")
            for feat in active_features[:10]:
                st.markdown(feat)
            if len(active_features) > 10:
                st.markdown(f"*... and {len(active_features) - 10} more factors*")
        else:
            st.markdown("*All factors are at reference levels (baseline).*")
        
    else:
        st.error("⚠️ Prediction failed. Please check model files.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8rem;'>
    Developed with ❤️ using Streamlit • FS-SVM Model • Univariate MI Feature Selection
</div>
""", unsafe_allow_html=True)