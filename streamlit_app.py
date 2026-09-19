from pathlib import Path
import math
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Finsta QuickLoans",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "finsta_rf_100_trees_compressed.joblib"
SCHEMA_PATH = BASE_DIR / "finsta_model_feature_schema.json"


@st.cache_resource(show_spinner="Loading Finsta credit-risk model...")
def load_model():
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]
    return artifact, pipeline, feature_columns


def money(value):
    return f"₹{float(value):,.0f}"


def calculate_policy(pd_value, salary, existing_emi, tenure, requested_amount):
    baseline_amount = max(0, (salary - existing_emi) * 12 * 0.50)

    if pd_value < 0.05:
        amount_factor = 1.00
        risk_category = "Very Low Risk"
    elif pd_value < 0.10:
        amount_factor = 0.90
        risk_category = "Low Risk"
    elif pd_value < 0.15:
        amount_factor = 0.75
        risk_category = "Moderate Risk"
    elif pd_value < 0.25:
        amount_factor = 0.50
        risk_category = "High Risk"
    elif pd_value < 0.35:
        amount_factor = 0.25
        risk_category = "Very High Risk"
    else:
        amount_factor = 0.00
        risk_category = "Decline / Manual Review"

    indicative_limit = baseline_amount * amount_factor

    priced_pd = min(pd_value, 0.35)
    base_rate = 12.0
    risk_premium = 21.0 * (priced_pd / 0.35)
    tenure_premium = {3: 0.0, 6: 0.5, 9: 1.0, 12: 1.5}.get(tenure, 1.5)

    indicative_rate = min(
        33.0,
        max(12.0, base_rate + risk_premium + tenure_premium),
    )

    if indicative_limit <= 0:
        decision = "Manual Review / Decline"
    elif requested_amount <= indicative_limit:
        decision = "Requested amount is within indicative eligibility"
    else:
        decision = "Requested amount exceeds indicative eligibility"

    return {
        "baseline_amount": round(baseline_amount, 2),
        "indicative_eligible_amount": round(indicative_limit, 2),
        "indicative_interest_rate": round(indicative_rate, 2),
        "risk_category": risk_category,
        "decision": decision,
    }


def build_model_input(data, feature_columns):
    model_input = pd.DataFrame(
        [
            {
                "Age": data["age"],
                "Gender": data["gender"],
                "Marital Status": data["marital_status"],
                "Education": data["education"],
                "City Tier": data["city_tier"],
                "Residence Type": data["residence_type"],
                "Employment Type": data["employment_type"],
                "Industry": data["industry"],
                "Years in Current Job / Business": float(data["years_in_current_job"]),
                "Last Drawn Monthly Salary": float(data["last_drawn_monthly_salary"]),
                "Net Worth": float(data["net_worth"]),
                "Credit Score": float(data["credit_score"]),
                "Existing Monthly EMI": float(data["existing_monthly_emi"]),
                "Existing Loans": int(data["existing_loans"]),
                "Credit Card Utilisation": float(data["credit_card_utilisation"]),
                "Prior Default / Write-off": data["prior_default"],
                "Application Channel": data["application_channel"],
                "Requested Loan Amount": float(data["requested_amount"]),
                "Requested Tenure": int(data["requested_tenure"]),
                "Number of Dependants": int(data["number_of_dependants"]),
            }
        ]
    )
    return model_input[feature_columns]


def calculate_amortization(principal, annual_rate, tenure_months):
    principal = float(principal)
    annual_rate = float(annual_rate)
    tenure_months = int(tenure_months)

    if principal <= 0 or tenure_months <= 0:
        return 0.0, 0.0, []

    monthly_rate = annual_rate / 100.0 / 12.0

    if monthly_rate == 0:
        emi = principal / tenure_months
    else:
        emi = (
            principal
            * monthly_rate
            * (1 + monthly_rate) ** tenure_months
            / ((1 + monthly_rate) ** tenure_months - 1)
        )

    balance = principal
    total_interest = 0.0
    rows = []

    for month in range(1, tenure_months + 1):
        opening = balance
        interest = opening * monthly_rate
        principal_part = emi - interest
        payment = emi

        if month == tenure_months:
            principal_part = opening
            payment = principal_part + interest

        balance = max(0.0, opening - principal_part)
        total_interest += interest

        rows.append(
            {
                "Month": month,
                "Opening Balance": round(opening, 2),
                "EMI": round(payment, 2),
                "Interest": round(interest, 2),
                "Principal": round(principal_part, 2),
                "Closing Balance": round(balance, 2),
            }
        )

    return emi, total_interest, rows


def init_state():
    defaults = {
        "latest_app": None,
        "latest_result": None,
        "active_application": None,
        "inquiry_recorded": False,
        "page": "Home",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_home():
    st.markdown(
        """
        <div class="hero">
            <div>
                <div class="kicker">SMART CREDIT • SIMPLE EXPERIENCE</div>
                <h1>Understand your loan eligibility <span>before you apply.</span></h1>
                <p>
                    Finsta QuickLoans combines affordability assessment with a machine-learning
                    Probability of Default (PD) model to generate an indicative loan offer.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indicative Rate", "12%–33% p.a.")
    c2.metric("Tenures", "3 / 6 / 9 / 12 months")
    c3.metric("ML Output", "Probability of Default")
    c4.metric("Model", "Random Forest • 100 trees")

    st.markdown("---")
    st.subheader("How Finsta QuickLoans works")
    steps = [
        ("01", "Enter your application details"),
        ("02", "Calculate affordability baseline"),
        ("03", "ML model estimates default risk"),
        ("04", "Generate indicative offer"),
    ]
    cols = st.columns(4)
    for col, (num, text) in zip(cols, steps):
        with col:
            st.markdown(f"### {num}")
            st.write(text)

    st.info(
        "This is a personal fintech project prototype. The eligibility, amount and interest rate "
        "are indicative and are not a final lending decision."
    )

    st.subheader("My Project Journey")
    st.write(
        "The project combines finance, credit-risk analysis and machine learning. "
        "A historical loan dataset is used to train the PD model, and the model output is "
        "combined with an affordability and pricing policy to create an indicative offer."
    )


def show_eligibility():
    st.header("Check Loan Eligibility")
    st.caption(
        "These are the application-time details used by the Finsta QuickLoans website-compatible credit model."
    )

    with st.form("eligibility_form"):
        st.subheader("Personal Information")
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", min_value=18, max_value=75, value=30, step=1)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with c3:
            marital_status = st.selectbox(
                "Marital Status", ["Single", "Married", "Divorced", "Widowed"]
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            education = st.selectbox(
                "Education",
                ["High School", "Graduate", "Post Graduate", "Professional Degree"],
            )
        with c2:
            city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        with c3:
            residence_type = st.selectbox(
                "Residence Type", ["Owned", "Rented", "Family Owned", "Other"]
            )

        number_of_dependants = st.number_input(
            "Number of Dependants", min_value=0, max_value=20, value=1, step=1
        )

        st.divider()
        st.subheader("Employment & Financial Information")

        c1, c2, c3 = st.columns(3)
        with c1:
            employment_type = st.selectbox(
                "Employment Type",
                [
                    "Salaried - Private",
                    "Salaried - Government",
                    "Self Employed",
                    "Business",
                ],
            )
        with c2:
            industry = st.selectbox(
                "Industry",
                [
                    "Information Technology",
                    "Banking & Financial Services",
                    "Manufacturing",
                    "Healthcare & Pharmaceuticals",
                    "Retail & Consumer",
                    "Automobile",
                    "Construction & Real Estate",
                    "Education",
                    "Telecommunications",
                    "Energy & Utilities",
                    "Logistics & Transportation",
                    "Government / Public Sector",
                    "Other",
                ],
            )
        with c3:
            years_in_current_job = st.number_input(
                "Years in Current Job / Business",
                min_value=0.0,
                max_value=60.0,
                value=4.0,
                step=0.1,
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            last_drawn_monthly_salary = st.number_input(
                "Last Drawn Monthly Salary (₹)",
                min_value=0.0,
                value=60000.0,
                step=1000.0,
            )
        with c2:
            other_monthly_income = st.number_input(
                "Other Monthly Income (₹)",
                min_value=0.0,
                value=10000.0,
                step=1000.0,
            )
        with c3:
            net_worth = st.number_input(
                "Net Worth (₹)", min_value=0.0, value=500000.0, step=10000.0
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            credit_score = st.number_input(
                "Credit Score", min_value=300, max_value=900, value=760, step=1
            )
        with c2:
            existing_monthly_emi = st.number_input(
                "Existing Monthly EMI (₹)",
                min_value=0.0,
                value=10000.0,
                step=1000.0,
            )
        with c3:
            existing_loans = st.number_input(
                "Existing Loans", min_value=0, max_value=30, value=1, step=1
            )

        c1, c2 = st.columns(2)
        with c1:
            credit_card_utilisation = st.slider(
                "Credit Card Utilisation (%)", 0, 100, 25, 1
            )
        with c2:
            prior_default = st.selectbox(
                "Prior Default / Write-off",
                ["No", "Yes"],
                index=0,
            )

        st.warning(
            "Other Monthly Income is collected for the application but is NOT used by the current "
            "trained PD model."
        )

        st.divider()
        st.subheader("Loan Request")

        c1, c2, c3 = st.columns(3)
        with c1:
            requested_amount = st.number_input(
                "Requested Loan Amount (₹)",
                min_value=1000.0,
                value=100000.0,
                step=5000.0,
            )
        with c2:
            requested_tenure = st.selectbox(
                "Requested Tenure", [3, 6, 9, 12], index=1, format_func=lambda x: f"{x} months"
            )
        with c3:
            application_channel = st.selectbox(
                "Application Channel",
                ["Organic/Play Store", "Website", "Partner", "Mobile App"],
            )

        submitted = st.form_submit_button(
            "Check Eligibility", type="primary", use_container_width=True
        )

    if not submitted:
        return

    if last_drawn_monthly_salary <= 0:
        st.error("Please enter a valid last drawn monthly salary.")
        return

    if existing_monthly_emi < 0:
        st.error("Existing EMI cannot be negative.")
        return

    application = {
        "age": int(age),
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "city_tier": city_tier,
        "residence_type": residence_type,
        "number_of_dependants": int(number_of_dependants),
        "employment_type": employment_type,
        "industry": industry,
        "years_in_current_job": float(years_in_current_job),
        "last_drawn_monthly_salary": float(last_drawn_monthly_salary),
        "other_monthly_income": float(other_monthly_income),
        "net_worth": float(net_worth),
        "credit_score": int(credit_score),
        "existing_monthly_emi": float(existing_monthly_emi),
        "existing_loans": int(existing_loans),
        "credit_card_utilisation": float(credit_card_utilisation),
        "prior_default": prior_default,
        "requested_amount": float(requested_amount),
        "requested_tenure": int(requested_tenure),
        "application_channel": application_channel,
    }

    try:
        artifact, pipeline, feature_columns = load_model()
        model_input = build_model_input(application, feature_columns)
        pd_value = float(pipeline.predict_proba(model_input)[0, 1])

        policy = calculate_policy(
            pd_value=pd_value,
            salary=application["last_drawn_monthly_salary"],
            existing_emi=application["existing_monthly_emi"],
            tenure=application["requested_tenure"],
            requested_amount=application["requested_amount"],
        )

        result = {
            **policy,
            "pd": pd_value,
            "pd_percentage": pd_value * 100,
            "model_type": artifact.get("model_type", "RandomForestClassifier"),
            "trees": artifact.get("trees", 100),
            "trained_on_rows": artifact.get("trained_on_rows"),
        }

        st.session_state.latest_app = application
        st.session_state.latest_result = result
        st.session_state.inquiry_recorded = False

    except Exception as exc:
        st.error(
            "The model could not be loaded or run. Check the model file and the pinned "
            "scikit-learn version in requirements.txt."
        )
        st.exception(exc)
        return

    st.success("Assessment completed.")

    latest_app = st.session_state.latest_app
    latest_result = st.session_state.latest_result

    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.subheader("Maximum Indicative Eligible Amount")
        st.markdown(
            f"<h1 style='color:#2457e6;margin:0'>{money(latest_result['indicative_eligible_amount'])}</h1>",
            unsafe_allow_html=True,
        )
        st.caption(f"Requested amount: {money(latest_app['requested_amount'])}")

        if latest_result["indicative_eligible_amount"] <= 0:
            st.error(
                "Based on the current prototype risk policy, there is no indicative eligible amount."
            )
        elif latest_app["requested_amount"] <= latest_result["indicative_eligible_amount"]:
            st.success(
                "Your requested loan amount is within the maximum indicative eligible amount."
            )
        else:
            st.warning(
                "Your requested amount is above the current indicative eligible limit. "
                "A lower requested amount may fit within the assessment."
            )

        a, b = st.columns(2)
        a.metric("Baseline Affordability", money(latest_result["baseline_amount"]))
        b.metric(
            "Indicative Interest Rate",
            f"{latest_result['indicative_interest_rate']:.2f}% p.a.",
        )

        st.markdown("**Affordability formula**")
        st.code(
            "(Last Drawn Monthly Salary − Existing Monthly EMI) × 12 × 50%",
            language="text",
        )

    with c2:
        st.subheader("Machine Learning Risk Assessment")
        st.metric("Probability of Default", f"{latest_result['pd_percentage']:.2f}%")
        st.metric("Risk Category", latest_result["risk_category"])

        a, b = st.columns(2)
        a.metric("Credit Score", latest_app["credit_score"])
        b.metric("Existing EMI", money(latest_app["existing_monthly_emi"]))
        a.metric("Other Monthly Income", money(latest_app["other_monthly_income"]))
        b.metric(
            "Credit Card Utilisation",
            f"{latest_app['credit_card_utilisation']:.0f}%",
        )
        st.write(f"**Decision Status:** {latest_result['decision']}")

        st.info(
            "The PD is produced by the trained Random Forest model. "
            "The indicative amount and rate are then calculated by the policy layer."
        )

    eligible = float(latest_result["indicative_eligible_amount"])
    requested = float(latest_app["requested_amount"])
    principal = min(requested, eligible)

    if principal > 0 and latest_result["indicative_interest_rate"] > 0:
        st.subheader("Indicative Amortization Schedule")
        emi, total_interest, rows = calculate_amortization(
            principal,
            latest_result["indicative_interest_rate"],
            latest_app["requested_tenure"],
        )

        if requested > eligible:
            st.caption(
                "Illustration based on the maximum indicative eligible amount because "
                "the requested amount exceeds the current indicative limit."
            )
        else:
            st.caption(
                "Illustration based on the requested amount and indicative interest rate."
            )

        a, b, c = st.columns(3)
        a.metric("Loan Amount Used", money(principal))
        b.metric("Monthly EMI", money(emi))
        c.metric("Total Interest", money(total_interest))

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        accept_disabled = (
            latest_result["indicative_eligible_amount"] <= 0
            or latest_app["requested_amount"] > latest_result["indicative_eligible_amount"]
        )
        if st.button(
            "Accept & Continue",
            type="primary",
            use_container_width=True,
            disabled=accept_disabled,
        ):
            st.session_state.active_application = {
                **latest_app,
                "pd": latest_result["pd"],
                "risk_category": latest_result["risk_category"],
                "eligible_amount": latest_result["indicative_eligible_amount"],
                "indicative_interest_rate": latest_result["indicative_interest_rate"],
                "status": "Processing",
            }
            st.session_state.page = "My Loans"
            st.rerun()

    with col2:
        if st.button("Forgo", use_container_width=True):
            st.session_state.inquiry_recorded = True

    with col3:
        if st.button("Clear Assessment", use_container_width=True):
            st.session_state.latest_app = None
            st.session_state.latest_result = None
            st.session_state.inquiry_recorded = False
            st.rerun()

    if st.session_state.inquiry_recorded:
        st.warning(
            "Inquiry recorded. This is not treated as a completed loan outcome and is "
            "not used as a default-training record."
        )


def show_loans():
    st.header("My Loans")
    st.caption("Accepted applications and completed loans appear here.")

    if st.session_state.active_application:
        a = st.session_state.active_application
        st.success("Loan Application — Processing")
        st.write(
            f"{money(a['requested_amount'])} requested • "
            f"{a['requested_tenure']} months • "
            f"Indicative rate {a['indicative_interest_rate']:.2f}% p.a."
        )
        st.write(f"Indicative eligible amount: **{money(a['eligible_amount'])}**")
        st.write(f"Indicative PD: **{a['pd'] * 100:.2f}%**")
        st.write(f"Risk category: **{a['risk_category']}**")
    else:
        st.info("No active loan. Accept an eligible offer to create an application.")

    st.markdown("---")
    st.write("### Completed Loan — Demo")
    c1, c2 = st.columns([4, 1])
    with c1:
        st.write("₹30,000 • 6 months • Fully repaid")
    with c2:
        st.success("COMPLETED")


def apply_css():
    st.markdown(
        """
        <style>
        .stApp { background: #f5f7fb; color: #172033; }
        [data-testid="stSidebar"] { background: #ffffff; }
        .kicker {
            display:inline-block;
            padding:7px 11px;
            border-radius:999px;
            background:#edf2ff;
            color:#2457e6;
            font-size:12px;
            font-weight:800;
            letter-spacing:.5px;
        }
        .hero { padding: 20px 0 10px 0; }
        .hero h1 { font-size: 3.4rem; line-height: 1.05; letter-spacing:-1.5px; margin: 14px 0; }
        .hero h1 span { color:#2457e6; }
        .hero p { font-size:1.15rem; color:#697386; max-width:900px; }
        div[data-testid="stMetric"] {
            background:#ffffff;
            border:1px solid #e7eaf1;
            padding:14px;
            border-radius:14px;
        }
        .stButton > button { border-radius:10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


init_state()
apply_css()

with st.sidebar:
    st.markdown("## Finsta QuickLoans")
    page = st.radio(
        "Navigate",
        ["Home", "Check Eligibility", "My Loans"],
        index=["Home", "Check Eligibility", "My Loans"].index(
            st.session_state.page
        ),
    )
    st.session_state.page = page

    st.markdown("---")
    st.caption("Personal fintech project prototype")
    st.caption("Indicative eligibility only")

if page == "Home":
    show_home()
elif page == "Check Eligibility":
    show_eligibility()
else:
    show_loans()
