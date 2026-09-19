from flask import Flask, request, jsonify, render_template
import os
import joblib
import pandas as pd

app = Flask(__name__, template_folder="Templates")

# ============================================================
# Finsta QuickLoans — Compact ML Backend
# ============================================================

MODEL_PATH = "finsta_final_compact_credit_risk_model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Put the compact .joblib file in the same folder as app.py."
    )

artifact = joblib.load(MODEL_PATH)

pipeline = artifact["pipeline"]
FEATURE_COLUMNS = artifact["feature_columns"]

# Other Monthly Income is intentionally application-only.
REQUIRED_FIELDS = [
    "age",
    "gender",
    "marital_status",
    "education",
    "city_tier",
    "residence_type",
    "number_of_dependants",
    "employment_type",
    "industry",
    "years_in_current_job",
    "last_drawn_monthly_salary",
    "other_monthly_income",
    "net_worth",
    "credit_score",
    "existing_monthly_emi",
    "existing_loans",
    "credit_card_utilisation",
    "prior_default",
    "requested_amount",
    "requested_tenure",
    "application_channel",
]


def calculate_policy(pd_value, salary, existing_emi, tenure, requested_amount):
    # --------------------------------------------------------
    # 1. Baseline affordability
    # --------------------------------------------------------
    baseline_amount = max(
        0,
        (salary - existing_emi) * 12 * 0.50
    )

    # --------------------------------------------------------
    # 2. PD -> indicative eligible amount factor
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 3. PD -> indicative interest rate
    #    Prototype range: 12%–33% p.a.
    # --------------------------------------------------------
    pd_for_pricing = min(pd_value, 0.35)

    base_rate = 12.0
    risk_premium = 21.0 * (pd_for_pricing / 0.35)

    tenure_premium = {
        3: 0.0,
        6: 0.5,
        9: 1.0,
        12: 1.5
    }.get(tenure, 1.5)

    indicative_rate = min(
        33.0,
        max(
            12.0,
            base_rate + risk_premium + tenure_premium
        )
    )

    # --------------------------------------------------------
    # 4. Decision message
    # --------------------------------------------------------
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


def build_model_input(data):
    # These names/order match the compact model's feature schema.
    model_input = pd.DataFrame([{
        "Age": data["age"],
        "Gender": data["gender"],
        "Marital Status": data["marital_status"],
        "Education": data["education"],
        "City Tier": data["city_tier"],
        "Residence Type": data["residence_type"],
        "Employment Type": data["employment_type"],
        "Industry": data["industry"],
        "Years in Current Job / Business": float(
            data["years_in_current_job"]
        ),
        "Last Drawn Monthly Salary": float(
            data["last_drawn_monthly_salary"]
        ),
        "Net Worth": float(data["net_worth"]),
        "Credit Score": float(data["credit_score"]),
        "Existing Monthly EMI": float(
            data["existing_monthly_emi"]
        ),
        "Existing Loans": int(data["existing_loans"]),
        "Credit Card Utilisation": float(
            data["credit_card_utilisation"]
        ),
        "Prior Default / Write-off": data["prior_default"],
        "Application Channel": data["application_channel"],
        "Requested Loan Amount": float(data["requested_amount"]),
        "Requested Tenure": int(data["requested_tenure"]),
        "Number of Dependants": int(data["number_of_dependants"]),
    }])

    # Use the exact order stored with the trained model.
    return model_input[FEATURE_COLUMNS]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model_type": artifact.get(
            "model_type",
            "RandomForestClassifier"
        ),
        "trees": artifact.get("trees"),
        "trained_on_rows": artifact.get("trained_on_rows"),
        "feature_count": len(FEATURE_COLUMNS),
    })


@app.route("/api/eligibility", methods=["POST"])
def eligibility():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "error": "No JSON data received."
            }), 400

        missing = [
            field
            for field in REQUIRED_FIELDS
            if field not in data
        ]

        if missing:
            return jsonify({
                "error": "Missing required fields.",
                "missing_fields": missing
            }), 400

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------
        age = float(data["age"])
        salary = float(data["last_drawn_monthly_salary"])
        existing_emi = float(data["existing_monthly_emi"])
        credit_score = float(data["credit_score"])
        credit_util = float(data["credit_card_utilisation"])
        dependants = int(data["number_of_dependants"])
        existing_loans = int(data["existing_loans"])
        requested_amount = float(data["requested_amount"])
        requested_tenure = int(data["requested_tenure"])

        if age <= 0:
            raise ValueError("Age must be greater than zero.")

        if salary < 0 or existing_emi < 0:
            raise ValueError("Income and EMI cannot be negative.")

        if credit_score < 0:
            raise ValueError("Credit score cannot be negative.")

        if not 0 <= credit_util <= 100:
            raise ValueError(
                "Credit card utilisation must be between 0 and 100."
            )

        if dependants < 0 or existing_loans < 0:
            raise ValueError(
                "Dependants and existing loans cannot be negative."
            )

        if requested_amount <= 0:
            raise ValueError(
                "Requested loan amount must be greater than zero."
            )

        if requested_tenure not in [3, 6, 9, 12]:
            raise ValueError(
                "Requested tenure must be 3, 6, 9 or 12 months."
            )

        # ----------------------------------------------------
        # ML prediction
        # ----------------------------------------------------
        model_input = build_model_input(data)

        pd_value = float(
            pipeline.predict_proba(model_input)[0, 1]
        )

        # ----------------------------------------------------
        # Policy layer
        # ----------------------------------------------------
        policy = calculate_policy(
            pd_value=pd_value,
            salary=salary,
            existing_emi=existing_emi,
            tenure=requested_tenure,
            requested_amount=requested_amount,
        )

        return jsonify({
            "pd": round(pd_value, 6),
            "pd_percentage": round(pd_value * 100, 2),
            "risk_category": policy["risk_category"],
            "baseline_amount": policy["baseline_amount"],
            "indicative_eligible_amount": (
                policy["indicative_eligible_amount"]
            ),
            "indicative_interest_rate": (
                policy["indicative_interest_rate"]
            ),
            "decision": policy["decision"],
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
        })

    except ValueError as e:
        return jsonify({
            "error": str(e)
        }), 400

    except Exception as e:
        return jsonify({
            "error": "Unable to process eligibility request.",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
