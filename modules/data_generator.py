"""
Synthetic Applicant Data Generator.
Produces realistic-looking but entirely fictional banking applicant records
using Faker. Maintains ground-truth labels for performance evaluation.
"""

import random
import string
import uuid
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
from config import (
    DEFAULT_NORMAL_COUNT, DEFAULT_SYNTHETIC_COUNT,
    DEFAULT_FRAUD_RING_COUNT, DEFAULT_NEW_COUNT,
    SCENARIO_SEEDS,
)

fake = Faker("en_IN")


# ── Helpers ────────────────────────────────────────────────────────────────

def _applicant_id():
    return "APP-" + "".join(random.choices(string.digits, k=8))


def _device_id():
    return "DEV-" + uuid.uuid4().hex[:12].upper()


def _id_ref():
    """Synthetic document reference – NOT a real ID number."""
    prefix = random.choice(["SYN", "REF", "DOC"])
    return f"{prefix}-" + "".join(random.choices(string.digits, k=10))


def _dob(min_age=21, max_age=65):
    days = random.randint(min_age * 365, max_age * 365)
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")


def _income(emp_type):
    if emp_type == "Salaried":
        return round(random.uniform(300_000, 2_000_000), 2)
    elif emp_type == "Self-Employed":
        return round(random.uniform(200_000, 5_000_000), 2)
    else:
        return round(random.uniform(80_000, 400_000), 2)


def _employment():
    emp_type = random.choice(["Salaried", "Self-Employed", "Contract", "Unemployed"])
    if emp_type == "Unemployed":
        employer = "N/A"
    else:
        employer = fake.company()
    return employer, emp_type


def _application_date(days_back=365):
    offset = random.randint(0, days_back)
    return (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")


def _base_record(seed_phone=None, seed_email=None,
                 seed_address=None, seed_device=None):
    """Generate one applicant record with optional shared attribute injection."""
    employer, emp_type = _employment()
    phone = seed_phone or fake.phone_number()
    email = seed_email or fake.email()
    address = seed_address or fake.street_address()
    city = fake.city()
    state = fake.state()
    pincode = fake.postcode()
    device = seed_device or _device_id()

    return {
        "applicant_id": _applicant_id(),
        "full_name": fake.name(),
        "dob": _dob(),
        "email": email,
        "phone": phone,
        "address": address,
        "city": city,
        "state": state,
        "pincode": pincode,
        "id_ref": _id_ref(),
        "device_id": device,
        "application_date": _application_date(),
        "employer": employer,
        "employment_type": emp_type,
        "annual_income": _income(emp_type),
        "application_status": "New",
        "ground_truth_label": "genuine",
        "applicant_type": "normal",
        "risk_score": 0.0,
        "risk_category": "Low",
        "risk_flags": [],
        "risk_explanation": "",
        "analyst_note": "",
    }


# ── Normal Applicants ──────────────────────────────────────────────────────

def _generate_normal(count=DEFAULT_NORMAL_COUNT):
    records = []
    # Inject a few legitimate family address sharers
    shared_addr = fake.street_address()
    shared_city = fake.city()
    shared_state = fake.state()
    n_family = max(2, count // 20)

    for i in range(count):
        rec = _base_record()
        if i < n_family:
            rec["address"] = shared_addr
            rec["city"] = shared_city
            rec["state"] = shared_state
        rec["ground_truth_label"] = "genuine"
        rec["applicant_type"] = "normal"
        records.append(rec)
    return records


# ── Synthetic Identity Applicants ──────────────────────────────────────────

def _generate_synthetic_identities(count=DEFAULT_SYNTHETIC_COUNT):
    """
    Synthetic identities: blend of genuine-looking + fabricated details
    + planted inconsistencies.
    """
    records = []
    for _ in range(count):
        rec = _base_record()
        # Plant one or more inconsistencies
        inconsistency_type = random.choice([
            "age_income",          # very high income for stated age
            "employer_income",     # unemployed but high income
            "address_pincode",     # pincode not matching state pattern
            "dob_format_odd",      # dob looks suspiciously round
            "reused_id_ref",       # id_ref prefix mismatch
            "email_name_mismatch", # email doesn't match name at all
        ])
        if inconsistency_type == "age_income":
            rec["dob"] = _dob(min_age=21, max_age=24)
            rec["annual_income"] = round(random.uniform(4_000_000, 10_000_000), 2)
        elif inconsistency_type == "employer_income":
            rec["employment_type"] = "Unemployed"
            rec["employer"] = "N/A"
            rec["annual_income"] = round(random.uniform(3_000_000, 8_000_000), 2)
        elif inconsistency_type == "address_pincode":
            rec["pincode"] = "000" + "".join(random.choices(string.digits, k=3))
        elif inconsistency_type == "dob_format_odd":
            rec["dob"] = f"{random.randint(1960,2000)}-01-01"
        elif inconsistency_type == "reused_id_ref":
            rec["id_ref"] = "SYN-" + "0" * 10   # suspiciously uniform
        elif inconsistency_type == "email_name_mismatch":
            rec["email"] = "xz" + "".join(random.choices(string.digits, k=6)) + "@tempmail.xyz"

        rec["ground_truth_label"] = "synthetic_fraud"
        rec["applicant_type"] = "synthetic_identity"
        records.append(rec)
    return records


# ── Fraud-Ring Applicants ──────────────────────────────────────────────────

def _generate_fraud_rings(count=DEFAULT_FRAUD_RING_COUNT):
    """
    Fraud rings: groups sharing phones/emails/devices/addresses.
    Uses 2-3 rings with guaranteed attribute sharing for reliable detection.
    """
    records = []

    # Distribute count across rings (2–3 rings, min 3 members each)
    n_rings = 2 if count < 10 else 3
    sizes   = []
    base    = count // n_rings
    for i in range(n_rings):
        sizes.append(base + (1 if i < count % n_rings else 0))

    for ring_idx, ring_size in enumerate(sizes):
        ring_size = max(3, ring_size)
        # Every member of a ring ALWAYS shares at least the ring phone
        shared_phone  = fake.phone_number()
        # 60% of rings also share a device
        shared_device = _device_id() if random.random() > 0.4 else None
        # 50% of rings share an email
        shared_email  = fake.email() if random.random() > 0.5 else None
        # 40% share an address
        shared_address = fake.street_address() if random.random() > 0.6 else None

        for member_idx in range(ring_size):
            # All members get the shared phone; other attrs probabilistic
            rec = _base_record(
                seed_phone=shared_phone,
                seed_email=shared_email  if (shared_email  and random.random() > 0.3) else None,
                seed_address=shared_address if (shared_address and random.random() > 0.4) else None,
                seed_device=shared_device if (shared_device and random.random() > 0.3) else None,
            )
            rec["ground_truth_label"] = "fraud_ring"
            rec["applicant_type"]     = "fraud_ring"
            records.append(rec)

        if len(records) >= count:
            break

    return records[:count]


# ── New / Isolated Applicants ──────────────────────────────────────────────

def _generate_new_identities(count=DEFAULT_NEW_COUNT):
    """
    Completely new applicants with no shared attributes.
    Some have subtle consistency issues to test non-graph detection.
    """
    records = []
    for i in range(count):
        rec = _base_record()
        if i % 2 == 0:
            # Subtle: apply date is in the future
            rec["application_date"] = (
                datetime.now() + timedelta(days=random.randint(1, 30))
            ).strftime("%Y-%m-%d")
        rec["ground_truth_label"] = "genuine"
        rec["applicant_type"] = "new_identity"
        records.append(rec)
    return records


# ── Main Generator ─────────────────────────────────────────────────────────

def generate_dataset(
    normal_count=DEFAULT_NORMAL_COUNT,
    synthetic_count=DEFAULT_SYNTHETIC_COUNT,
    fraud_ring_count=DEFAULT_FRAUD_RING_COUNT,
    new_count=DEFAULT_NEW_COUNT,
    seed=None,
):
    """
    Generate a complete mixed synthetic dataset.
    Returns a DataFrame with ground-truth labels (kept hidden from scoring).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)

    records = []
    records.extend(_generate_normal(normal_count))
    records.extend(_generate_synthetic_identities(synthetic_count))
    records.extend(_generate_fraud_rings(fraud_ring_count))
    records.extend(_generate_new_identities(new_count))

    # Shuffle to avoid ordering bias
    random.shuffle(records)

    df = pd.DataFrame(records)
    # Ensure unique applicant IDs
    df["applicant_id"] = [_applicant_id() for _ in range(len(df))]
    return df


# ── Scenario Generators ────────────────────────────────────────────────────

def scenario_synthetic_identity():
    """Scenario 1 – single synthetic identity applicant."""
    random.seed(SCENARIO_SEEDS["synthetic_identity"])
    Faker.seed(SCENARIO_SEEDS["synthetic_identity"])
    recs = _generate_synthetic_identities(1)
    df = pd.DataFrame(recs)
    df["applicant_id"] = [_applicant_id()]
    return df


def scenario_fraud_ring():
    """Scenario 2 – fraud ring with 5 connected applicants."""
    random.seed(SCENARIO_SEEDS["fraud_ring"])
    Faker.seed(SCENARIO_SEEDS["fraud_ring"])
    shared_phone = fake.phone_number()
    shared_device = _device_id()
    recs = []
    for _ in range(5):
        rec = _base_record(
            seed_phone=shared_phone if random.random() > 0.3 else None,
            seed_device=shared_device if random.random() > 0.3 else None,
        )
        rec["ground_truth_label"] = "fraud_ring"
        rec["applicant_type"] = "fraud_ring"
        recs.append(rec)
    df = pd.DataFrame(recs)
    df["applicant_id"] = [_applicant_id() for _ in range(len(df))]
    return df


def scenario_genuine_family():
    """Scenario 3 – family members sharing address (should NOT be flagged as fraud)."""
    random.seed(SCENARIO_SEEDS["genuine_family"])
    Faker.seed(SCENARIO_SEEDS["genuine_family"])
    shared_address = fake.street_address()
    shared_city = fake.city()
    shared_state = fake.state()
    shared_pincode = fake.postcode()
    recs = []
    for _ in range(3):
        rec = _base_record(seed_address=shared_address)
        rec["city"] = shared_city
        rec["state"] = shared_state
        rec["pincode"] = shared_pincode
        rec["ground_truth_label"] = "genuine"
        rec["applicant_type"] = "normal"
        recs.append(rec)
    df = pd.DataFrame(recs)
    df["applicant_id"] = [_applicant_id() for _ in range(len(df))]
    return df


def scenario_new_identity():
    """Scenario 4 – completely new applicant with no prior matches."""
    random.seed(SCENARIO_SEEDS["new_identity"])
    Faker.seed(SCENARIO_SEEDS["new_identity"])
    recs = _generate_new_identities(1)
    df = pd.DataFrame(recs)
    df["applicant_id"] = [_applicant_id()]
    return df
