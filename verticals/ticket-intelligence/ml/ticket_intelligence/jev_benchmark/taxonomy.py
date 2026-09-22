"""Label spaces and Jev criteria for the parent_queue benchmark.

Two taxonomies are scored so the Jev result is comparable to BOTH published
ModernBERT numbers: the 10-class original (0.5229) and the merged clean_v1
space used for the ~0.73 run.
"""

from __future__ import annotations

# 10-class original taxonomy, with criteria written to separate the queues that
# the ModernBERT confusion matrix showed bleeding into each other
# (technical / it / product / customer_service accounted for the largest cells).
CRITERIA_10 = {
    "technical_support": (
        "The product itself is malfunctioning: errors, crashes, failed jobs, broken "
        "integrations, API or database faults, or incorrect behaviour the customer "
        "needs debugged. Choose this when the fault is in the vendor's software."
    ),
    "it_support": (
        "The customer's own environment or access is the problem: login and account "
        "lockouts, passwords and SSO, device or network setup, installs, upgrades, "
        "permissions and provisioning. Choose this when the fault sits on the "
        "customer's side rather than in the product's own logic."
    ),
    "product_support": (
        "The product is working as designed but the customer needs help using it: "
        "how-to questions, configuration guidance, feature explanations, best "
        "practice, or confusion about what a feature does. No defect is reported."
    ),
    "customer_service": (
        "An account or relationship matter with no technical fault: complaints, "
        "dissatisfaction, cancellations, contact or plan changes, chasing a prior "
        "ticket, or general service recovery."
    ),
    "billing_and_payments": (
        "Money owed or charged: invoices, payment failures, duplicate or incorrect "
        "charges, refunds, subscription and pricing changes, tax and billing details."
    ),
    "returns_and_exchanges": (
        "Physical goods moving back or being swapped: returns, exchanges, RMAs, "
        "wrong or damaged items received, shipping a replacement."
    ),
    "service_outages_and_maintenance": (
        "A shared service is down or degraded for many users, or scheduled "
        "maintenance is the subject: outages, downtime, widespread unavailability, "
        "maintenance windows and status updates. Distinct from a single customer's "
        "isolated technical fault."
    ),
    "sales_and_pre_sales": (
        "A prospective or expanding purchase: quotes, pricing for new business, "
        "demos, trials, capability questions asked before buying, upgrades and "
        "contract expansion."
    ),
    "human_resources": (
        "Employment matters: recruitment and applications, onboarding, payroll and "
        "benefits, internal staff or workplace policy questions."
    ),
    "general_inquiry": (
        "A generic question that fits none of the other queues and reports no fault: "
        "broad information requests, company or policy questions, or messages too "
        "vague to place elsewhere."
    ),
}

LABELS_10 = list(CRITERIA_10)

# clean_v1 merge, matching train_modernbert.py's --label-map clean_v1.
CLEAN_V1_MAP = {
    "technical_support": "technical_product_support",
    "it_support": "technical_product_support",
    "product_support": "technical_product_support",
    "customer_service": "customer_general",
    "general_inquiry": "customer_general",
}


def to_clean_v1(label: str) -> str:
    """Map a 10-class label into the merged clean_v1 space (7 classes)."""
    return CLEAN_V1_MAP.get(label, label)


CRITERIA_CLEAN_V1 = {
    "technical_product_support": (
        "The customer needs something in the product made to work, or needs to be "
        "shown how to operate it. Pick this only when a support engineer would have "
        "to touch the product, the account's configuration, or the customer's "
        "environment to resolve it: an error or crash, a feature behaving wrongly, "
        "a setup, integration or login that will not complete, or a how-to question "
        "about using a feature. Do NOT pick this merely because the message names "
        "the product, describes frustration with it, or asks what the company will "
        "do about a past problem."
    ),
    "customer_general": (
        "The customer wants a person or the company to respond, rather than wanting "
        "the product fixed or explained. This covers dissatisfaction and complaints "
        "about the service received, chasing an earlier ticket or a missing reply, "
        "cancelling or changing an account or plan, updating contact details, asking "
        "what the company's policy or process is, and any broad or vague enquiry "
        "that names no specific fault. When a message both complains about handling "
        "AND mentions a product problem, but the customer is asking for a response "
        "or a remedy rather than a fix, choose this queue."
    ),
    "billing_and_payments": CRITERIA_10["billing_and_payments"],
    "returns_and_exchanges": CRITERIA_10["returns_and_exchanges"],
    "service_outages_and_maintenance": CRITERIA_10["service_outages_and_maintenance"],
    "sales_and_pre_sales": CRITERIA_10["sales_and_pre_sales"],
    "human_resources": CRITERIA_10["human_resources"],
}

LABELS_CLEAN_V1 = list(CRITERIA_CLEAN_V1)

TAXONOMIES = {
    "original_10": (LABELS_10, CRITERIA_10),
    "clean_v1": (LABELS_CLEAN_V1, CRITERIA_CLEAN_V1),
}

INSTRUCTIONS = (
    "A customer has written to a support desk. Decide which single operational "
    "queue should own this ticket, based only on the subject and body. Judge what "
    "the customer actually needs done, not the vocabulary they happen to use."
)
