"""
CICIoT2023 Label Mapping

Maps the 34 original CICIoT2023 labels into
9 higher-level attack categories.

This mapping is used consistently throughout
the preprocessing, training, evaluation and
Streamlit inference pipeline.
"""


LABEL_MAPPING = {
    # =========================================================
    # BENIGN
    # =========================================================

    "BENIGN": "Benign",

    # =========================================================
    # DDOS
    # =========================================================

    "DDOS-ICMP_FLOOD": "DDoS",
    "DDOS-UDP_FLOOD": "DDoS",
    "DDOS-TCP_FLOOD": "DDoS",
    "DDOS-PSHACK_FLOOD": "DDoS",
    "DDOS-SYN_FLOOD": "DDoS",
    "DDOS-RSTFINFLOOD": "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDoS",
    "DDOS-ICMP_FRAGMENTATION": "DDoS",
    "DDOS-UDP_FRAGMENTATION": "DDoS",
    "DDOS-ACK_FRAGMENTATION": "DDoS",
    "DDOS-HTTP_FLOOD": "DDoS",
    "DDOS-SLOWLORIS": "DDoS",

    # =========================================================
    # DOS
    # =========================================================

    "DOS-UDP_FLOOD": "DoS",
    "DOS-TCP_FLOOD": "DoS",
    "DOS-SYN_FLOOD": "DoS",
    "DOS-HTTP_FLOOD": "DoS",

    # =========================================================
    # MIRAI
    # =========================================================

    "MIRAI-GREETH_FLOOD": "Mirai",
    "MIRAI-UDPPLAIN": "Mirai",
    "MIRAI-GREIP_FLOOD": "Mirai",

    # =========================================================
    # RECON
    # =========================================================

    "RECON-HOSTDISCOVERY": "Recon",
    "RECON-OSSCAN": "Recon",
    "RECON-PORTSCAN": "Recon",
    "RECON-PINGSWEEP": "Recon",

    # =========================================================
    # SPOOFING
    # =========================================================

    "MITM-ARPSPOOFING": "Spoofing",
    "DNS_SPOOFING": "Spoofing",

    # =========================================================
    # WEB-BASED ATTACKS
    # =========================================================

    "XSS": "WebAttack",
    "SQLINJECTION": "WebAttack",
    "COMMANDINJECTION": "WebAttack",
    "BROWSERHIJACKING": "WebAttack",
    "UPLOADING_ATTACK": "WebAttack",

    # =========================================================
    # BRUTE FORCE
    # =========================================================

    "DICTIONARYBRUTEFORCE": "BruteForce",

    # =========================================================
    # OTHER ATTACKS
    # =========================================================

    "VULNERABILITYSCAN": "OtherAttack",
    "BACKDOOR_MALWARE": "OtherAttack",
}


# Expected high-level classes
HIGH_LEVEL_CLASSES = [
    "Benign",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "WebAttack",
    "BruteForce",
    "OtherAttack",
]


def map_label(label):
    """
    Convert an original CICIoT2023 label
    into its high-level category.
    """

    label = str(label).strip()

    if label not in LABEL_MAPPING:
        raise ValueError(
            f"Unknown CICIoT2023 label encountered: {label}"
        )

    return LABEL_MAPPING[label]


def validate_mapping(labels):
    """
    Verify that every dataset label has
    a corresponding mapping.
    """

    labels = set(str(label).strip() for label in labels)

    missing = labels - set(LABEL_MAPPING.keys())

    if missing:
        raise ValueError(
            "The following labels are missing from LABEL_MAPPING:\n"
            + "\n".join(sorted(missing))
        )

    return True


if __name__ == "__main__":

    print("=" * 70)
    print("CICIoT2023 LABEL MAPPING")
    print("=" * 70)

    print(f"\nOriginal labels mapped : {len(LABEL_MAPPING)}")
    print(f"High-level classes     : {len(HIGH_LEVEL_CLASSES)}")

    print("\nHigh-level classes:")

    for index, class_name in enumerate(
        HIGH_LEVEL_CLASSES,
        start=1
    ):
        print(f"{index}. {class_name}")

    print("\nMapping validation:")

    if len(LABEL_MAPPING) == 34:
        print("✓ All 34 expected labels are mapped.")
    else:
        print(
            f"WARNING: Expected 34 labels, "
            f"found {len(LABEL_MAPPING)} mappings."
        )

    print("\nDetailed mapping:")

    for original, grouped in LABEL_MAPPING.items():
        print(f"{original:35} → {grouped}")