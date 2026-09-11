from dataclasses import dataclass

@dataclass(frozen=True)
class Finding:
    area: str
    status: str
    statement: str

def build_findings() -> list[Finding]:
    return [
        Finding("Data and prediction integrity", "PASS",
                "Validation data and stored predictions satisfy the core integrity checks."),
        Finding("Discrimination", "ACCEPTABLE",
                "The baseline model provides meaningful rank-order discrimination."),
        Finding("Calibration", "ACCEPTABLE",
                "Aggregate calibration is close to the observed validation default rate."),
        Finding("Stability and robustness", "PASS",
                "Bootstrap and subsample analyses indicate stable aggregate performance."),
        Finding("Segment performance", "SCREENING SIGNAL",
                "Segment-level degradation is observed in a subset of segments and requires further investigation."),
    ]
