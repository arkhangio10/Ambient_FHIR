"""Entity validation checks — deterministic quality gate.

No AI calls. No external dependencies. Pure Python logic.
This module is called ONLY by the orchestrator.
The orchestrator calls run_all_checks() — NOTHING ELSE.

Never mutate the entities object — return ValidationReport only.
"""

import re

from app.schemas.entities import (
    ExtractedEntities,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)


# ── Required field rules ────────────────────────────────────────────
# Fields that MUST be present before serialization.
# Adding a field here automatically enforces it across the pipeline.

REQUIRED_FIELDS: list[str] = [
    "chief_complaint",
    "diagnosis",
    "procedure_or_intervention",
    "clinical_rationale",
]

# Fields required only when a parent field is present
CONDITIONAL_REQUIREMENTS: dict[str, list[str]] = {
    "medication": ["dosage", "frequency"],
}

# Fields that produce warnings if absent (not errors)
RECOMMENDED_FIELDS: list[str] = [
    "diagnosis_code",
    "procedure_code",
    "modifier_flags",
]

# Minimum character lengths for text fields
MIN_FIELD_LENGTHS: dict[str, int] = {
    "clinical_rationale": 20,
    "chief_complaint": 5,
    "diagnosis": 3,
}

# Confidence threshold below which a warning is issued
CONFIDENCE_WARNING_THRESHOLD: float = 0.65

# ICD-10 code pattern (basic format check only)
ICD10_PATTERN: str = r"^[A-Z]\d{2}(\.\d{1,4})?$"

# CPT code pattern (5 digits)
CPT_PATTERN: str = r"^\d{5}$"

# Max clarification rounds before forcing manual review
MAX_CLARIFICATION_ROUNDS: int = 3


# ── 1. check_required_fields ───────────────────────────────────────

def check_required_fields(
    entities: ExtractedEntities,
) -> list[ValidationIssue]:
    """Check every field in REQUIRED_FIELDS."""
    issues: list[ValidationIssue] = []
    entity_dict = entities.model_dump()

    for field in REQUIRED_FIELDS:
        value = entity_dict.get(field)
        if not value or (isinstance(value, str) and not value.strip()):
            issues.append(ValidationIssue(
                field=field,
                severity=ValidationSeverity.ERROR,
                message=f"Required field '{field}' is missing or empty",
                suggested_action=(
                    f"Please provide the {field.replace('_', ' ')} "
                    f"for this patient encounter"
                ),
            ))

    return issues


# ── 2. check_conditional_requirements ──────────────────────────────

def check_conditional_requirements(
    entities: ExtractedEntities,
) -> list[ValidationIssue]:
    """If a parent field is present, its dependent fields become required."""
    issues: list[ValidationIssue] = []
    entity_dict = entities.model_dump()

    for parent_field, dependent_fields in CONDITIONAL_REQUIREMENTS.items():
        parent_value = entity_dict.get(parent_field)

        if not parent_value or (
            isinstance(parent_value, str) and not parent_value.strip()
        ):
            continue

        for dep_field in dependent_fields:
            dep_value = entity_dict.get(dep_field)
            if not dep_value or (
                isinstance(dep_value, str) and not dep_value.strip()
            ):
                issues.append(ValidationIssue(
                    field=dep_field,
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"'{dep_field}' is required because "
                        f"'{parent_field}' is present ({parent_value})"
                    ),
                    suggested_action=(
                        f"Please specify the {dep_field.replace('_', ' ')} "
                        f"for {parent_value}"
                    ),
                ))

    return issues


# ── 3. check_field_quality ─────────────────────────────────────────

def check_field_quality(
    entities: ExtractedEntities,
) -> list[ValidationIssue]:
    """Check minimum length rules and format patterns."""
    issues: list[ValidationIssue] = []
    entity_dict = entities.model_dump()

    # Minimum length checks
    for field, min_len in MIN_FIELD_LENGTHS.items():
        value = entity_dict.get(field)
        if value and isinstance(value, str):
            if len(value.strip()) < min_len:
                issues.append(ValidationIssue(
                    field=field,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"'{field}' is unusually short "
                        f"({len(value.strip())} chars, minimum {min_len})"
                    ),
                    suggested_action=(
                        f"Consider expanding the {field.replace('_', ' ')}"
                    ),
                ))

    # ICD-10 format check
    if entities.diagnosis_code:
        if not re.match(ICD10_PATTERN, entities.diagnosis_code.strip().upper()):
            issues.append(ValidationIssue(
                field="diagnosis_code",
                severity=ValidationSeverity.WARNING,
                message=(
                    f"ICD-10 code format unrecognized: "
                    f"'{entities.diagnosis_code}'"
                ),
                suggested_action="Verify the ICD-10 code format (e.g. F33.1, F41.1)",
            ))

    # CPT format check
    if entities.procedure_code:
        if not re.match(CPT_PATTERN, entities.procedure_code.strip()):
            issues.append(ValidationIssue(
                field="procedure_code",
                severity=ValidationSeverity.WARNING,
                message=(
                    f"CPT code format unrecognized: "
                    f"'{entities.procedure_code}'"
                ),
                suggested_action="Verify the CPT code format (e.g. 90834, 90837)",
            ))

    return issues


# ── 4. check_recommended_fields ────────────────────────────────────

def check_recommended_fields(
    entities: ExtractedEntities,
) -> list[ValidationIssue]:
    """Check RECOMMENDED_FIELDS. Returns WARNING only — never blocks."""
    issues: list[ValidationIssue] = []
    entity_dict = entities.model_dump()

    for field in RECOMMENDED_FIELDS:
        value = entity_dict.get(field)

        # Special case: modifier_flags is a list
        if field == "modifier_flags":
            if not value:
                issues.append(ValidationIssue(
                    field=field,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        "No billing modifiers detected. "
                        "Behavioral health claims often "
                        "require modifiers (e.g. GT, 95)"
                    ),
                    suggested_action=(
                        "Confirm whether telehealth or "
                        "complexity modifiers apply"
                    ),
                ))
            continue

        if not value or (isinstance(value, str) and not value.strip()):
            issues.append(ValidationIssue(
                field=field,
                severity=ValidationSeverity.WARNING,
                message=f"Recommended field '{field}' is absent",
                suggested_action=(
                    f"Adding {field.replace('_', ' ')} improves claim accuracy"
                ),
            ))

    return issues


# ── 5. check_confidence ────────────────────────────────────────────

def check_confidence(
    entities: ExtractedEntities,
) -> list[ValidationIssue]:
    """Issue a warning if extraction confidence is below threshold."""
    issues: list[ValidationIssue] = []

    if entities.confidence < CONFIDENCE_WARNING_THRESHOLD:
        issues.append(ValidationIssue(
            field="confidence",
            severity=ValidationSeverity.WARNING,
            message=(
                f"Extraction confidence is low: "
                f"{entities.confidence:.0%}. "
                f"Manual review recommended before submission"
            ),
            suggested_action=(
                "Review extracted entities carefully "
                "before prior auth submission"
            ),
        ))

    return issues


# ── 6. run_all_checks — PRIMARY ENTRY POINT ───────────────────────

def run_all_checks(
    entities: ExtractedEntities,
    clarification_round: int = 0,
) -> ValidationReport:
    """Run all checks and aggregate into a single ValidationReport.

    This is the ONLY function the orchestrator calls.
    """
    all_issues: list[ValidationIssue] = []

    all_issues += check_required_fields(entities)
    all_issues += check_conditional_requirements(entities)
    all_issues += check_field_quality(entities)
    all_issues += check_recommended_fields(entities)
    all_issues += check_confidence(entities)

    errors = [i for i in all_issues if i.severity == ValidationSeverity.ERROR]
    warnings = [
        i.message for i in all_issues if i.severity == ValidationSeverity.WARNING
    ]
    missing_required = [i.field for i in errors]

    # Determine if clarification is still possible
    clarification_needed = (
        len(errors) > 0 and clarification_round < MAX_CLARIFICATION_ROUNDS
    )

    if len(errors) > 0 and clarification_round >= MAX_CLARIFICATION_ROUNDS:
        # Force pass with warning after max rounds
        warnings.append(
            f"Max clarification rounds ({MAX_CLARIFICATION_ROUNDS}) reached. "
            f"Proceeding with incomplete data. Manual review required."
        )
        passed = True
        clarification_needed = False
    else:
        passed = len(errors) == 0

    return ValidationReport(
        session_id=entities.session_id,
        passed=passed,
        issues=all_issues,
        missing_required=missing_required,
        warnings=warnings,
        clarification_needed=clarification_needed,
        first_missing_field=missing_required[0] if missing_required else None,
    )


# ── Legacy compatibility ───────────────────────────────────────────
# build_clarification_prompt is still used by the orchestrator


def build_clarification_prompt(
    missing_fields: list[str],
    entities: ExtractedEntities,
) -> str:
    """Build a concise clarification prompt for the FIRST missing field.

    Max 150 characters. Behavioral-health-context-aware phrasing.
    """
    if not missing_fields:
        return ""

    field = missing_fields[0]

    field_prompts: dict[str, str] = {
        "dosage": (
            f"What is the dosage for {entities.medication or 'the medication'}?"
        ),
        "frequency": (
            f"How often should {entities.medication or 'the medication'} be taken?"
        ),
        "diagnosis": (
            "Could you confirm the primary diagnosis for this patient?"
        ),
        "diagnosis_code": (
            f"What is the ICD-10 code for {entities.diagnosis or 'the diagnosis'}?"
        ),
        "procedure_code": (
            f"What CPT code applies to "
            f"{entities.procedure_or_intervention or 'this procedure'}?"
        ),
        "clinical_rationale": (
            "Could you briefly state the clinical rationale supporting this treatment?"
        ),
        "procedure_or_intervention": (
            "What treatment or procedure is being recommended for this patient?"
        ),
        "chief_complaint": (
            "What is the patient's primary presenting complaint today?"
        ),
        "medication": (
            "Is any medication being prescribed for this patient?"
        ),
    }

    prompt = field_prompts.get(
        field,
        f"Could you clarify the {field.replace('_', ' ')} for this patient?",
    )

    if len(prompt) > 150:
        prompt = prompt[:147] + "..."

    return prompt
