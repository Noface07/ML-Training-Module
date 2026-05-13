"""
Column-name parser for the ``tag_{TAG_ID}_{FEATURE_SUFFIX}`` convention.

**Dynamic feature discovery**: instead of hard-coding a fixed list of 13
suffixes, this module *discovers* every suffix present in the dataset at
runtime.  A small set of suffixes (``raw``, ``roll_mean``, ``roll_std``,
``roc_1``) are still treated as **mandatory** (every tag is expected to
have them); everything else found in ``tag_*`` columns is classified as
**optional** automatically.

**Suffix overrides**: users can force classification by appending
``_MAN`` or ``_OPT`` to any custom suffix:

- ``tag_100_my_feature_MAN`` → suffix ``my_feature_MAN`` → **mandatory**
- ``tag_100_my_feature_OPT`` → suffix ``my_feature_OPT`` → **optional**

Cross-tag features and special columns (``timestamp``, ``will_fail``) are
detected by exact name match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ── Pattern ────────────────────────────────────────────────────────────
_TAG_COL_RE = re.compile(r"^tag_(\d+)_(.+)$")

# ── Mandatory suffixes (always expected) ───────────────────────────────
MANDATORY_SUFFIXES: set[str] = {"raw", "roll_mean", "roll_std", "roc_1"}

# ── Well-known optional suffixes (for documentation only — the parser
#    treats *any* suffix not in MANDATORY_SUFFIXES as optional) ─────────
WELL_KNOWN_OPTIONAL_SUFFIXES: set[str] = {
    "pct_range", "dist_to_max", "dist_to_min",
    "roll_min", "roll_max", "roc_3", "accel",
    "steps_to_max", "steps_to_min",
}

# ── Cross-tag features ─────────────────────────────────────────────────
CROSS_TAG_FEATURES: set[str] = {
    "system_avg_pct", "system_max_pct", "system_tags_high",
}

# ── Special columns ───────────────────────────────────────────────────
TARGET_COLUMN = "will_fail"
TIMESTAMP_COLUMN = "timestamp"


@dataclass
class ParsedColumn:
    """Result of parsing a single column name."""
    original: str
    tag_id: str | None = None
    suffix: str | None = None
    is_tag_feature: bool = False
    is_cross_tag: bool = False
    is_target: bool = False
    is_timestamp: bool = False


@dataclass
class FeatureSchema:
    """Structured classification of all columns in a dataset.

    ``per_tag_features`` maps each suffix to whether it is mandatory or
    optional.  The lists are built **dynamically** from whatever columns
    are actually present in the dataset.
    """
    tags: list[str] = field(default_factory=list)
    per_tag_features_mandatory: list[str] = field(default_factory=list)
    per_tag_features_optional: list[str] = field(default_factory=list)
    cross_tag_available: list[str] = field(default_factory=list)
    cross_tag_present: list[str] = field(default_factory=list)
    target_col: str | None = None
    target_present: bool = False
    total_columns: int = 0
    tag_column_map: dict[str, dict[str, str]] = field(default_factory=dict)
    all_suffixes: set[str] = field(default_factory=set)


def parse_column(col: str) -> ParsedColumn:
    """Parse a single column name into its components.

    Args:
        col: Column name string.

    Returns:
        A ``ParsedColumn`` with tag_id / suffix populated for tag
        feature columns, or special flags set for cross-tag / target /
        timestamp columns.
    """
    # Check special columns first
    if col == TIMESTAMP_COLUMN:
        return ParsedColumn(original=col, is_timestamp=True)
    if col == TARGET_COLUMN:
        return ParsedColumn(original=col, is_target=True)

    m = _TAG_COL_RE.match(col)
    if m:
        return ParsedColumn(
            original=col,
            tag_id=m.group(1),
            suffix=m.group(2),
            is_tag_feature=True,
        )
    # Any other column is treated as a dynamically discovered cross-tag feature
    return ParsedColumn(original=col, is_cross_tag=True)


def parse_columns(columns: list[str]) -> FeatureSchema:
    """Parse a full list of dataset column names.

    This is the main entry-point.  It returns a :class:`FeatureSchema`
    describing all tags, per-tag features (mandatory vs optional —
    discovered dynamically), cross-tag features, and the target column.

    Args:
        columns: Column name list from the parquet schema.

    Returns:
        Fully-populated ``FeatureSchema``.
    """
    tag_map: dict[str, dict[str, str]] = {}
    cross_tag_present: list[str] = []
    target_present = False
    all_suffixes: set[str] = set()

    for col in columns:
        pc = parse_column(col)
        if pc.is_tag_feature:
            tag_map.setdefault(pc.tag_id, {})[pc.suffix] = col  # type: ignore[arg-type]
            all_suffixes.add(pc.suffix)  # type: ignore[arg-type]
        elif pc.is_cross_tag:
            cross_tag_present.append(col)
        elif pc.is_target:
            target_present = True

    # Classify suffixes into mandatory / optional dynamically.
    # Respect user-specified overrides:
    #   - Suffix ending with ``_MAN`` → forced mandatory
    #   - Suffix ending with ``_OPT`` → forced optional
    #   - Otherwise fall back to the default MANDATORY_SUFFIXES set
    mandatory: list[str] = []
    optional: list[str] = []
    for s in sorted(all_suffixes):
        if s.endswith("_MAN"):
            mandatory.append(s)
        elif s.endswith("_OPT"):
            optional.append(s)
        elif s in MANDATORY_SUFFIXES:
            mandatory.append(s)
        else:
            optional.append(s)

    tags = sorted(tag_map.keys())

    return FeatureSchema(
        tags=tags,
        per_tag_features_mandatory=mandatory,
        per_tag_features_optional=optional,
        cross_tag_available=sorted(set(CROSS_TAG_FEATURES) | set(cross_tag_present)),
        cross_tag_present=sorted(cross_tag_present),
        target_col=TARGET_COLUMN if target_present else None,
        target_present=target_present,
        total_columns=len(columns),
        tag_column_map=tag_map,
        all_suffixes=all_suffixes,
    )


def build_column_list(
    tags: list[str],
    mandatory_suffixes: list[str],
    optional_suffixes: list[str],
    include_cross_tag: bool,
    cross_tag_present: list[str],
    tag_column_map: dict[str, dict[str, str]] | None = None,
    target_col: str | None = None,
    selected_cross_tag_features: list[str] | None = None,
) -> list[str]:
    """Build the ordered list of columns to select for training.

    Only includes columns that **actually exist** for each tag.  When
    ``tag_column_map`` is provided (recommended), a suffix is only
    emitted for a tag if the map contains that tag→suffix entry.

    Args:
        tags: Tag IDs to include.
        mandatory_suffixes: Mandatory per-tag suffixes.
        optional_suffixes: Selected optional per-tag suffixes.
        include_cross_tag: Whether to include cross-tag features.
        cross_tag_present: Cross-tag columns that actually exist.
        tag_column_map: ``{tag_id: {suffix: col_name, …}, …}`` from
            ``FeatureSchema.tag_column_map``.  When provided, only
            columns present in this map are emitted.
        target_col: Target column name (appended last).
        selected_cross_tag_features: List of specific cross-tag features to include.

    Returns:
        Ordered list of column names.
    """
    cols: list[str] = []
    for tag in sorted(tags):
        tag_suffixes = tag_column_map.get(tag, {}) if tag_column_map else None
        for suffix in mandatory_suffixes:
            if tag_suffixes is None or suffix in tag_suffixes:
                cols.append(f"tag_{tag}_{suffix}")
        for suffix in optional_suffixes:
            if tag_suffixes is None or suffix in tag_suffixes:
                cols.append(f"tag_{tag}_{suffix}")
    if selected_cross_tag_features is not None and len(selected_cross_tag_features) > 0:
        cols.extend(sorted(f for f in selected_cross_tag_features if f in cross_tag_present))
    elif include_cross_tag:
        cols.extend(sorted(cross_tag_present))
    if target_col:
        cols.append(target_col)
    return cols


def build_mandatory_columns(
    tags: list[str],
    mandatory_suffixes: list[str],
    tag_column_map: dict[str, dict[str, str]],
) -> list[str]:
    """Build the list of mandatory columns (always included in training).

    Args:
        tags: Tag IDs to include.
        mandatory_suffixes: Mandatory per-tag suffixes.
        tag_column_map: ``{tag_id: {suffix: col_name}}`` from the schema.

    Returns:
        Ordered list of mandatory column names.
    """
    cols: list[str] = []
    for tag in sorted(tags):
        tag_suffs = tag_column_map.get(tag, {})
        for suffix in mandatory_suffixes:
            if suffix in tag_suffs:
                cols.append(f"tag_{tag}_{suffix}")
    return cols


def build_optional_groups(
    tags: list[str],
    optional_suffixes: list[str],
    tag_column_map: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    """Group optional columns by suffix for combinatorial feature selection.

    Returns a dict like::

        {
            "pct_range": ["tag_100_pct_range", "tag_200_pct_range"],
            "custom_feature_1": ["tag_100_custom_feature_1"],
        }

    A suffix is only included if at least one tag actually has it.

    Args:
        tags: Tag IDs to include.
        optional_suffixes: Selected optional per-tag suffixes.
        tag_column_map: ``{tag_id: {suffix: col_name}}`` from the schema.

    Returns:
        Dict mapping suffix → list of actual column names.
    """
    groups: dict[str, list[str]] = {}
    for suffix in sorted(optional_suffixes):
        cols: list[str] = []
        for tag in sorted(tags):
            if suffix in tag_column_map.get(tag, {}):
                cols.append(f"tag_{tag}_{suffix}")
        if cols:
            groups[suffix] = cols
    return groups

