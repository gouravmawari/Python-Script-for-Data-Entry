import os
import re
import sys
import pandas as pd
from rapidfuzz import fuzz

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

SRC1 = os.path.join(DATA_DIR, "1787203408799_source1_naukri_applicants.csv")
SRC2 = os.path.join(DATA_DIR, "1787203408799_source2_gig_workers.csv")
SRC3 = os.path.join(DATA_DIR, "1787203408798_source3_cbnexus_contacts.csv")


# ---------------------------------------------------------------------------
# Normalization helpers (vectorized where possible — no row-by-row loops)
# ---------------------------------------------------------------------------

def normalize_phone(series: pd.Series) -> pd.Series:
    """
    Strip all non-digits, then drop a leading '91' country code if the
    result is 12 digits, leaving a clean 10-digit Indian mobile number.
    Handles: '9000000254', '919000000254', '+91-9000000131', '91 9000000254'
    """
    digits = series.astype(str).str.replace(r"\D", "", regex=True)
    digits = digits.where(
        ~((digits.str.len() == 12) & digits.str.startswith("91")),
        digits.str[2:],
    )
    digits = digits.replace("", pd.NA)
    return digits.astype("string")  # pandas nullable string dtype, never becomes float


def normalize_email(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().replace("nan", pd.NA)


def normalize_text(series: pd.Series) -> pd.Series:
    """Title-case text fields like city/name for comparison + display."""
    return series.astype(str).str.strip().str.title().replace("Nan", pd.NA)


def normalize_name(series: pd.Series) -> pd.Series:
    return normalize_text(series)


# ---------------------------------------------------------------------------
# Source 1 — Naukri applicants
# ---------------------------------------------------------------------------

def load_source1() -> pd.DataFrame:
    df = pd.read_csv(SRC1)

    df = df.rename(columns={
        "Full Name": "name",
        "Email": "email",
        "Phone": "phone",
        "City": "city",
        "Experience (Years)": "experience_years",
        "Current CTC": "ctc",
        "Applied Date": "applied_date",
        "Skills": "skills",
    })

    df["name"] = normalize_name(df["name"])
    df["email"] = normalize_email(df["email"])
    df["phone"] = normalize_phone(df["phone"])
    df["city"] = normalize_text(df["city"])
    df["skills"] = df["skills"].astype(str).str.strip()

    # Issue: CTC has implausible tiny values (e.g. 4.2, 8.3) mixed with
    # normal salary values (e.g. 417964) — looks like some rows were
    # entered in lakhs instead of absolute rupees. Flag, don't guess-fix.
    df["ctc_suspect"] = df["ctc"] < 1000

    # Issue: dates arrive in at least 4 different formats
    # ('24-07-2026', '2026-08-08', '7 Jul 2026', '07/13/2026').
    # Parse with dayfirst inference per-row, fall back gracefully.
    df["applied_date_parsed"] = pd.to_datetime(
        df["applied_date"], errors="coerce", dayfirst=True
    )

    df["source"] = "naukri"
    return df


# ---------------------------------------------------------------------------
# Source 2 — Gig workers
# ---------------------------------------------------------------------------

def load_source2() -> pd.DataFrame:
    df = pd.read_csv(SRC2)

    # Issue: at least one fully-blank row -> drop
    df = df.dropna(how="all")

    # Issue: at least one row has its columns rotated left by one position
    # (email_id holds what should be skill_tags, worker_name holds the
    # email, rate holds the name, location holds the rate, status holds
    # the location, skill_tags holds the status). Detect via: a valid
    # email_id must contain '@'; if it doesn't but worker_name does,
    # the row is rotated -> shift every field back into place.
    cols = ["email_id", "worker_name", "rate", "location", "status", "skill_tags"]

    def fix_rotated_row(row):
        if (pd.isna(row["email_id"]) or "@" not in str(row["email_id"])) and \
           ("@" in str(row["worker_name"])):
            vals = [row[c] for c in cols]
            # value in position i actually belongs in position i-1
            rotated = vals[1:] + vals[:1]
            for c, v in zip(cols, rotated):
                row[c] = v
        return row

    df = df.apply(fix_rotated_row, axis=1)

    df = df.rename(columns={
        "email_id": "email",
        "worker_name": "name",
        "rate": "rate_raw",
        "location": "city",
        "status": "status",
        "skill_tags": "skills",
    })

    df["name"] = normalize_name(df["name"])
    df["email"] = normalize_email(df["email"])
    df["city"] = normalize_text(df["city"])
    df["skills"] = df["skills"].astype(str).str.strip()

    # Issue: status has inconsistent casing (Active/active/ACTIVE)
    df["status"] = df["status"].astype(str).str.strip().str.lower()

    # Issue: rate is mixed units — '1415/hr' vs '15k/month'. Parse both
    # into a normalized hourly rate so they're comparable.
    def parse_rate(val):
        if pd.isna(val):
            return pd.NA
        s = str(val).lower().strip()
        m = re.match(r"([\d.]+)\s*/\s*hr", s)
        if m:
            return float(m.group(1))
        m = re.match(r"([\d.]+)\s*k\s*/\s*month", s)
        if m:
            # assume ~22 working days x 8 hours = 176 hrs/month
            return round(float(m.group(1)) * 1000 / 176, 2)
        return pd.NA

    df["rate_per_hour_normalized"] = df["rate_raw"].apply(parse_rate)

    # Source 2 has no phone number at all -> explicit column for the merge step
    df["phone"] = pd.Series(pd.NA, index=df.index, dtype="string")

    df["source"] = "gig_workers"
    return df


# ---------------------------------------------------------------------------
# Source 3 — CBNexus contacts
# ---------------------------------------------------------------------------

def load_source3() -> pd.DataFrame:
    df = pd.read_csv(SRC3)

    # Issue: the header row got duplicated as a data row somewhere in the
    # middle of the file (Name == 'Name', Phone Number == 'Phone Number').
    df = df[df["Name"] != "Name"].copy()

    df = df.rename(columns={
        "Name": "name",
        "Phone Number": "phone",
        "City": "city",
        "Verified": "verified_raw",
        "Projects Completed": "projects_completed",
    })

    df["name"] = normalize_name(df["name"])
    df["phone"] = normalize_phone(df["phone"])
    df["city"] = normalize_text(df["city"])

    # Issue: Verified column uses Y/yes/Yes/No/N inconsistently -> bool
    df["verified"] = df["verified_raw"].astype(str).str.strip().str.lower().isin(
        ["y", "yes"]
    )

    df["email"] = pd.NA  # Source 3 has no email at all
    df["source"] = "cbnexus"
    return df


# ---------------------------------------------------------------------------
# Matching + merging across sources
# ---------------------------------------------------------------------------
#
# A single row-level key (email OR phone) is not enough: e.g. two rows can
# share an email (linking naukri <-> gig_workers), and a *different* pair
# of rows can share a phone (linking naukri <-> cbnexus) - and if the
# naukri row is common to both pairs, all three should end up as ONE
# person even though no two rows share the *same* single key. This is a
# classic connected-components problem, solved with Union-Find so that
# matches chain transitively across sources.

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def resolve_identity_groups(df: pd.DataFrame) -> pd.Series:
    """Return a group id per row such that rows referring to the same
    real person (linked by shared email OR shared phone, transitively)
    get the same group id."""
    uf = UnionFind(len(df))

    for key_col in ("email", "phone"):
        buckets = {}
        for idx, val in df[key_col].items():
            if pd.isna(val) or str(val).strip() in ("", "<NA>"):
                continue
            buckets.setdefault(val, []).append(idx)
        for idx_list in buckets.values():
            first = idx_list[0]
            for other in idx_list[1:]:
                uf.union(df.index.get_loc(first), df.index.get_loc(other))

    roots = [uf.find(df.index.get_loc(i)) for i in df.index]
    return pd.Series(roots, index=df.index)


def coalesce(series: pd.Series):
    """Return the first non-null value in a group, else NA."""
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else pd.NA


def coalesce_best_name(series: pd.Series):
    """
    Names can appear as abbreviations in one source ('R. Verma') and in
    full elsewhere ('Rohit Verma') for the exact same person. Prefer the
    longest / most complete-looking name rather than just the first seen.
    """
    non_null = [str(v) for v in series.dropna() if str(v).strip()]
    if not non_null:
        return pd.NA
    # Prefer names without a period (abbreviation marker) and then longest
    non_null.sort(key=lambda n: ("." in n, -len(n)))
    return non_null[0]


def _merge_skills(skills_series: pd.Series):
    """Combine skill lists from multiple sources into one deduped set,
    case-insensitively, but keep a readable Title Case display form."""
    all_skills = set()
    for entry in skills_series.dropna():
        entry = str(entry).strip()
        if not entry or entry.lower() == "nan":
            continue
        for skill in entry.split(","):
            skill = skill.strip()
            if skill:
                all_skills.add(skill.lower())
    if not all_skills:
        return pd.NA
    return ", ".join(sorted(s.title() if s not in ("n8n", "sql", "rest apis")
                             else {"n8n": "n8n", "sql": "SQL", "rest apis": "REST APIs"}[s]
                             for s in all_skills))


def merge_group(group: pd.DataFrame) -> pd.Series:
    sources = sorted(group["source"].unique().tolist())
    applied_date = coalesce(
        group.get("applied_date_parsed", pd.Series(dtype=object))
    )
    return pd.Series({
        "name": coalesce_best_name(group["name"]),
        "email": coalesce(group.get("email", pd.Series(dtype=object))),
        "phone": coalesce(group.get("phone", pd.Series(dtype=object))),
        "city": coalesce(group.get("city", pd.Series(dtype=object))),
        "skills": _merge_skills(group.get("skills", pd.Series(dtype=object))),
        "applied_date": applied_date.strftime("%Y-%m-%d") if pd.notna(applied_date) else pd.NA,
        "experience_years": coalesce(group.get("experience_years", pd.Series(dtype=object))),
        "ctc": coalesce(group.get("ctc", pd.Series(dtype=object))),
        "status": coalesce(group.get("status", pd.Series(dtype=object))),
        "rate_per_hour_normalized": coalesce(
            group.get("rate_per_hour_normalized", pd.Series(dtype=object))
        ),
        "verified": coalesce(group.get("verified", pd.Series(dtype=object))),
        "projects_completed": coalesce(
            group.get("projects_completed", pd.Series(dtype=object))
        ),
        "sources": ", ".join(sources),
        "source_count": len(sources),
    })


def quality_score(row) -> str:
    key_fields = [row["name"], row["email"], row["phone"], row["city"], row["skills"]]
    present = sum(1 for f in key_fields if pd.notna(f) and str(f).strip() not in ("", "nan"))
    if present == len(key_fields):
        return "complete"
    if present >= 3:
        return "partial"
    return "poor"


# ---------------------------------------------------------------------------
# Review queue — ambiguous cross-source matches we deliberately DON'T
# auto-merge, because guessing wrong here silently corrupts the dataset.
# A human (you, in the video) makes the final call on these.
# ---------------------------------------------------------------------------

NAME_SIMILARITY_THRESHOLD = 88  # 0-100, on first+last name compared separately


def _name_similarity(name_a: str, name_b: str) -> float:
    """
    Compare first and last name parts SEPARATELY and take the minimum.
    A plain whole-string ratio over-weights a shared surname alone
    (e.g. 'Rahul Malhotra' vs 'Sahil Malhotra' scores ~86% on a naive
    ratio despite being different people) - requiring both parts to be
    similar avoids that false-positive pattern.
    """
    pa, pb = name_a.lower().split(), name_b.lower().split()
    if len(pa) < 2 or len(pb) < 2:
        return fuzz.ratio(name_a.lower(), name_b.lower())
    return min(fuzz.ratio(pa[0], pb[0]), fuzz.ratio(pa[-1], pb[-1]))


def find_review_candidates(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Look for pairs of DIFFERENT merged people (i.e. not already linked by
    exact email/phone) whose names are suspiciously similar. These are
    NOT merged automatically - they're flagged with a reason so a human
    can decide. This is the same logic Task 2's n8n flow re-uses when
    checking a newly-arriving person against the existing database.
    """
    records = merged.to_dict("records")
    review_rows = []

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            name_a, name_b = str(a["name"]), str(b["name"])
            score = _name_similarity(name_a, name_b)

            if score < NAME_SIMILARITY_THRESHOLD:
                continue

            same_city = (
                pd.notna(a["city"]) and pd.notna(b["city"])
                and str(a["city"]).lower() == str(b["city"]).lower()
            )

            if score == 100:
                reason = (
                    "Identical name, different email AND different phone "
                    "(same city)" if same_city else
                    "Identical name, different email AND different phone, "
                    "different/unknown city — likely two different people, "
                    "kept separate"
                )
                # Only flag exact-name-but-unmerged as HIGH priority when
                # cities also match (raises real chance it's the same person)
                if not same_city:
                    continue
            else:
                reason = f"Similar name ({score}% match) — possible typo/duplicate"

            review_rows.append({
                "person_a_name": a["name"],
                "person_a_email": a["email"],
                "person_a_phone": a["phone"],
                "person_a_city": a["city"],
                "person_a_sources": a["sources"],
                "person_b_name": b["name"],
                "person_b_email": b["email"],
                "person_b_phone": b["phone"],
                "person_b_city": b["city"],
                "person_b_sources": b["sources"],
                "name_similarity_pct": score,
                "reason": reason,
            })

    return pd.DataFrame(review_rows)


def main():
    print("Loading and cleaning sources...")
    df1 = load_source1()
    df2 = load_source2()
    df3 = load_source3()

    print(f"  source1 (naukri):      {len(df1)} rows after cleaning")
    print(f"  source2 (gig_workers): {len(df2)} rows after cleaning")
    print(f"  source3 (cbnexus):     {len(df3)} rows after cleaning")

    # Common columns each source contributes (missing ones become NA on concat)
    combined = pd.concat([df1, df2, df3], ignore_index=True, sort=False)

    combined = combined.reset_index(drop=True)
    combined["match_group"] = resolve_identity_groups(combined)

    no_id = combined[combined["email"].isna() & combined["phone"].isna()]
    if len(no_id):
        print(f"  WARNING: {len(no_id)} row(s) had neither email nor phone — "
              f"kept as standalone unmatched records")

    print("Matching and merging records across sources (union-find over email+phone)...")
    merged = combined.groupby("match_group").apply(merge_group).reset_index(drop=True)
    merged["quality_score"] = merged.apply(quality_score, axis=1)

    print(f"\nResult: {len(combined)} raw rows across 3 sources -> "
          f"{len(merged)} unique merged people")
    print(f"  Matched from 2+ sources: {(merged['source_count'] > 1).sum()}")
    print(f"  Quality breakdown:\n{merged['quality_score'].value_counts().to_string()}")

    out_path = os.path.join(os.path.dirname(__file__), "merged_people.csv")
    merged.to_csv(out_path, index=False)
    print(f"\nSaved merged dataset -> {out_path}")

    print("\nScanning for ambiguous matches (same/similar name, NOT auto-merged)...")
    review_df = find_review_candidates(merged)
    review_path = os.path.join(os.path.dirname(__file__), "review_queue.csv")
    review_df.to_csv(review_path, index=False)
    print(f"  Found {len(review_df)} pair(s) needing human review -> {review_path}")
    if len(review_df):
        print(review_df[["person_a_name", "person_b_name", "name_similarity_pct", "reason"]].to_string(index=False))

    return merged, review_df


if __name__ == "__main__":
    main()
