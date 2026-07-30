import argparse
from collections import Counter
from Bio import SeqIO

DEFAULT_AMP_KEYWORDS = [
    "antimicrobial",
    "antibacterial",
    "antifungal",
    "antiviral peptide",
    "defensin",
    "cecropin",
    "magainin",
    "bacteriocin",
    "cathelicidin",
    "thionin",
    "amp",
    "host defense peptide",
    "antibiotic peptide",
    "ppep",
    "ppepdb",
]


def parse_fasta_flexible(fasta_file):
    """Yields (description, sequence) tuples.
    Handles standard FASTA files (with '>') and non-standard alternating line files (without '>').
    """
    records = list(SeqIO.parse(fasta_file, "fasta"))
    if records:
        for record in records:
            yield record.description, str(record.seq).upper()
        return

    # Fallback for plain header/sequence line pairs missing '>'
    with open(fasta_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        desc = lines[i]
        if desc.startswith(">"):
            desc = desc[1:].strip()

        if i + 1 < len(lines):
            seq = lines[i + 1].strip().upper()
            yield desc, seq
            i += 2
        else:
            break


def has_amp_keyword(description, keywords):
    desc = description.lower()
    return any(kw in desc for kw in keywords)


def charge_at_ph7(seq):
    # Rough net-charge estimate (Henderson-Hasselbalch, ignores sequence-position effects)
    pos = seq.count("K") + seq.count("R") + 0.1 * seq.count("H")
    neg = seq.count("D") + seq.count("E")
    return pos - neg


def convert_fasta_to_csv(
    fasta_file,
    output_csv,
    min_len=5,
    max_len=32,
    require_amp_keyword=False,
    keywords=None,
    output_format="tokens",
):
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    keywords = keywords or DEFAULT_AMP_KEYWORDS

    stats = Counter()
    seen = set()
    kept = []  # tuple of (description, sequence)

    for desc, seq in parse_fasta_flexible(fasta_file):
        stats["total_records"] += 1

        if not set(seq).issubset(valid_aa):
            stats["dropped_invalid_aa"] += 1
            continue

        if not (min_len <= len(seq) <= max_len):
            stats["dropped_length"] += 1
            continue

        if require_amp_keyword and not has_amp_keyword(desc, keywords):
            stats["dropped_no_amp_keyword"] += 1
            continue

        if seq in seen:
            stats["dropped_duplicate_sequence"] += 1
            continue
        seen.add(seq)

        kept.append((desc, seq))
        stats["kept"] += 1

    with open(output_csv, "w", encoding="utf-8") as f:
        if output_format == "csv":
            f.write("id,sequence\n")
            for desc, seq in kept:
                f.write(f'"{desc}","{seq}"\n')
        else:
            # Tokenized format: space-separated amino acids
            for _, seq in kept:
                f.write(" ".join(list(seq)) + "\n")

    # --- Diagnostics ---
    print("=== Filtering Summary ===")
    print(f"Total records in FASTA:       {stats['total_records']}")
    print(f"Dropped (invalid AA):         {stats['dropped_invalid_aa']}")
    print(
        f"Dropped (length outside {min_len}-{max_len}): {stats['dropped_length']}"
    )
    if require_amp_keyword:
        print(
            f"Dropped (no AMP keyword):     {stats['dropped_no_amp_keyword']}"
        )
    print(
        f"Dropped (duplicate sequence): {stats['dropped_duplicate_sequence']}"
    )
    print(f"Kept:                         {stats['kept']}")

    if kept:
        kept_seqs = [s for _, s in kept]
        lengths = [len(s) for s in kept_seqs]
        charges = [charge_at_ph7(s) for s in kept_seqs]
        frac_cationic = sum(1 for c in charges if c > 1) / len(kept_seqs)

        print("\n=== Sanity Check on Kept Sequences ===")
        print(
            f"Length: min={min(lengths)} max={max(lengths)} mean={sum(lengths)/len(lengths):.1f}"
        )
        print(
            f"Net charge (approx, pH 7): mean={sum(charges)/len(charges):.2f}, "
            f"% cationic (charge > +1): {frac_cationic * 100:.1f}%"
        )
        print(
            "Reference: Characterized AMPs are typically net cationic "
            "(commonly 40-70% with charge > +1). A much lower figure here "
            "is a sign this dataset is not AMP-enriched, regardless of how "
            "many sequences it contains."
        )

    print(f"\nWrote {len(kept)} sequences to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter and convert FASTA sequences for AMP training."
    )
    parser.add_argument("--fasta_file", default="new_deduplicated.fasta")
    parser.add_argument("--output_csv", default="train_plant_amps.csv")
    parser.add_argument("--min_len", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument(
        "--require_amp_keyword",
        action="store_true",
        help="Only keep records whose header matches an AMP keyword.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Custom keywords list to match against header.",
    )
    parser.add_argument(
        "--format",
        choices=["tokens", "csv"],
        default="tokens",
        help="Output format: 'tokens' (space-separated AA) or 'csv' (tabular id,sequence).",
    )
    args = parser.parse_args()

    convert_fasta_to_csv(
        args.fasta_file,
        args.output_csv,
        min_len=args.min_len,
        max_len=args.max_len,
        require_amp_keyword=args.require_amp_keyword,
        keywords=args.keywords,
        output_format=args.format,
    )
