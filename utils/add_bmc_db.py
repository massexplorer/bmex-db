#   # Build BMC for all eligible quantities and overwrite /BMC
#   python add_bmc_db.py --h5 ../old_dbs/2-27-25.h5 --quantity ALL --overwrite --iterations 50000 --components 3
#
#   # Add/update BMC columns for all eligible quantities, merging into existing /BMC
#   python add_bmc_db.py --h5 ../old_dbs/2-27-25.h5 --quantity ALL --overwrite --iterations 50000 --components 3 

#   # Build BMC for a single quantity
#   python add_bmc_db.py --h5 ../old_dbs/2-27-25.h5 --quantity BE --merge

#   --store_bounds for upper/lower bound columns

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd

from pybmc.bmc import BayesianModelCombination
from pybmc.data import Dataset


DEFAULT_MODELS = [
    "BCPM",
    "D1M",
    "FRDM12",
    "HFB24",
    "ME2",
    "MEdelta",
    "NL3S",
    "PC1",
    "SKMS",
    "SKP",
    "SLY4",
    "SV",
    "UNEDF0",
    "UNEDF1",
    "UNEDF2",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update a BMEX-style HDF5 by adding/updating a 'BMC' dataset for one or all quantities, plus /BMC_META."
    )
    parser.add_argument("--h5", required=True, help="Path to BMEX HDF5 database (e.g., 2-27-25.h5).")
    parser.add_argument(
        "--quantity",
        default="BE",
        help="Quantity/column name to build BMC for, or ALL to iterate over AME2020 columns.",
    )

    parser.add_argument("--output_key", default="BMC", help="HDF5 key to write results into (default: BMC).")
    parser.add_argument("--meta_key", default="BMC_META", help="HDF5 key to write metadata into (default: BMC_META).")
    parser.add_argument("--truth", default="AME2020", help="Truth dataset/key name (default: AME2020).")

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Predictor model keys to consider (default: standard list excluding truth).",
    )

    parser.add_argument(
        "--domain_cols",
        nargs=2,
        default=["N", "Z"],
        help="Domain columns (default: N Z).",
    )

    parser.add_argument("--components", type=int, default=3, help="SVD components kept for orthogonalize().")
    parser.add_argument("--iterations", type=int, default=50000, help="Gibbs iterations for train().")

    parser.add_argument("--train_size", type=float, default=0.6, help="Train split fraction (default: 0.6).")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation split fraction (default: 0.2).")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split fraction (default: 0.2).")

    parser.add_argument(
        "--uncertainty_mode",
        choices=["half_interval", "sigma95"],
        default="half_interval",
        help="uQ: half_interval=(upper-lower)/2, sigma95=(upper-lower)/(2*1.96).",
    )

    parser.add_argument(
        "--store_bounds",
        action="store_true",
        help="Store quantity_lower and quantity_upper columns in the BMC dataset.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the entire output_key dataset if it exists.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge into existing output_key on domain cols, overwriting only updated quantity columns.",
    )

    parser.add_argument(
        "--min_overlap",
        type=int,
        default=50,
        help="Minimum overlap with truth required to include a model AND to run the quantity (default: 50).",
    )

    # ONLY CHANGE ADDED: output prediction domain
    parser.add_argument(
        "--predict_domain",
        choices=["truth", "model"],
        default="truth",
        help="Where to write BMC predictions: 'truth' outputs only nuclei with truth; 'model' outputs intersection of predictor domains.",
    )

    return parser.parse_args()

# checks split fractions sum to 1.0
def validate_sizes(train_size, val_size, test_size):
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0 (got {0}).".format(total))

# When --quantity ALL, you don’t want to run BMC on N, Z, uncertainty columns or error columns 
def is_bmc_target_quantity(col, domain_cols):
    if col in domain_cols:
        return False
    if col.startswith("u") or col.startswith("e"):
        return False
    return True

# reads truth dataset columns to find all possible quantities. This is what BMC is ran over when --quantity ALL
def list_ame_columns(h5_path, truth_key):
    with pd.HDFStore(h5_path, mode="r") as store:
        if "/" + truth_key not in store.keys():
            raise ValueError("Truth key '{0}' not found in HDF5.".format(truth_key))
        truth_df = store[truth_key]
    return list(truth_df.columns)

# per-model eligibility and overlap checker for one quantity
def model_column_stats(h5_path, model_key, truth_key, quantity, domain_cols):
    """
    Returns:
      exists (bool): key exists in HDF5
      has_col (bool): quantity column present
      non_nan (int): count of non-NaN in model quantity
      overlap (int): count of overlapping (N,Z) rows where truth and model both have quantity non-NaN
      reason (str or None): if excluded, why
    """
    key_path = "/" + model_key
    truth_path = "/" + truth_key

    with pd.HDFStore(h5_path, mode="r") as store:
        if key_path not in store.keys():
            return False, False, 0, 0, "missing_dataset"
        if truth_path not in store.keys():
            return False, False, 0, 0, "missing_truth_dataset"

        mdf = store[model_key]
        if quantity not in mdf.columns:
            return True, False, 0, 0, "missing_column"

        non_nan = int(mdf[quantity].notna().sum())
        if non_nan == 0:
            return True, True, 0, 0, "all_nan"

        truth_df = store[truth_key][list(domain_cols) + [quantity]].dropna(subset=[quantity]).drop_duplicates(subset=list(domain_cols))
        model_df = store[model_key][list(domain_cols) + [quantity]].dropna(subset=[quantity]).drop_duplicates(subset=list(domain_cols))

        merged = pd.merge(truth_df, model_df, on=list(domain_cols), how="inner", suffixes=("_t", "_m"))
        overlap = int(len(merged))

        return True, True, non_nan, overlap, None


def compute_overlap_all_models(h5_path, truth_key, models, quantity, domain_cols):
    """
    Overlap count after intersecting truth with ALL included predictors.
    Only uses domain columns to avoid suffix/duplicate-column issues.
    """
    with pd.HDFStore(h5_path, mode="r") as store:
        truth_df = store[truth_key][list(domain_cols) + [quantity]].dropna(subset=[quantity])
        df = truth_df[list(domain_cols)].drop_duplicates(subset=list(domain_cols))

        for m in models:
            mdf = store[m][list(domain_cols) + [quantity]].dropna(subset=[quantity])
            mdf = mdf[list(domain_cols)].drop_duplicates(subset=list(domain_cols))
            df = pd.merge(df, mdf, on=list(domain_cols), how="inner")
            if len(df) == 0:
                return 0

        return int(len(df))


def build_bmc_for_quantity(
    h5_path,
    quantity,
    predictors,
    truth,
    domain_cols,
    components,
    iterations,
    train_size,
    val_size,
    test_size,
    uncertainty_mode,
    min_overlap,
    predict_domain,  
):
    dataset = Dataset(h5_path, verbose=False)

    models_for_load = list(predictors) + [truth]
    data_dict = dataset.load_data(
        models=models_for_load,
        keys=[quantity],
        domain_keys=domain_cols,
        truth_column_name=truth,
    ) # creates one wide table (df) for 'quantity' where rows are nuclei and columns include N, Z, and all models + truth

    df = data_dict[quantity].copy() 

    if truth not in df.columns: # enfore truth exists
        raise ValueError("Truth column '{0}' not present after load_data for '{1}'.".format(truth, quantity))

    df_with_truth = df.dropna(subset=[truth]).reset_index(drop=True) # throw away every row where truth is missing
    if len(df_with_truth) < min_overlap:
        raise ValueError(
            "Too few rows after requiring truth for '{0}': {1} (< {2}).".format(quantity, len(df_with_truth), min_overlap)
        )

    validate_sizes(train_size, val_size, test_size)
    train_df, val_df, test_df = dataset.split_data(
        {quantity: df_with_truth}, # dict with one entry
        quantity,
        splitting_algorithm="random",
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    bmc = BayesianModelCombination(
        models_list=list(predictors), 
        data_dict=data_dict,
        truth_column_name=truth,
    )

    bmc.orthogonalize(quantity, train_df, components_kept=components)
    bmc.train(training_options={"iterations": iterations})

    rndm_m, lower_df, median_df, upper_df = bmc.predict(quantity)

    # Prepare prediction table
    pred = median_df[list(domain_cols)].copy()
    pred[quantity] = pd.to_numeric(median_df["Predicted_Median"], errors="coerce")
    pred[quantity + "_lower"] = pd.to_numeric(lower_df["Predicted_Lower"], errors="coerce")
    pred[quantity + "_upper"] = pd.to_numeric(upper_df["Predicted_Upper"], errors="coerce")
    pred = pred.drop_duplicates(subset=list(domain_cols))

    # restrict output domain optionally so that only nuclei with truth are included
    if predict_domain == "truth":
        truth_domain = df_with_truth[list(domain_cols)].drop_duplicates(subset=list(domain_cols)).copy()
        tmp = pd.merge(truth_domain, pred, on=list(domain_cols), how="inner")
    else:
        # 'model' domain: whatever domain predict() returns (intersection of included predictors)
        tmp = pred.copy()

    if uncertainty_mode == "half_interval":
        tmp["u" + quantity] = (tmp[quantity + "_upper"] - tmp[quantity + "_lower"]) / 2.0
    else:
        tmp["u" + quantity] = (tmp[quantity + "_upper"] - tmp[quantity + "_lower"]) / (2.0 * 1.96)

    needed_cols = list(domain_cols) + [quantity, "u" + quantity]
    tmp = tmp.dropna(subset=needed_cols).sort_values(domain_cols).reset_index(drop=True)

    lower_vals = tmp[quantity + "_lower"].copy()
    upper_vals = tmp[quantity + "_upper"].copy()

    out_df = tmp[list(domain_cols) + [quantity, "u" + quantity]].copy()
    return out_df, lower_vals, upper_vals


def write_bmc_to_h5(
    h5_path,
    output_key,
    domain_cols,
    out_df,
    quantity,
    lower_vals,
    upper_vals,
    store_bounds,
    overwrite,
    merge,
):
    key_path = "/" + output_key

    with pd.HDFStore(h5_path, mode="a") as store:
        exists = key_path in store.keys()

        if exists and (not overwrite) and (not merge):
            raise ValueError(
                "Key '{0}' already exists. Use --overwrite to replace it or --merge to update columns.".format(output_key)
            )

        df_to_write = out_df.copy()

        if store_bounds:
            df_to_write[quantity + "_lower"] = np.asarray(lower_vals.values, dtype=float)
            df_to_write[quantity + "_upper"] = np.asarray(upper_vals.values, dtype=float)

        df_to_write = df_to_write.sort_values(list(domain_cols)).reset_index(drop=True)

        if exists and merge:
            old = store[output_key].copy()

            for col in domain_cols:
                if col not in old.columns:
                    raise ValueError("Existing '{0}' dataset missing domain column '{1}'.".format(output_key, col))

            merged_df = pd.merge(old, df_to_write, on=list(domain_cols), how="outer", suffixes=("", "_new"))

            cols_to_update = [quantity, "u" + quantity]
            if store_bounds:
                cols_to_update.extend([quantity + "_lower", quantity + "_upper"])

            for col in cols_to_update:
                new_col = col + "_new"
                if new_col in merged_df.columns:
                    merged_df[col] = merged_df[new_col]
                    merged_df = merged_df.drop(columns=[new_col])

            drop_cols = [c for c in merged_df.columns if c.endswith("_new")]
            if len(drop_cols) > 0:
                merged_df = merged_df.drop(columns=drop_cols)

            merged_df = merged_df.sort_values(list(domain_cols)).reset_index(drop=True)
            store.put(output_key, merged_df, format="fixed")
        else:
            store.put(output_key, df_to_write, format="fixed")


def upsert_bmc_meta(h5_path, meta_key, meta_row):
    """
    Maintain a small table /BMC_META with ONE row per quantity.
    Upsert rule: match on 'quantity' and overwrite that row.
    """
    key_path = "/" + meta_key
    new_df = pd.DataFrame([meta_row])

    with pd.HDFStore(h5_path, mode="a") as store:
        if key_path not in store.keys():
            store.put(meta_key, new_df, format="fixed")
            return

        old = store[meta_key].copy()
        if "quantity" not in old.columns:
            store.put(meta_key, new_df, format="fixed")
            return

        old = old[old["quantity"] != meta_row["quantity"]].copy()
        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.sort_values(["quantity"]).reset_index(drop=True)
        store.put(meta_key, combined, format="fixed")


def main():
    args = parse_args()

    if not os.path.exists(args.h5):
        raise FileNotFoundError("HDF5 file not found: {0}".format(args.h5))

    if args.overwrite and args.merge:
        raise ValueError("Use only one of --overwrite or --merge, not both.")

    truth = args.truth
    domain_cols = [c.strip() for c in args.domain_cols]
    validate_sizes(args.train_size, args.val_size, args.test_size)

    if args.quantity.upper() == "ALL":
        ame_cols = list_ame_columns(args.h5, truth)
        quantities = [c for c in ame_cols if is_bmc_target_quantity(c, domain_cols)]
    else:
        quantities = [args.quantity]

    candidate_models = [m for m in args.models if m != truth]

    if args.overwrite:
        with pd.HDFStore(args.h5, mode="a") as store:
            for key in [args.output_key, args.meta_key]:
                if "/" + key in store.keys():
                    del store[key]
        sys.stdout.write(
            "[BMC] Overwrite enabled: removed existing '/{0}' and '/{1}' datasets.\n"
            .format(args.output_key, args.meta_key)
        )


    total_done = 0
    total_skipped = 0

    for q in quantities:
        sys.stdout.write("\n=== BMC: quantity '{0}' ===\n".format(q))

        per_model_overlap = {}
        per_model_non_nan = {}
        excluded = {}

        models_for_q = []
        for m in candidate_models:
            exists, has_col, non_nan, overlap, reason = model_column_stats(
                args.h5, m, truth, q, domain_cols
            )

            if exists and has_col:
                per_model_non_nan[m] = int(non_nan)
                per_model_overlap[m] = int(overlap)

            if (not exists) or (not has_col):
                excluded[m] = reason if reason else "missing"
                continue
            if non_nan == 0:
                excluded[m] = "all_nan"
                continue
            if overlap < args.min_overlap:
                excluded[m] = "low_overlap_with_truth"
                continue

            models_for_q.append(m)

        if len(models_for_q) == 0:
            msg = "no predictor models passed (require column + non_nan>0 + overlap_with_truth>=min_overlap)"
            sys.stdout.write("[Skipped] Quantity '{0}': {1}\n".format(q, msg))
            total_skipped += 1

            meta_row = {
                "quantity": q,
                "truth": truth,
                "predictors_json": json.dumps([]),
                "n_models": 0,
                "overlap_all_models": 0,
                "per_model_overlap_json": json.dumps(per_model_overlap),
                "per_model_non_nan_json": json.dumps(per_model_non_nan),
                "excluded_models_json": json.dumps(excluded),
                "components": int(args.components),
                "iterations": int(args.iterations),
                "train_size": float(args.train_size),
                "val_size": float(args.val_size),
                "test_size": float(args.test_size),
                "uncertainty_mode": args.uncertainty_mode,
                "store_bounds": bool(args.store_bounds),
                "predict_domain": args.predict_domain,  # ONLY CHANGE: stored in metadata
                "status": "skipped",
                "message": msg,
            }
            upsert_bmc_meta(args.h5, args.meta_key, meta_row)
            continue

        overlap_all = compute_overlap_all_models(args.h5, truth, models_for_q, q, domain_cols)

        sys.stdout.write("Models kept ({0}): {1}\n".format(len(models_for_q), ", ".join(models_for_q)))
        sys.stdout.write("Overlap nuclei (truth + all kept models): {0}\n".format(overlap_all))
        sys.stdout.flush()

        if overlap_all < args.min_overlap:
            msg = "overlap_all_models {0} < min_overlap {1}".format(overlap_all, args.min_overlap)
            sys.stdout.write("[Skipped] Quantity '{0}': {1}\n".format(q, msg))
            total_skipped += 1

            meta_row = {
                "quantity": q,
                "truth": truth,
                "predictors_json": json.dumps(models_for_q),
                "n_models": int(len(models_for_q)),
                "overlap_all_models": int(overlap_all),
                "per_model_overlap_json": json.dumps(per_model_overlap),
                "per_model_non_nan_json": json.dumps(per_model_non_nan),
                "excluded_models_json": json.dumps(excluded),
                "components": int(args.components),
                "iterations": int(args.iterations),
                "train_size": float(args.train_size),
                "val_size": float(args.val_size),
                "test_size": float(args.test_size),
                "uncertainty_mode": args.uncertainty_mode,
                "store_bounds": bool(args.store_bounds),
                "predict_domain": args.predict_domain,  # ONLY CHANGE: stored in metadata
                "status": "skipped",
                "message": msg,
            }
            upsert_bmc_meta(args.h5, args.meta_key, meta_row)
            continue

        try:
            out_df, lower_vals, upper_vals = build_bmc_for_quantity(
                h5_path=args.h5,
                quantity=q,
                predictors=models_for_q,
                truth=truth,
                domain_cols=domain_cols,
                components=args.components,
                iterations=args.iterations,
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                uncertainty_mode=args.uncertainty_mode,
                min_overlap=args.min_overlap,
                predict_domain=args.predict_domain,  # ONLY CHANGE: passed through
            )

            write_bmc_to_h5(
                h5_path=args.h5,
                output_key=args.output_key,
                domain_cols=domain_cols,
                out_df=out_df,
                quantity=q,
                lower_vals=lower_vals,
                upper_vals=upper_vals,
                store_bounds=bool(args.store_bounds),
                overwrite=False,
                merge=True,
            )

            sys.stdout.write("[Done] '{0}' -> wrote {1} rows to '/{2}'\n".format(q, len(out_df), args.output_key))
            sys.stdout.flush()
            total_done += 1

            meta_row = {
                "quantity": q,
                "truth": truth,
                "predictors_json": json.dumps(models_for_q),
                "n_models": int(len(models_for_q)),
                "overlap_all_models": int(overlap_all),
                "per_model_overlap_json": json.dumps(per_model_overlap),
                "per_model_non_nan_json": json.dumps(per_model_non_nan),
                "excluded_models_json": json.dumps(excluded),
                "components": int(args.components),
                "iterations": int(args.iterations),
                "train_size": float(args.train_size),
                "val_size": float(args.val_size),
                "test_size": float(args.test_size),
                "uncertainty_mode": args.uncertainty_mode,
                "store_bounds": bool(args.store_bounds),
                "predict_domain": args.predict_domain,  # ONLY CHANGE: stored in metadata
                "status": "done",
                "message": "",
            }
            upsert_bmc_meta(args.h5, args.meta_key, meta_row)

        except Exception as exc:
            msg = str(exc)
            sys.stdout.write("[Failed] Quantity '{0}': {1}\n".format(q, msg))
            sys.stdout.flush()
            total_skipped += 1

            meta_row = {
                "quantity": q,
                "truth": truth,
                "predictors_json": json.dumps(models_for_q),
                "n_models": int(len(models_for_q)),
                "overlap_all_models": int(overlap_all),
                "per_model_overlap_json": json.dumps(per_model_overlap),
                "per_model_non_nan_json": json.dumps(per_model_non_nan),
                "excluded_models_json": json.dumps(excluded),
                "components": int(args.components),
                "iterations": int(args.iterations),
                "train_size": float(args.train_size),
                "val_size": float(args.val_size),
                "test_size": float(args.test_size),
                "uncertainty_mode": args.uncertainty_mode,
                "store_bounds": bool(args.store_bounds),
                "predict_domain": args.predict_domain,  # ONLY CHANGE: stored in metadata
                "status": "failed",
                "message": msg,
            }
            upsert_bmc_meta(args.h5, args.meta_key, meta_row)
            continue

    sys.stdout.write(
        "\n[BMC] Finished. Done: {0}, Skipped/Failed: {1}. Output key: '/{2}', Meta key: '/{3}'\n".format(
            total_done, total_skipped, args.output_key, args.meta_key
        )
    )


if __name__ == "__main__":
    main()


