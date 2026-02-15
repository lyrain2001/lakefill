"""
Complete LakeFill Baseline Runner - Uses THEIR Actual Code

This script:
1. Converts your benchmark to LakeFill's format
2. Runs THEIR retrieval code (evaluate.py)
3. Runs THEIR imputation code (impute.py)
4. Converts results back to your benchmark format

Datalake: up to 1000 cases (or as many as exist) are used as the collection for
retrieval; the row with [MISSING] in each case is excluded. Queries/evaluation
use the first `--variant` cases (e.g. 200) that have a missing value.

Usage:
    python lakefill/run_lakefill_baseline.py --dataset EnterpriseExcel --variant 200 --lakefill_repo ./lakefill --checkpoint_dir ./lakefill_checkpoints --use_azure
"""

import argparse
import json
import jsonlines
import os
import random
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
from tqdm import tqdm

# Model -> Azure region keys in llm-autofill/clients.yaml (for --use_azure auto-select)
AZURE_REGIONS_BY_MODEL = {
    "gpt-4o-mini": ["chat_completion_mini", "chat_completion_mini_2"],
    "gpt-4o": ["chat_completion", "chat_completion_2", "chat_completion_3"],
    "gpt-4": ["chat_completion", "chat_completion_2", "chat_completion_3"],
}

# Import your datasets module (llm-autofill is sibling of lakefill)
_project_root = Path(__file__).resolve().parent.parent
_llm_autofill = _project_root / "llm-autofill"
if _llm_autofill.exists():
    sys.path.insert(0, str(_llm_autofill))
from autofill.data.datasets import get_path_and_cases, DATASETS

# Max cases to use as the datalake (collection) for retrieval; queries are a subset.
LAKE_VARIANT_DEFAULT = "1000"
LAKE_MAX_CASES = 1000


def get_path_and_cases_for_lake(
    dataset_name: str,
    max_lake_cases: int = LAKE_MAX_CASES,
    seed: int = 42,
) -> Tuple[Path, List[str]]:
    """
    Get dataset path and case names to use as the datalake (up to max_lake_cases).
    Prefers variant 1000; falls back to 200 then 300 if path missing. Uses as many
    cases as exist (no error if fewer than max_lake_cases).
    """
    for variant in (LAKE_VARIANT_DEFAULT, "200", "300"):
        try:
            path, cases = get_path_and_cases(dataset_name, variant=variant, seed=seed)
            use = cases[:max_lake_cases]
            return path, use
        except (FileNotFoundError, KeyError):
            continue
    # Last resort: benchmark or first variant path that exists
    if dataset_name not in DATASETS:
        raise KeyError(f"Dataset '{dataset_name}' not found in registry.")
    ds = DATASETS[dataset_name]
    path = ds.benchmark
    if path and path.exists():
        cases = sorted([p.name for p in path.iterdir() if p.is_dir()])
        return path, cases[:max_lake_cases]
    for v in ("200", "1000", "300", "100"):
        if v in ds.variants and ds.variants[v].exists():
            cases = sorted([p.name for p in ds.variants[v].iterdir() if p.is_dir()])
            return ds.variants[v], cases[:max_lake_cases]
    raise FileNotFoundError(f"No path found for dataset '{dataset_name}' (tried 1000, 200, 300 and benchmark/variants).")


def load_case(case_dir: Path) -> Tuple[pd.DataFrame, str, str, int]:
    """Load a case with data.csv and info.json."""
    csv_path = case_dir / "data.csv"
    info_path = case_dir / "info.json"
    
    df = pd.read_csv(csv_path)
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # Find the [MISSING] cell
    missing_row_idx = None
    missing_col = None
    for col in df.columns:
        mask = df[col].astype(str).str.contains(r'\[MISSING\]', na=False, regex=True)
        if mask.any():
            missing_row_idx = mask.idxmax()
            missing_col = col
            break
    
    ground_truth = info.get('label', '').strip()
    
    return df, missing_col, ground_truth, missing_row_idx


def _sanitize_tsv_value(s: str) -> str:
    """Replace newlines and tabs so one tuple = one line for TSV parsing by retriever."""
    return s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()


def row_to_lakefill_tuple(row: pd.Series, columns: List[str], table_id: int) -> str:
    """
    Convert a dataframe row to LakeFill's specific tuple format:
    [table_id]: table_name attribute col1 value val1 attribute col2 value val2 ...
    Values and column names are sanitized (no newlines/tabs) so one record = one line in collection/queries TSV.
    """
    parts = [f"[{table_id}]: tuple"]
    
    for col in columns:
        col_safe = _sanitize_tsv_value(str(col))
        val = row[col]
        if pd.isna(val):
            val = "N/A"
        elif str(val) == '[MISSING]':
            val = "N/A"
        else:
            val = _sanitize_tsv_value(str(val))
        
        parts.append(f"attribute {col_safe} value {val}")
    
    line = " ".join(parts)
    # Final safety: ensure no newline/tab in output (retriever expects one tab per line)
    return _sanitize_tsv_value(line) or " "


def convert_to_lakefill_format(
    dataset_name: str,
    variant: str,
    output_dir: Path,
    seed: int = 42
):
    """
    Convert benchmark to LakeFill format.
    Test set: exactly the `variant` cases (e.g. 200) from get_path_and_cases(..., variant, seed).
    Datalake (collection): those same cases plus more from the same path, up to
    LAKE_MAX_CASES (1000); missing-value row excluded per case.
    Their expected files:
    - queries.tsv: qid<tab>serialized_tuple_with_N/A
    - collection.tsv: docid<tab>serialized_tuple_complete
    - qrels.tsv: qid<tab>docid<tab>score (relevance - we set to 1 for same case)
    - folds.json: {"train": [], "test": [qids]}
    - answers.jsonl: [{"query_id": qid, "answers": {col: value}}]
    """
    # Test set: exactly the variant cases from get_path_and_cases (same 200 as dataset code).
    try:
        root, query_cases = get_path_and_cases(dataset_name, variant=variant, seed=seed)
    except (FileNotFoundError, KeyError):
        root, query_cases = get_path_and_cases_for_lake(
            dataset_name, max_lake_cases=LAKE_MAX_CASES, seed=seed
        )
        query_cases = query_cases[: min(int(variant), len(query_cases))]

    # Datalake: query_cases first (so test set is always in the lake), then fill up to 1000 from same path
    all_in_path = sorted([p.name for p in root.iterdir() if p.is_dir()])
    extra = [c for c in all_in_path if c not in query_cases]
    lake_cases = list(query_cases) + extra[: LAKE_MAX_CASES - len(query_cases)]

    print(f"Dataset: {dataset_name}")
    print(f"Variant (test set size): {variant} → {len(query_cases)} cases")
    print(f"Datalake cases: {len(lake_cases)} (max {LAKE_MAX_CASES})")
    print(f"Root: {root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    queries_file = output_dir / "queries.tsv"
    collection_file = output_dir / "collection.tsv"
    qrels_file = output_dir / "qrels.tsv"
    folds_file = output_dir / "folds.json"
    answers_file = output_dir / "answers.jsonl"

    # Load all lake cases (for collection)
    lake_case_data = []
    for case_name in tqdm(lake_cases, desc="Loading cases"):
        case_dir = root / case_name
        try:
            df, missing_col, ground_truth, missing_row_idx = load_case(case_dir)
            lake_case_data.append({
                "case_name": case_name,
                "df": df,
                "missing_col": missing_col,
                "ground_truth": ground_truth,
                "missing_row_idx": missing_row_idx,
            })
        except Exception as e:
            print(f"Warning: Error loading {case_name}: {e}")
            continue

    by_name = {c["case_name"]: c for c in lake_case_data}
    # Query set = exactly the variant cases (same 200 as before), in order; skip if no missing
    query_case_data = []
    for case_name in query_cases:
        if case_name not in by_name:
            continue
        c = by_name[case_name]
        if c["missing_col"] is not None:
            query_case_data.append(c)
    if len(query_case_data) < len(query_cases):
        print(f"Note: {len(query_case_data)} of {len(query_cases)} variant cases have missing values")

    print(f"Loaded {len(lake_case_data)} lake cases; {len(query_case_data)} query (test) cases")

    # Build queries (incomplete tuples with N/A) from query_case_data only
    queries = []
    test_qids = []
    for qid, case in enumerate(query_case_data):
        query_row = case["df"].iloc[case["missing_row_idx"]]
        query_text = row_to_lakefill_tuple(query_row, list(case["df"].columns), qid)
        queries.append((qid, query_text))
        test_qids.append(qid)

    # Build collection from ALL lake cases.
    # - Query cases (the 200 we test on): exclude the missing row so the query row is never in the lake (no leakage).
    # - Non-query cases (extra lake cases): include the missing row after replacing [MISSING] with that case's
    #   ground truth so the lake has complete tuples only.
    query_case_names = set(query_cases)
    collection = []
    doc_id_counter = 0
    for case in lake_case_data:
        df = case["df"]
        missing_row_idx = case["missing_row_idx"]
        missing_col = case["missing_col"]
        ground_truth = case["ground_truth"]
        is_query_case = case["case_name"] in query_case_names
        for row_idx in range(len(df)):
            if row_idx == missing_row_idx:
                if is_query_case:
                    continue  # never add query row to collection
                # Non-query case: add row with [MISSING] replaced by this case's ground truth
                row = df.iloc[row_idx].copy()
                if missing_col is not None and ground_truth is not None:
                    row[missing_col] = ground_truth
                tuple_text = row_to_lakefill_tuple(row, list(df.columns), doc_id_counter)
                collection.append((doc_id_counter, tuple_text))
                doc_id_counter += 1
                continue
            row = df.iloc[row_idx]
            tuple_text = row_to_lakefill_tuple(row, list(df.columns), doc_id_counter)
            collection.append((doc_id_counter, tuple_text))
            doc_id_counter += 1

    # Qrels: all query-doc pairs (all docs potentially relevant)
    qrels = []
    for qid in range(len(query_case_data)):
        for docid in range(len(collection)):
            qrels.append((qid, docid, 1))
    
    # Write queries.tsv
    print(f"Writing {len(queries)} queries to {queries_file}")
    with open(queries_file, 'w', encoding='utf-8') as f:
        for qid, qtext in queries:
            f.write(f"{qid}\t{qtext}\n")
    
    # Write collection.tsv
    print(f"Writing {len(collection)} tuples to {collection_file}")
    with open(collection_file, 'w', encoding='utf-8') as f:
        for docid, ttext in collection:
            f.write(f"{docid}\t{ttext}\n")
    
    # Write qrels.tsv
    print(f"Writing {len(qrels)} relevance judgments to {qrels_file}")
    with open(qrels_file, 'w', encoding='utf-8') as f:
        for qid, docid, score in qrels:
            f.write(f"{qid}\t{docid}\t{score}\n")
    
    # Write folds.json
    print(f"Writing folds to {folds_file}")
    with open(folds_file, 'w') as f:
        json.dump({
            "train": [],  # We don't have training data
            "test": test_qids
        }, f, indent=2)
    
    # Write answers.jsonl (only for query cases)
    print(f"Writing answers to {answers_file}")
    with open(answers_file, 'w', encoding='utf-8') as f:
        for qid, case in enumerate(query_case_data):
            answer = {
                "query_id": qid,
                "answers": {case["missing_col"]: case["ground_truth"]},
                "case_name": case["case_name"],
            }
            f.write(json.dumps(answer) + '\n')

    print(f"\nConversion complete!")
    print(f"  Output: {output_dir}")
    
    return queries_file, collection_file, qrels_file, folds_file, answers_file


def run_lakefill_retrieval(
    lakefill_repo: Path,
    data_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    dataset_name: str
):
    """
    Run LakeFill's retrieval code (their evaluate.py).
    """
    print("\n" + "="*60)
    print("STEP 2: Running LakeFill's Retrieval")
    print("="*60)
    
    retriever_dir = lakefill_repo / "retriever"
    output_file = output_dir / "retrieval_results.tsv"
    index_dir = output_dir / "index"
    
    # Clean up any existing index files to avoid SQL/FAISS mismatch
    # FAISSDocumentStore creates both .faiss file and SQL database
    if index_dir.exists():
        import shutil
        print(f"Cleaning existing index directory: {index_dir}")
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command to run their evaluate.py
    # Use relative path since cwd is set to retriever_dir
    cmd = [
        "python", "evaluate.py",
        "--model_name", "siamese",
        "--dataset_name", dataset_name,
        "--save_model_dir", str(checkpoint_dir.resolve()),
        "--default_path", "bert-base-uncased",  # They use BERT base
        "--temp_index_path", str(output_dir.resolve() / "index"),
        "--data_path", str(data_dir.resolve()),
        "--num_retrieved", "100"
    ]
    
    print(f"Running command:")
    print(" ".join(cmd))
    print(f"Working directory: {retriever_dir}")
    
    # Run their retrieval
    try:
        result = subprocess.run(
            cmd,
            cwd=str(retriever_dir),
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running retrieval: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise
    
    # Their evaluate.py writes to ./retrieval_results/
    # Move it to our output directory
    their_output = retriever_dir / "retrieval_results" / f"Siamese_{dataset_name}_top100_res_with_score.tsv"
    if their_output.exists():
        import shutil
        shutil.copy(their_output, output_file)
        print(f"Retrieval results saved to: {output_file}")
    else:
        raise FileNotFoundError(f"Expected output file not found: {their_output}")
    
    return output_file


def _check_entra_token() -> None:
    """Verify Entra ID (DefaultAzureCredential) can get a token; exit with clear message if not."""
    try:
        from azure.identity import DefaultAzureCredential
        cred = DefaultAzureCredential()
        cred.get_token("https://cognitiveservices.azure.com/.default")
    except Exception as e:
        msg = str(e).split("\n")[0] if "\n" in str(e) else str(e)
        print("Entra ID token check failed:", msg[:200])
        print()
        print("To fix: run the following, then retry (token expires in ~12h):")
        print("  az login --scope https://cognitiveservices.azure.com/.default")
        print()
        print("Or use API key instead: add --azure_use_api_key to your command (if the resource allows key auth).")
        sys.exit(1)


def run_lakefill_imputation(
    lakefill_repo: Path,
    data_dir: Path,
    retrieval_results: Path,
    output_dir: Path,
    dataset_name: str,
    variant: str,
    results_dir: Path,
    model: str = "gpt-4",
    top_k: int = 10,
    threshold: float = 0.8,
    num_threads: int = 4,
    openai_key: str = None,
    gpt_region: str = None,
    clients_yaml: Path = None,
    azure_use_entra_id: bool | None = False,
):
    """
    Run LakeFill's imputation code (impute.py).
    Azure: same as llm-autofill get_gpt_client — use region's use_entra_id from clients.yaml;
    or pass True/False to force Entra vs api_key. Entra uses same pattern as gpt.py (get_bearer_token_provider + DefaultAzureCredential).
    """
    print("\n" + "="*60)
    print("STEP 3: Running LakeFill's Imputation")
    print("="*60)
    
    imputation_dir = lakefill_repo / "imputation"
    model_safe = model.replace("/", "-")
    # Results saved under results_dir with names: results_${dataset}_lakefill_${model}_th${threshold}_stats.csv and _lakesize${LAKE_VARIANT_DEFAULT}.jsonl
    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"results_{dataset_name}_lakefill_{model_safe}_th{threshold}_lakesize{LAKE_VARIANT_DEFAULT}.jsonl"
    stats_csv = results_dir / f"results_{dataset_name}_lakefill_{model_safe}_th{threshold}_lakesize{LAKE_VARIANT_DEFAULT}_stats.csv"
    usage_csv_path = results_dir / f"results_{dataset_name}_lakefill_{model_safe}_th{threshold}_lakesize{LAKE_VARIANT_DEFAULT}_usage.csv"
    
    use_azure = bool(gpt_region and clients_yaml and clients_yaml.exists())
    if use_azure:
        import yaml
        with open(clients_yaml, "r") as f:
            config = yaml.safe_load(f)
        if "gpt" not in config or gpt_region not in config["gpt"]:
            raise ValueError(f"Region '{gpt_region}' not in {clients_yaml} under 'gpt'")
        client_info = config["gpt"][gpt_region]
        use_entra = True
        if use_entra:
            _check_entra_token()
        azure_endpoint = client_info["azure_endpoint"]  # keep as in clients.yaml (match autofill gpt.py)
        api_version = client_info["api_version"]
        cmd = [
            "python", "impute.py",
            "--model", model,
            "--azure_endpoint", azure_endpoint,
            "--api_version", api_version,
            "--data_path", str(data_dir.resolve()),
            "--retrieval_results_path", str(retrieval_results.resolve()),
            "--output_path", str(output_file.resolve()),
            "--dataset", dataset_name,
            "--top_k", str(top_k),
            "--threshold", str(threshold),
            "--num_threads", str(num_threads),
            "--stats_csv_path", str(stats_csv.resolve()),
            "--usage_csv_path", str(usage_csv_path.resolve()),
            "--max_prompt_chars", "128000",
        ]
        if use_entra:
            cmd.append("--azure_use_entra_id")
            print("Azure auth: Entra ID (DefaultAzureCredential)")
        else:
            cmd.extend(["--api_key", client_info["api_key"]])
            print("Azure auth: API key from clients.yaml")
    else:
        if not openai_key:
            raise ValueError("Provide --gpt_region + --clients_yaml (Azure) or set OPENAI_API_KEY (OpenAI)")
        cmd = [
            "python", "impute.py",
            "--model", model,
            "--api_url", "https://api.openai.com/v1",
            "--api_key", openai_key,
            "--data_path", str(data_dir.resolve()),
            "--retrieval_results_path", str(retrieval_results.resolve()),
            "--output_path", str(output_file.resolve()),
            "--dataset", dataset_name,
            "--top_k", str(top_k),
            "--threshold", str(threshold),
            "--num_threads", str(num_threads),
            "--stats_csv_path", str(stats_csv.resolve()),
            "--usage_csv_path", str(usage_csv_path.resolve()),
            "--max_prompt_chars", "128000",
        ]
    
    print(f"Running command:")
    safe_cmd = cmd
    if "--api_key" in cmd:
        i = cmd.index("--api_key")
        if i + 1 < len(cmd):
            safe_cmd = cmd[: i + 1] + ["***"] + cmd[i + 2 :]
    print(" ".join(safe_cmd))
    print(f"Working directory: {imputation_dir}")
    
    # Run their imputation (do not capture output so progress bar and logs stream in real time)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(imputation_dir),
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running imputation (exit code {e.returncode})")
        raise
    
    print(f"Imputation results saved to: {output_file}")
    print(f"Stats CSV: {stats_csv}")
    print(f"Usage CSV: {usage_csv_path}")
    
    return output_file, stats_csv, usage_csv_path


def convert_results_to_benchmark_format(
    imputation_results: Path,
    answers_file: Path,
    output_file: Path
):
    """
    Convert LakeFill's results to benchmark stats format: case, acc, fail, output, ground_truth, conf, runtime.
    Reads full jsonl and writes CSV in case order (no token columns).
    """
    print("\n" + "="*60)
    print("STEP 4: Converting Results to Benchmark Format")
    print("="*60)
    
    # Load case names from answers (for ordering and names)
    case_names = {}
    with jsonlines.open(answers_file, 'r') as f:
        for line in f:
            case_names[line['query_id']] = line['case_name']
    
    # Load imputation results (one row per case)
    rows_by_qid = {}
    with jsonlines.open(imputation_results, 'r') as f:
        for line in f:
            qid = line['query_id']
            final_preds = line['final_predictions']
            final_confs = line.get('final_confidences', {})
            ground_truth = line['ground_truth']
            final_correctness = line.get('final_correctness', {})
            runtime = line.get('runtime', 0.0)
            case_name = case_names.get(qid, f'query_{qid}')
            if not ground_truth:
                acc, fail, output, gt_str, conf = 0, 1.0, '', '', 0.0
            else:
                cols = list(ground_truth.keys())
                correct_list = [final_correctness.get(c, 0) for c in cols]
                acc = 1 if all(correct_list) and correct_list else 0
                fail = 0.0 if acc == 1 else 1.0
                output = final_preds.get(cols[0], '[UNKNOWN]') if cols else '[UNKNOWN]'
                gt_str = ground_truth.get(cols[0], '') if cols else ''
                confs = [final_confs.get(c, 0.0) for c in cols if c in final_confs]
                conf = (sum(confs) / len(confs)) if confs else 0.0
            rows_by_qid[qid] = {
                'case': case_name,
                'acc': acc,
                'fail': fail,
                'output': output,
                'ground_truth': gt_str,
                'conf': conf,
                'runtime': runtime,
            }
    
    # Sort by query_id so output is in case order
    qids_sorted = sorted(rows_by_qid.keys())
    stats = [rows_by_qid[q] for q in qids_sorted]
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_file, index=False)
    
    accuracy = stats_df['acc'].mean()
    print(f"\nResults Summary:")
    print(f"  Total cases: {len(stats_df)}")
    print(f"  Correct: {stats_df['acc'].sum()}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"\nResults saved to: {output_file}")
    return output_file


def _collapse_usage_csv_by_case(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse usage rows to one per case by summing cost/token metrics over strict and relaxed modes."""
    if df.empty:
        return df
    sum_cols = [
        "prompt_tokens", "completion_tokens", "total_tokens",
        "input_cost", "output_cost", "total_cost",
        "retries", "latency_ms",
    ]
    sum_cols = [c for c in sum_cols if c in df.columns]
    agg = {c: "sum" for c in sum_cols}
    # Keep first mode/model per case; if multiple rows, mark mode as "strict+relaxed"
    if "mode" in df.columns:
        agg["mode"] = lambda x: "strict+relaxed" if len(x) > 1 else x.iloc[0]
    if "model" in df.columns:
        agg["model"] = "first"
    return df.groupby("case", as_index=False).agg(agg)


def _update_usage_csv_summary(usage_fp: Path) -> None:
    """Collapse to one row per case (sum strict+relaxed), then append a mean row to usage CSV."""
    if not usage_fp.exists():
        return
    df = pd.read_csv(usage_fp, dtype={"case": str})
    df = df[df["case"] != "mean"]
    if df.empty:
        return
    # Collapse: one row per case, summed over strict/relaxed
    collapsed = _collapse_usage_csv_by_case(df)
    if collapsed.empty:
        return
    num_cols = collapsed.select_dtypes(include="number").columns
    mean_row = {col: (collapsed[col].mean() if col in num_cols else "") for col in collapsed.columns}
    mean_row["case"] = "mean"
    out = pd.concat([collapsed, pd.DataFrame([mean_row])], ignore_index=True)
    out.to_csv(usage_fp, index=False)


def _resolve_azure_entra(args) -> bool | None:
    """
    When --use_azure: use region's use_entra_id from clients.yaml (same as get_gpt_client).
    Return True to force Entra, False to force api_key, None to use region default from clients.yaml.
    """
    if getattr(args, "azure_use_entra_id", False):
        return True
    if getattr(args, "azure_use_api_key", False):
        return False
    return None  # use region default (client_info.get("use_entra_id", False) in run_lakefill_imputation)


def main():
    parser = argparse.ArgumentParser(description="Run LakeFill baseline using their code")
    
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g., AR_semantics). Not required if --collapse_usage_only.")
    parser.add_argument("--variant", type=str, default="200",
                        help="Variant: 200, 100, 300, or 1000")
    parser.add_argument("--lakefill_repo", type=str, default=".",
                        help="Path to LakeFill repository (lakefill). Not required if --collapse_usage_only.")
    parser.add_argument("--checkpoint_dir", type=str, default=".",
                        help="Path to LakeFill's pretrained retriever checkpoint. Not required if --collapse_usage_only.")
    parser.add_argument("--output_dir", type=str, default="./lakefill_baseline",
                        help="Output directory for intermediate data (format conversion, retrieval)")
    parser.add_argument("--results_dir", type=str, default="/datadrive/yurong/lakefill/results",
                        help="Directory for final result files (stats CSV, jsonl, usage CSV)")
    # Paper baseline defaults (aligned with lakefill/imputation/impute.py)
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="LLM model for imputation (baseline default: gpt-4)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of tuples to use for imputation (baseline default: 10)")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Confidence threshold for strict mode (baseline default: 0.8)")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Number of threads for parallel imputation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use_azure", action="store_true",
                        help="Use Azure OpenAI; load endpoint from llm-autofill/clients.yaml and auto-pick region by model.")
    parser.add_argument("--gpt_region", type=str, default=None,
                        help="Azure region key in clients.yaml (e.g. chat_completion_mini). Optional when --use_azure; overrides auto-selected region.")
    parser.add_argument("--clients_yaml", type=str, default=None,
                        help="Path to clients.yaml for Azure (default: <project_root>/llm-autofill/clients.yaml)")
    parser.add_argument("--azure_use_entra_id", action="store_true",
                        help="Use Azure Entra ID (same as get_gpt_ft_client). Default: use region's use_entra_id from clients.yaml.")
    parser.add_argument("--azure_use_api_key", action="store_true",
                        help="Use api_key from clients.yaml (overrides region default)")
    parser.add_argument("--collapse_usage_only", action="store_true",
                        help="Only collapse usage CSVs in results_dir to one row per case (sum strict+relaxed), then exit.")
    
    args = parser.parse_args()
    
    # Optional: just collapse all *_usage.csv in results_dir (and subdirs like 200/)
    if getattr(args, "collapse_usage_only", False):
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Results dir not found: {results_dir}")
            return
        usage_files = list(results_dir.glob("*_usage.csv")) + list(results_dir.glob("*/*_usage.csv"))
        for fp in sorted(usage_files):
            _update_usage_csv_summary(fp)
            print(f"Collapsed: {fp}")
        print(f"Done. Collapsed {len(usage_files)} usage CSV(s).")
        return
    
    if not args.dataset:
        parser.error("--dataset is required unless --collapse_usage_only")
    
    lakefill_repo = Path(args.lakefill_repo)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) / args.dataset / args.variant
    data_dir = output_dir / "data"

    # Resolve Azure config when --use_azure or --gpt_region is set
    default_clients_yaml = _project_root / "llm-autofill" / "clients.yaml"
    use_azure = getattr(args, "use_azure", False) or bool(getattr(args, "gpt_region", None))
    if use_azure:
        args.clients_yaml = Path(args.clients_yaml) if args.clients_yaml else default_clients_yaml
        if not args.clients_yaml.exists():
            print(f"Error: clients.yaml not found at {args.clients_yaml}")
            return
        if not args.gpt_region:
            # Auto-pick region from model using AZURE_REGIONS_BY_MODEL
            for model_key, regions in AZURE_REGIONS_BY_MODEL.items():
                if model_key in (args.model or "").lower():
                    args.gpt_region = random.choice(regions)
                    break
            if not args.gpt_region:
                # Fallback: use first region from config under "gpt"
                import yaml
                with open(args.clients_yaml, "r") as f:
                    config = yaml.safe_load(f)
                gpt_regions = list(config.get("gpt", {}).keys())
                if gpt_regions:
                    args.gpt_region = random.choice(gpt_regions)
            if args.gpt_region:
                print(f"Using Azure: region={args.gpt_region}, config={args.clients_yaml}")
            else:
                print("Error: No Azure region found for model and clients.yaml has no 'gpt' entries.")
                return
    else:
        args.gpt_region = None
        args.clients_yaml = None

    from dotenv import load_dotenv
    load_dotenv()
    args.openai_key = os.getenv("OPENAI_API_KEY")
    if not use_azure and not args.openai_key:
        print("Error: Set OPENAI_API_KEY or use --use_azure to use Azure endpoints from llm-autofill/clients.yaml.")
        return
    
    # Validate paths
    if not lakefill_repo.exists():
        print(f"Error: LakeFill repository not found at: {lakefill_repo}")
        return
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Define expected file paths
    queries_file = data_dir / "queries.tsv"
    collection_file = data_dir / "collection.tsv"
    qrels_file = data_dir / "qrels.tsv"
    folds_file = data_dir / "folds.json"
    answers_file = data_dir / "answers.jsonl"
    retrieval_results = output_dir / "retrieval_results.tsv"
    
    # Step 1: Convert format (skip if files exist)
    if all(f.exists() for f in [queries_file, collection_file, qrels_file, folds_file, answers_file]):
        print("="*60)
        print("STEP 1: SKIPPED - LakeFill format files already exist")
        print("="*60)
        print(f"  Using existing files in: {data_dir}")
    else:
        print("="*60)
        print("STEP 1: Converting Benchmark to LakeFill Format")
        print("="*60)
        queries_file, collection_file, qrels_file, folds_file, answers_file = convert_to_lakefill_format(
            args.dataset,
            args.variant,
            data_dir,
            args.seed
        )
    
    # Step 2: Run retrieval (skip if results exist)
    if retrieval_results.exists():
        print("\n" + "="*60)
        print("STEP 2: SKIPPED - Retrieval results already exist")
        print("="*60)
        print(f"  Using existing file: {retrieval_results}")
    else:
        retrieval_results = run_lakefill_retrieval(
            lakefill_repo,
            data_dir,
            checkpoint_dir,
            output_dir,
            args.dataset
        )
    
    # Step 3: Run imputation (writes jsonl + stats CSV + usage CSV in real time; skips already-processed cases on restart)
    results_dir = Path(getattr(args, "results_dir", "/datadrive/yurong/lakefill/results"))
    imputation_results, stats_csv, usage_csv_path = run_lakefill_imputation(
        lakefill_repo,
        data_dir,
        retrieval_results,
        output_dir,
        args.dataset,
        variant=args.variant,
        results_dir=results_dir,
        model=args.model,
        top_k=args.top_k,
        threshold=args.threshold,
        num_threads=args.num_threads,
        openai_key=args.openai_key,
        gpt_region=getattr(args, "gpt_region", None),
        clients_yaml=getattr(args, "clients_yaml", None),
        azure_use_entra_id=_resolve_azure_entra(args),
    )
    
    # Step 4: Regenerate stats CSV from jsonl in case order (case, acc, fail, output, ground_truth, conf, runtime)
    convert_results_to_benchmark_format(
        imputation_results,
        answers_file,
        stats_csv
    )
    
    # Step 5: Append mean row to usage CSV (benchmark-style summary)
    if usage_csv_path.exists():
        _update_usage_csv_summary(usage_csv_path)
    
    print("\n" + "="*60)
    print("ALL STEPS COMPLETE!")
    print("="*60)
    print(f"Final results: {stats_csv}, {imputation_results}, {usage_csv_path}")


if __name__ == "__main__":
    main()
