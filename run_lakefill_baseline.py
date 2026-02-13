"""
Complete LakeFill Baseline Runner - Uses THEIR Actual Code

This script:
1. Converts your benchmark to LakeFill's format
2. Runs THEIR retrieval code (evaluate.py)
3. Runs THEIR imputation code (impute.py)  
4. Converts results back to your benchmark format

Usage:
    python lakefill/run_lakefill_baseline.py --dataset AR_semantics --variant 200 --lakefill_repo ./lakefill --checkpoint_dir ./lakefill_checkpoints
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
from autofill.data.datasets import get_path_and_cases


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
    Their expected files:
    - queries.tsv: qid<tab>serialized_tuple_with_N/A
    - collection.tsv: docid<tab>serialized_tuple_complete
    - qrels.tsv: qid<tab>docid<tab>score (relevance - we set to 1 for same case)
    - folds.json: {"train": [], "test": [qids]}
    - answers.jsonl: [{"query_id": qid, "answers": {col: value}}]
    """
    
    # Get dataset path and cases
    root, cases = get_path_and_cases(dataset_name, variant=variant, seed=seed)
    
    print(f"Dataset: {dataset_name}")
    print(f"Variant: {variant}")
    print(f"Root: {root}")
    print(f"Cases: {len(cases)}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output files (matching their format)
    queries_file = output_dir / "queries.tsv"
    collection_file = output_dir / "collection.tsv"
    qrels_file = output_dir / "qrels.tsv"
    folds_file = output_dir / "folds.json"
    answers_file = output_dir / "answers.jsonl"
    
    # Load all cases
    all_case_data = []
    for case_name in tqdm(cases, desc="Loading cases"):
        case_dir = root / case_name
        try:
            df, missing_col, ground_truth, missing_row_idx = load_case(case_dir)
            
            if missing_col is None:
                print(f"Warning: No missing value in {case_name}, skipping")
                continue
            
            all_case_data.append({
                'case_name': case_name,
                'df': df,
                'missing_col': missing_col,
                'ground_truth': ground_truth,
                'missing_row_idx': missing_row_idx
            })
            
        except Exception as e:
            print(f"Error loading {case_name}: {e}")
            continue
    
    print(f"Loaded {len(all_case_data)} valid cases")
    
    # Build queries (incomplete tuples with N/A)
    queries = []
    qrels = []
    test_qids = []
    case_to_docid = {}  # Track which doc belongs to which case
    
    for query_idx, case in enumerate(all_case_data):
        qid = query_idx  # Use simple integer IDs
        query_row = case['df'].iloc[case['missing_row_idx']]
        query_text = row_to_lakefill_tuple(query_row, list(case['df'].columns), qid)
        queries.append((qid, query_text))
        test_qids.append(qid)
        case_to_docid[case['case_name']] = qid
    
    # Build collection (data lake with complete tuples)
    # CRITICAL DESIGN DECISION: What should be in the data lake?
    # 
    # Option 1: Include ALL rows from all cases (with [MISSING] kept as-is)
    #   - Pro: Realistic - the lake has incomplete data too
    #   - Con: The missing row itself is in the lake (though incomplete)
    #
    # Option 2: Exclude the specific missing row from each case
    #   - Pro: No leakage risk - the query row is not in the lake at all
    #   - Con: Less realistic - reduces lake size
    #
    # We choose Option 2 for strict no-leakage evaluation
    
    collection = []
    doc_id_counter = 0
    query_to_excluded_docs = {}  # Track which docs to exclude for each query
    
    for case_idx, case in enumerate(all_case_data):
        df = case['df']
        missing_row_idx = case['missing_row_idx']
        query_to_excluded_docs[case_idx] = []
        
        # Add ALL rows EXCEPT the missing row to the collection
        for row_idx in range(len(df)):
            if row_idx == missing_row_idx:
                # Skip the row with [MISSING] - it's the query, not part of the lake
                continue
                
            row = df.iloc[row_idx]
            tuple_text = row_to_lakefill_tuple(row, list(df.columns), doc_id_counter)
            collection.append((doc_id_counter, tuple_text))
            doc_id_counter += 1
    
    # Build qrels (relevance judgments)
    # Since we excluded the missing rows from collection, all docs in collection
    # are potentially relevant for all queries
    for qid in range(len(all_case_data)):
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
    
    # Write answers.jsonl
    print(f"Writing answers to {answers_file}")
    with open(answers_file, 'w', encoding='utf-8') as f:
        for qid, case in enumerate(all_case_data):
            answer = {
                "query_id": qid,
                "answers": {case['missing_col']: case['ground_truth']},
                "case_name": case['case_name']
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


def run_lakefill_imputation(
    lakefill_repo: Path,
    data_dir: Path,
    retrieval_results: Path,
    output_dir: Path,
    dataset_name: str,
    model: str = "gpt-4",
    top_k: int = 10,
    threshold: float = 0.8,
    num_threads: int = 4,
    openai_key: str = None,
    gpt_region: str = None,
    clients_yaml: Path = None,
):
    """
    Run LakeFill's imputation code (impute.py).
    Use Azure when gpt_region + clients_yaml are set; otherwise use openai_key (OpenAI API).
    """
    print("\n" + "="*60)
    print("STEP 3: Running LakeFill's Imputation")
    print("="*60)
    
    imputation_dir = lakefill_repo / "imputation"
    output_file = output_dir / f"imputation_results_top{top_k}_threshold{threshold}.jsonl"
    model_safe = model.replace("/", "-")
    stats_csv = output_dir / f"results_{dataset_name}_lakefill_{model_safe}_stats.csv"
    
    use_azure = bool(gpt_region and clients_yaml and clients_yaml.exists())
    if use_azure:
        import yaml
        with open(clients_yaml, "r") as f:
            config = yaml.safe_load(f)
        if "gpt" not in config or gpt_region not in config["gpt"]:
            raise ValueError(f"Region '{gpt_region}' not in {clients_yaml} under 'gpt'")
        client_info = config["gpt"][gpt_region]
        api_key = client_info["api_key"]
        azure_endpoint = client_info["azure_endpoint"].rstrip("/")
        api_version = client_info["api_version"]
        cmd = [
            "python", "impute.py",
            "--model", model,
            "--azure_endpoint", azure_endpoint,
            "--api_key", api_key,
            "--api_version", api_version,
            "--data_path", str(data_dir.resolve()),
            "--retrieval_results_path", str(retrieval_results.resolve()),
            "--output_path", str(output_file.resolve()),
            "--dataset", dataset_name,
            "--top_k", str(top_k),
            "--threshold", str(threshold),
            "--num_threads", str(num_threads),
            "--stats_csv_path", str(stats_csv.resolve()),
        ]
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
        ]
    
    print(f"Running command:")
    print(" ".join(cmd))
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
    print(f"Stats CSV (real-time): {stats_csv}")
    
    return output_file, stats_csv


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


def main():
    parser = argparse.ArgumentParser(description="Run LakeFill baseline using their code")
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., AR_semantics)")
    parser.add_argument("--variant", type=str, default="200",
                        help="Variant: 200, 100, 300, or 1000")
    parser.add_argument("--lakefill_repo", type=str, required=True,
                        help="Path to LakeFill repository (lakefill)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to LakeFill's pretrained retriever checkpoint")
    parser.add_argument("--output_dir", type=str, default="./lakefill_baseline",
                        help="Output directory")
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
    
    args = parser.parse_args()
    
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
    
    # Step 3: Run imputation (writes jsonl + stats CSV in real time; skips already-processed cases on restart)
    imputation_results, stats_csv = run_lakefill_imputation(
        lakefill_repo,
        data_dir,
        retrieval_results,
        output_dir,
        args.dataset,
        model=args.model,
        top_k=args.top_k,
        threshold=args.threshold,
        num_threads=args.num_threads,
        openai_key=args.openai_key,
        gpt_region=getattr(args, "gpt_region", None),
        clients_yaml=getattr(args, "clients_yaml", None),
    )
    
    # Step 4: Regenerate stats CSV from jsonl in case order (case, acc, fail, output, ground_truth, conf, runtime)
    convert_results_to_benchmark_format(
        imputation_results,
        answers_file,
        stats_csv
    )
    
    print("\n" + "="*60)
    print("ALL STEPS COMPLETE!")
    print("="*60)
    print(f"Final results: {stats_csv}")


if __name__ == "__main__":
    main()
