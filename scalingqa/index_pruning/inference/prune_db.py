"""
Prune db according to index
"""
import pickle
import sqlite3

from tqdm import tqdm

from ...index.db import PassageDB

if __name__ == "__main__":
    # database with passages
    FULL_DB_PATH = ".wikipedia/wiki2018_dpr_blocks.db"

    # pickled set of all relevant passages
    # PRUNE_FILE = f".pruning/relevant_psgs_w100_irrelevant_passage_probs_electra_nqopen_with_nq-open_train_short_maxlen_5_ms_with_dpr_annotation.jsonl_p1700000.pkl"
    PRUNE_FILE = ".pruning/relevant_psgs_w100_irrelevant_passage_probs_electra_trivia_with_triviaqa-open_train_with_dpr_annotation.jsonl_p1700000.pkl"

    # output file
    P = 1_700_000
    DATASET = "triviaopen"
    OUTPUT_DB_FILE = FULL_DB_PATH[:-3] + f"_{DATASET}_pruned_P{P}" + ".db"

    with open(PRUNE_FILE, 'rb') as f:
        prune_indices = sorted(list(pickle.load(f)))

    db = PassageDB(db_path=FULL_DB_PATH)
    doc_ids = db.get_doc_ids()

    # prune doc_ids
    doc_ids_new = prune_indices
    # make sure all prune ids are in doc_ids
    print("Making sure, all pruning indices are valid...")
    doc_ids_int_set = set(doc_ids)
    for p in prune_indices:
        assert p in doc_ids_int_set
    doc_ids = doc_ids_new

    with sqlite3.connect(OUTPUT_DB_FILE) as pruned_db:
        db_cursor = pruned_db.cursor()
        db_cursor.execute(
            "CREATE TABLE paragraphs (id PRIMARY KEY, raw_document_title, raw_paragraph_context)")
        pruned_db.commit()

        for new_id, doc_id in enumerate(tqdm(doc_ids)):
            title, context = db.get_doc_text(doc_id,
                                             columns=["raw_document_title", "raw_paragraph_context"])
            json_e = {
                "id": new_id,
                "raw_document_title": title,
                "raw_paragraph_context": context,
            }
            db_cursor.execute(
                "INSERT INTO paragraphs VALUES (:id, :raw_document_title, :raw_paragraph_context)",
                json_e)
        pruned_db.commit()
