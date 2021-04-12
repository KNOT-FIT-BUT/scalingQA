"""
This script creates irrelevant documents dataset, the script:
  extracts golden passage ids
  extract K non-golden passages for each golden passage
  build dataset with numericalized data
"""
import os
import random

from jsonlines import jsonlines

from ....index.db import PassageDB
from ....common.utility.utility import mkdir, set_seed


def get_golden_passages(data_source):
    dataset = jsonlines.open(data_source)
    try:
        r = set([ex['contexts']['positive_ctx'] for ex in dataset if ex['is_mapped']])
    finally:
        dataset.close()
    return r


def build_dataset_for_NQ(raw_passages_db, output_folder, training_data_source, validation_test_data_source,
                         xtimes_more_irr_psgs_train=2,
                         total_length=21_015_324):
    mkdir(output_folder)

    global para_db
    para_db = PassageDB(db_path=raw_passages_db)

    def extract_dataset(infile, outfile, negative_X=1, golden_passages=None):
        if not golden_passages:
            golden_passages = get_golden_passages(infile)

        # too slow
        print("Precomputing negatives...\n")
        all_negative_passages = list(set(range(total_length)) - golden_passages)

        def get_random_negative():
            return random.choice(all_negative_passages)

        print("Constructing dataset...\n")

        total_processed_examples = 0
        total_created_examples = 0
        with jsonlines.open(outfile, mode='w') as writer:
            for example_idx, example in enumerate(jsonlines.open(infile)):
                if not example['is_mapped'] or not example['contexts']['positive_ctx'] in golden_passages:
                    continue

                total_processed_examples += 1
                # write positive
                title, passage = para_db.get_doc_text(example['contexts']['positive_ctx'],
                                                      ['raw_document_title', 'raw_paragraph_context'])
                total_created_examples += 1
                writer.write({
                    "id": example['example_id'],
                    "title": title,
                    "psg": passage,
                    "label": 0,
                })

                # assert example['title'] == title, f"Titles do not match {example['title']} /=/ {title}"

                # write negatives
                raw_negatives_ids = [get_random_negative() for _ in range(negative_X)]
                raw_negative_titles, raw_negatives = [], []
                for negative_id in raw_negatives_ids:
                    title, text = para_db.get_doc_text(negative_id,
                                                       columns=["raw_document_title", "raw_paragraph_context"])
                    raw_negative_titles.append(title)
                    raw_negatives.append(text)
                for n_id, n_title, n_psg in zip(raw_negatives_ids, raw_negative_titles, raw_negatives):
                    total_created_examples += 1
                    writer.write({
                        "id": n_id,
                        "title": n_title,
                        "psg": n_psg,
                        "label": 1,
                    })
                if total_processed_examples % 2000 == 0 and total_processed_examples > 0:
                    print(f"Processed {total_processed_examples} examples")
                    print(f"Created {total_created_examples} examples")
        print(f"Total processed {total_processed_examples} examples")
        print(f"Total created {total_created_examples} examples")

    nq_train_data_file = os.path.join(output_folder, f"train.jsonl")
    extract_dataset(training_data_source, nq_train_data_file, negative_X=xtimes_more_irr_psgs_train)

    nq_val_data_file = os.path.join(output_folder, "val.jsonl")
    nq_test_data_file = os.path.join(output_folder, "test.jsonl")

    # split NQ-open validation data 1/3 for validation, 2/3 for test
    nq_open_val = list(get_golden_passages(validation_test_data_source))
    random.shuffle(nq_open_val)
    val_golden_passages, test_golden_passages = set(nq_open_val[:len(nq_open_val) // 3]), \
                                                set(nq_open_val[len(nq_open_val) // 3:])

    extract_dataset(validation_test_data_source, nq_val_data_file, negative_X=1, golden_passages=val_golden_passages)
    extract_dataset(validation_test_data_source, nq_test_data_file, negative_X=1, golden_passages=test_golden_passages)


def main():
    raw_passages_db = ".index/wiki2018_dpr_blocks.db"
    output_folder = ".data/nq_corpus_pruning"
    training_data_source = ".data/nqopen/nq-open_train_short_maxlen_5_ms_with_dpr_annotation.jsonl"
    validation_data_source = ".data/nqopen/nq-open_dev_short_maxlen_5_ms_with_dpr_annotation.jsonl"
    set_seed(1234)

    build_dataset_for_NQ(raw_passages_db, output_folder, training_data_source, validation_data_source,
                         xtimes_more_irr_psgs_train=2)


if __name__ == "__main__":
    main()
