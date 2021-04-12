import h5py
import pickle
import os
import tables
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    # full embeddings to be pruned
    FULL_EMBEDDINGS = ".embeddings/DPR_multiset_official.h5"

    # Nq-open pruners
    # PRUNE_FILE = f".pruning/relevant_psgs_w100_irrelevant_passage_probs_electra_nqopen_with_nq-open_train_short_maxlen_5_ms_with_dpr_annotation.jsonl_p1700000.pkl"

    # Trivia pruners
    PRUNE_FILE = ".pruning/relevant_psgs_w100_irrelevant_passage_probs_electra_trivia_with_triviaqa-open_train_with_dpr_annotation.jsonl_p1700000.pkl"

    D_MODEL = 768  # dimensionality of output embeddings
    P = 1_700_000  # how many passages are kept
    OUTPUT_EMBMATRIX_FILE = f".embeddings/{os.path.basename(FULL_EMBEDDINGS)[:-3] + f'_electrapruner_triviaopen_{P}.h5'}"

    if os.path.isfile(OUTPUT_EMBMATRIX_FILE):
        raise ValueError(f"File {OUTPUT_EMBMATRIX_FILE} already exists!")

    data = h5py.File(FULL_EMBEDDINGS, 'r')['data']
    with open(PRUNE_FILE, 'rb') as f:
        prune_indices = sorted(list(pickle.load(f)))

    f = tables.open_file(OUTPUT_EMBMATRIX_FILE, mode='w')
    try:
        atom = tables.Float32Atom()
        array_c = f.create_earray(f.root, 'data', atom, (0, D_MODEL))

        relevant_idx = prune_indices.pop(0)
        for idx, vector in tqdm(enumerate(data), total=21_015_324):
            if idx == relevant_idx:
                array_c.append(vector[np.newaxis, :])
                if len(prune_indices) > 0:
                    relevant_idx = prune_indices.pop(0)
                else:
                    break
    finally:
        f.close()
