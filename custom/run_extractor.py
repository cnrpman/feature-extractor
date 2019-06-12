from custom.bert_extractor import BertExtractor
import argparse
import pickle
import re
import spacy
import numpy as np
from tqdm import tqdm

def read_file(path):
    print("loading spacy model")
    nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    # index of doc sentences, doc key, all ssentences
    indices, keys, all_sents = [], [], []
    max_len = 0

    pattern = re.compile(r"^([^\t]*)\t\s*(.*)")
    print("loading file & sentence breaking")
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            m = pattern.match(line.strip())
            if not m:
                continue
            keys.append(m.group(1))
            value = m.group(2)
            doc = nlp(value)
            sents = [sen.string.strip() for sen in doc.sents]
            max_len = max(max_len, max(map(len, doc.sents)))
            indices.append(list(range(len(all_sents), len(all_sents)+len(sents))))
            all_sents.extend(sents)

    print("max length of sentence %d" % max_len)
    
    return indices, keys, all_sents
                

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()

    indices, keys, all_sents = read_file(args.input_file)

    extractor = BertExtractor(
        bert_model=args.bert_model, 
        do_lower_case=args.do_lower_case, 
        layers=args.layers.split(','),
        reduce_method="sum", 
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size, 
        local_rank=args.local_rank
    )

    features = extractor.extract(all_sents)

    print("merging sentences")
    kv = {}
    for key, index in zip(keys, indices):
        kv[key] = np.average(np.array([features[idx] for idx in index]), axis=0)

    print("writing to path %s" % args.output_file)
    with open(args.output_file, 'wb') as f:
        pickle.dump(kv, f)

if __name__ == "__main__":
    main()
