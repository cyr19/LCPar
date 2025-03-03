import collections

import numpy as np

from conll18_ud_eval import *
import pandas as pd

from parse_all import *
from argparse import ArgumentParser
from glob import glob
import os
import re

def exists_cycle(heads):
    visited = [False] * len(heads)
    graph = collections.defaultdict(list)
    for i, head in enumerate(heads):
        graph[head].append(i+1)

    def dfs(graph, node, visited, start):
        if visited[node-1]:
            if node == start:
                return True

        visited[node-1] = True
        for child in graph[node]:
            dfs(graph, child, visited, start)
        visited[node-1] = False

    detected = 0
    for i in range(1, len(heads)+1):
        detected = dfs(graph, i, visited, i)
        if detected:
            break
    return detected


def read_conllu(path, discard=[]):
    if 'conllu' in path:
        with open(path, 'r', encoding='utf8') as f:
            data = f.read().strip()
    else:
        data = path.strip()

    instances = re.split('\n{2,}', string=data)
    tokenized_sentences, gold_heads, gold_rels, gold_roots, mw_parents = collections.defaultdict(list), [], [], [], [] * len(instances)
    for i, instance in enumerate(instances):
        if i in discard:
            continue
        instance = instance.strip()
        sent, toknized_sent, upos, xpos, lemma = None, [], [], [], []
        gold_h, gold_r = [], []
        root = []
        mw = {}
        for l in instance.split('\n'):
            
            if l.startswith("#"):
                if l.startswith("# text = "):
                    sent = l.split("# text = ")[-1]
                continue
            tks = l.split('\t')
            if not re.match("^\d+$", string=tks[0]):
                if "-" in tks[0]:
                    pos = tks[0].split('-')
                    mw.update({j: tks[1] for j in range(int(pos[0])-1, int(pos[1]))})
                continue
            toknized_sent.append(tks[1])
            upos.append(tks[3])
            lemma.append(tks[2])
            xpos.append(tks[4])

            gold_h.append(int(tks[6]))
            gold_r.append(tks[7])
            if tks[6] == '0':
                root.append(int(tks[0]))
        if not args.tokenized:
            assert sent is not None, instance
            tokenized_sentences['sentence'].append(sent)
        else:
            tokenized_sentences['sentence'].append(" ".join(toknized_sent))
        tokenized_sentences['tokenized'].append(" ".join(toknized_sent))
        tokenized_sentences['upos'].append(upos)
        tokenized_sentences['xpos'].append(xpos)
        tokenized_sentences['lemma'].append(lemma)

        gold_heads.append(gold_h)
        gold_rels.append(gold_r)
        gold_roots.append(root[0] if len(root) > 0 else -1)
        mw = collections.OrderedDict(sorted(mw.items()))
        mw_parents.append(mw)

    return tokenized_sentences, gold_heads, gold_rels, gold_roots, mw_parents


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--language', '-l', required=True)
    parser.add_argument('--mode', '-m', type=str, default='parse')
    parser.add_argument('--data_dir', type=str, default='../../data/ud_treebanks/ud-treebanks-v2.12')
    parser.add_argument('--treebanks', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--parser', '-p', type=str, default='stanza')
    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--leaderboard', type=str, default="evaluation/revision.csv")
    parser.add_argument("--checkpoint", '-c', type=str)
    parser.add_argument("--tokenized", action='store_true')
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--use_cache", action='store_true')
    parser.add_argument("--cache_dir", type=str, default='cache')
    parser.add_argument("--ocr_attack", action="store_true")

    
    args = parser.parse_args()

    print(args)

    abb2lang = {
        'en': 'English',
        'de': 'German'
    }
    if not args.ocr_attack:
        if not args.data_path:
            paths = glob(os.path.join(args.data_dir, f"UD_{abb2lang[args.language]}*/*test.conllu"))
            if args.treebanks:
                treebanks = args.treebanks.split(",")
            
        else:
            paths = [args.data_path]
    else:
        paths = sorted(glob("../../data/adv_treebanks/ocr/*.conllu"))
        print(paths)
        #raise ValueError
    if not args.use_cache:
        parser = init_parser(args)

    final = collections.defaultdict(list)

    for path in sorted(paths):
        if args.data_path is None:
            dataset = path.split('/')[-2].split('-')[-1] if not args.ocr_attack else '-'.join(path.split('/')[-2:])
            #print(dataset)
            if args.treebanks:
                if dataset not in treebanks:
                    print(f"skip {dataset}...")
                    continue
        else:
            dataset = '-'.join(path.split('/')[-2:])

        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
        out_path = f"{args.cache_dir}/{args.language}_{dataset}_{args.parser}{'_tokenized' if args.tokenized else ''}.conllu"

        discarded = []
        if not args.use_cache:
            tokenized_sentences, gold_heads, gold_rels, gold_roots, mw_parents = read_conllu(path)
            if args.tokenized:
                results = parser.parse(sentences=tokenized_sentences['tokenized'], out='conllu', tokenized=True, mw_parents=mw_parents)
            else:
                results = parser.parse(sentences=tokenized_sentences['sentence'], out='conllu', tokenized=False)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(results)
            if args.parser in ['towerparse', 'stackpointer', 'biaffine', 'crf2o']:
                discarded = parser.discard
                print(f'\n{len(discarded)} sents were discarded...\n')

        gold_ud, gold_cycle_count, gold_multi_roots_count = load_conllu_file(path, discarded=discarded)
        system_ud, system_cycle_count, system_multi_roots_count = load_conllu_file(out_path)
        evaluation = evaluate(gold_ud, system_ud)

        uas = evaluation["UAS"].f1
        las = evaluation["LAS"].f1

        print("UAS F1 Score: {:.2f}".format(100 * uas))
        print("LAS F1 Score: {:.2f}".format(100 * las))
        print(f"{system_cycle_count} cycles detected.\n{system_multi_roots_count} multi roots detected.")

        final['language'].append(args.language)
        final['treebank'].append(dataset)
        final['tokenized'].append(args.tokenized)
        final['parser'].append(args.parser)
        #final['root_acc'].append(root_acc)
        final['uas'].append(uas)
        final['las'].append(las)
        final['cycle_count'].append(system_cycle_count)
        final['multi_roots_count'].append(system_multi_roots_count)
        final['skipped'].append(len(discarded))
        #break

    if len(paths) > 1:
        final['language'].append(args.language)
        final['treebank'].append('avg')
        final['tokenized'].append(args.tokenized)
        final['parser'].append(args.parser)
        final['uas'].append(np.average(final['uas']))
        final['las'].append(np.average(final['las']))
        final['cycle_count'].append(np.sum(final['cycle_count']))
        final['multi_roots_count'].append(np.sum(final['multi_roots_count']))
        final['skipped'].append(np.sum(final['skipped']))


    final = pd.DataFrame(final)
    if args.ocr_attack:
        attack = final[final.treebank.str.contains('attack')]
        ori = final[final.treebank.str.contains('ori')]
        assert len(attack) == 6 and len(ori) == 1
        '''
        final = final.append(pd.DataFrame({'language': [args.language], 'treebank': ["avg_attack"], 'tokenized': [args.tokenized], 'parser': [args.parser], 
                                   'uas': [np.mean(attack['uas'])], 'las': [np.mean(attack['las'])], 'cycle_count': [np.sum(attack['cycle_count'])],
                                   'multi_roots_count': [np.sum(attack['multi_roots_count'])], 'skipped': [np.sum(attack['skipped'])]}))
        final = final.append(pd.DataFrame({'language': [args.language], 'treebank': ["avg_ori"], 'tokenized': [args.tokenized], 'parser': [args.parser], 
                                   'uas': [np.mean(ori['uas'])], 'las': [np.mean(ori['las'])], 'cycle_count': [np.sum(ori['cycle_count'])],
                                   'multi_roots_count': [np.sum(ori['multi_roots_count'])], 'skipped': [np.sum(ori['skipped'])]}))
        final = final.append(pd.DataFrame({'language': [args.language], 'treebank': ["avg_drop"], 'tokenized': [args.tokenized], 'parser': [args.parser], 
                                   'uas': [np.mean(attack['uas'])-np.mean(ori['uas'])], 'las': [np.mean(attack['las'])-np.mean(ori['las'])], 'cycle_count': [np.sum(ori['cycle_count'])],
                                   'multi_roots_count': [np.sum(ori['multi_roots_count'])], 'skipped': [np.sum(ori['skipped'])]}))
        '''
        attack = pd.DataFrame({'language': [args.language], 'treebank': ["avg_attack"], 'tokenized': [args.tokenized], 'parser': [args.parser], 
                                   'uas': [np.mean(attack['uas'])], 'las': [np.mean(attack['las'])], 'cycle_count': [np.sum(attack['cycle_count'])],
                                   'multi_roots_count': [np.sum(attack['multi_roots_count'])], 'skipped': [np.sum(attack['skipped'])]})
        ori = pd.DataFrame({'language': [args.language], 'treebank': ["avg_ori"], 'tokenized': [args.tokenized], 'parser': [args.parser], 
                                   'uas': [np.mean(ori['uas'])], 'las': [np.mean(ori['las'])], 'cycle_count': [np.sum(ori['cycle_count'])],
                                   'multi_roots_count': [np.sum(ori['multi_roots_count'])], 'skipped': [np.sum(ori['skipped'])]})
        drop = pd.DataFrame({'language': [args.language], 'treebank': ["avg_drop"], 'tokenized': [args.tokenized], 'parser': [args.parser], 
                                   'uas': [np.mean(attack['uas'])-np.mean(ori['uas'])], 'las': [np.mean(attack['las'])-np.mean(ori['las'])], 'cycle_count': [np.sum(ori['cycle_count'])],
                                   'multi_roots_count': [np.sum(ori['multi_roots_count'])], 'skipped': [np.sum(ori['skipped'])]})
        final = pd.concat([final, attack, ori, drop], ignore_index=True)

    print(final)
    if not os.path.exists(args.leaderboard):
        final.to_csv(args.leaderboard, mode='w', index=False)
    else:
        final.to_csv(args.leaderboard, mode='a', index=False, header=False)



