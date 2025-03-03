import json
import time

#def init_parser(parser_name='stanza', language=None):
def init_parser(args):
    if args.language:
        lang = args.language
    else:
        data2lang = {
            "deuparl": 'de',
            "hansard": 'en'
        }
        lang = data2lang[args.data]

    if args.parser == "towerparse":
        from tower_parser import TowParser
        parser = TowParser(batch_size=args.batch_size, language=lang, ckpt_dir=args.checkpoint)
    elif args.parser == "stackpointer":
        from stackpointer_parser import StackPointerParser
        parser = StackPointerParser(ckpt_dir=args.checkpoint, language=lang, batch_size=args.batch_size)
    elif args.parser == "stanza":
        from stanford_parser import StanfordParser
        parser = StanfordParser(language=lang, batch_size=args.batch_size)
    elif args.parser in ["biaffine", "crf2o"]:
        from biaffine_supar import SuparBiaffineParser
        parser = SuparBiaffineParser(batch_size=args.batch_size, language=lang, ckpt_dir=args.checkpoint, parser_type=args.parser)
    elif args.parser == 'corenlp':
        from corenlp_parser import StanfordParser
        parser = StanfordParser(language=lang, batch_size=args.batch_size, ckpt_dir=args.checkpoint, port=args.port)
    else:
        raise NotImplementedError("This parser is not implemented.")
    return parser


if __name__ == '__main__':
    from argparse import ArgumentParser
    from glob import glob
    from tqdm import tqdm
    import os
    import pandas as pd
    import json

    args_parser = ArgumentParser()
    args_parser.add_argument("--data", '-d', type=str, required=True)
    args_parser.add_argument("--language", '-l', type=str)
    args_parser.add_argument("--parser", "-p", type=str, required=True)
    args_parser.add_argument("--checkpoint", "-c", type=str)
    args_parser.add_argument("--version", "-v", type=int, default=4)
    args_parser.add_argument("--start", '-s', type=int, default=1800)
    args_parser.add_argument("--end", "-e", type=int, default=2020)
    args_parser.add_argument("--batch_size", '-b', type=int, default=4)
    args_parser.add_argument("--port", type=int, default=9000)
    args_parser.add_argument("--id_json", type=str, default=None)
    args_parser.add_argument("--data_path", type=str, default=None)
    args_parser.add_argument("--corr_check", action='store_true')
    args = args_parser.parse_args()

    if args.id_json:
        with open(args.id_json, 'r') as f:
            ids = json.load(f)['data']
            #print(ids)
        #raise ValueError
    if not args.language:
        args.language = 'de' if args.data == 'deuparl' else 'en'
    
    if not args.corr_check:
        out_dir = f"../../data/{args.data}_final/parsed_v{args.version}{'_'+args.id_json.split('/')[-1].split('.')[0] if args.id_json else ''}/{args.parser}{'_'+args.checkpoint.split('/')[-1] if args.checkpoint else ''}/"
        data = f"../../data/{args.data}_final/stanza_tokenized_v{args.version}/*.csv"
    else:
        data = f"../../data/{args.data}_validation/annotation.csv"
        out_dir = f"../../data/{args.data}_validation/parsed/{args.parser}{'_'+args.checkpoint.split('/')[-1] if args.checkpoint else ''}/"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Out Directory doesn't exist. Makedirs.")

    parser = init_parser(args)
    
    for file in tqdm(sorted(glob(data))):
        if not args.corr_check:
            decade = file.split('/')[-1][:4]
            if args.id_json:
                decade_ids = [int(i.split('-')[-1]) for i in ids if i.split('-')[0]==decade]
            decade = int(decade)
        
            print()
            print(decade)
            if args.start <= decade <= args.end:
                df = pd.read_csv(file, delimiter='\t')#[:10]#[119190:119200]
                #if args.parser in ["stanza", "biaffine"]:
                sentences = list(df['tokenized'])
                if args.id_json:
                    sentences = [sentences[i] for i in decade_ids]
                    print(len(sentences))
                results = parser.parse(sentences=sentences, out='conllu', tokenized=True, mw_parents=[[]] * len(sentences))

                with open(os.path.join(out_dir, f"{decade}.conllu"), 'w', encoding='utf8') as f:
                    f.write(results)
                if args.parser in ["towerparse", "stackpointer", "biaffine", "crf2o"]:
                    with open(os.path.join(out_dir, f"{decade}_discarded.json"), 'w', encoding='utf8') as f:
                        json.dump([decade_ids[i] for i in parser.discard], f, indent=2)
            else:
                print("skipped")
        else:
            df = pd.read_csv(file)
            
            if args.data == 'deuparl':
                df = df[df.in_first_10 == 1]
            df = df[(df.is_sent==True) & (df.correction!='unknown')]
            #print(len(df[~df.correction.isna()]))
            #raise ValueError
            correction = [row['correction'] if isinstance(row['correction'], str) else row['text'] for _, row in df.iterrows()]
            #print(correction)
            #raise ValueError
            original = list(df['text'])
            print('Parsing corrections...')
            #print(correction[22])
            #raise ValueError
            correction = parser.parse(sentences=correction, out='conllu', tokenized=False, mw_parents=[[]] * len(correction))
            discarded = parser.discard
            print('Parsing original...')
            original = parser.parse(sentences=original, out='conllu', tokenized=False, mw_parents=[[]] * len(original))
            
            with open(os.path.join(out_dir, f"correction.conllu"), 'w', encoding='utf8') as f:
                f.write(correction)
            with open(os.path.join(out_dir, f"original.conllu"), 'w', encoding='utf8') as f:
                f.write(original)
            #print(df)
            

        #raise ValueError