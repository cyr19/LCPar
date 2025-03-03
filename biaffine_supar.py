from basic_parser import *
from supar import Parser


class SuparBiaffineParser(UniversalParser):
    def __init__(self, ckpt_dir="dep-biaffine-xlmr", batch_size=128, language='en', parser_type='biaffine'):
        super(SuparBiaffineParser, self).__init__(ckpt_dir=ckpt_dir, parser_type=parser_type, batch_size=batch_size, language=language)
        self.parser = self.init_parser()

    def init_parser(self):
        parser = Parser.load(self.ckpt_dir)
        return parser

    def parse(self, sentences, out='conllu', tokenized=True, mw_parents=[]):
        results = []
        if tokenized:
            print('tokenized...')
            print(sentences[0])
            tokenized_sentences = [s.split() for s in sentences]
        else:
            print('not tokenized...')
            print(sentences[0])
            tokenized_sentences, mw_parents = self.tokenize(sentences)

        self.discard = [i for i, tokens in enumerate(tokenized_sentences) if len(tokens) >= 200]
        print(f"\n{len(self.discard)} out of {len(sentences)} were discarded.\n")
        tokenized_sentences = [tokens for i, tokens in enumerate(tokenized_sentences) if i not in self.discard]
        sentences = [s for i, s in enumerate(sentences) if i not in self.discard]
        mw_parents = [mw for i, mw in enumerate(mw_parents) if i not in self.discard]

        parsed = self.parser.predict(tokenized_sentences, batch_size=self.batch_size, prob=False, verbose=True, proj=False, tree=True)

        assert len(parsed) == len(tokenized_sentences)
        for j, p in enumerate(parsed):
            r = []
            heads = p.arcs
            tokens = p.texts
            deprel = p.rels
            for i in range(len(heads)):
                tmp_r = {'tid': j, 'id': i+1, "token": tokens[i], "head_id": heads[i], "head": tokens[heads[i]-1] if heads[i] > 0 else "root", "deprel": deprel[i], "pos": "_"}
                r.append(tmp_r)
            results.append(r)

        assert len(results) == len(tokenized_sentences)
        if out == 'conllu':
            return self.convert_to_conull(results, sentences, mw_parents)
        else:
            return results


