import os.path
import stanza
from stanza.server import CoreNLPClient
from basic_parser import *
from stanza import download_corenlp_models

#stanza.install_corenlp()
#stanza.download

class StanfordParser(UniversalParser):
    def __init__(self, ckpt_dir="/home/ychen/stanza_corenlp/", batch_size=1024, language='de', port=9001):
        super(StanfordParser, self).__init__(parser_type='stanza', language=language, batch_size=batch_size, ckpt_dir=ckpt_dir)
        self.port = port
        self.init_parser()

    def init_parser(self):
        lang2full = {
            "de": "german",
            "en": "english-kbp",
        }
        if not os.path.exists(os.path.join(self.ckpt_dir, f"stanford-corenlp-4.5.5-models-{lang2full[self.language]}.jar")):
            download_corenlp_models(model=lang2full[self.language], version='4.5.5', dir=self.ckpt_dir)

    def parse(self, sentences, out="conllu", tokenized=False, mw_parents=[]):
        results = []
        if tokenized:
            print('tokenized...')
            print(sentences[0])
            tokenized_sentences = sentences
        else:
            print('not tokenized...')
            print(sentences[0])
            tokenized_sentences, mw_parents = self.tokenize(sentences)
            tokenized_sentences = [' '.join(s) for s in tokenized_sentences]
        print(tokenized_sentences[0])

        properties = self.language if self.language != 'en' else {"ssplit.isOneSentence": True, "tokenize.whitespace": True}

        with CoreNLPClient(
                             annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'depparse'],
                             properties = properties,
                             timeout=50000,
                             memory='6G',
                             threads=16,
                             endpoint=f"http://localhost:{self.port}",
                             #output_format='json',
                             be_quiet=True) as client:

            for i, s in tqdm(enumerate(tokenized_sentences), total=len(tokenized_sentences)):
                response = client.annotate(s).sentence
                assert len(response) == 1, response
                tokens = response[0].token
                tokens_ori = s.split()
                response = response[0].basicDependencies
                edges = response.edge
                root = response.root
                assert len(root) == 1 or len(set(root)) == 1, root
                root_token = tokens_ori[root[0]-1]
                r = []
                r.append({'tid': i, "id": root[0], "token": root_token, "head_id": 0,
                          "head": "root", "deprel": "root", "pos": "_"})

                for edge in edges:
                    token = tokens[edge.target-1]
                    r.append({'tid': i, "id": edge.target, "token": tokens_ori[edge.target-1], "head_id": edge.source,
                                  "head": tokens_ori[edge.source-1], "deprel": edge.dep, "pos": token.pos})

                r = sorted(r, key=lambda x: x["id"])
                results.append(r)
        self.port += 12
        assert len(results) == len(sentences)
        if out == 'conllu':
            return self.convert_to_conull(results, sentences, mw_parents)
        else:
            return results

