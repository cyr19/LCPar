import stanza
from basic_parser import *
from stanza.models.common.doc import Document
import re
import string

class StanfordParser(UniversalParser):
    def __init__(self, ckpt_dir=None, batch_size=1024, language='de'):
        super(StanfordParser, self).__init__(parser_type='stanza', language=language, batch_size=batch_size, ckpt_dir=ckpt_dir)
        self.parser = self.init_parser()

    def init_parser(self):
        parser = stanza.Pipeline(lang=self.language, processors='tokenize,pos,lemma,depparse',
                                 tokenize_pretokenized=True,
                                 use_gpu=True if self.device == 'cuda' else False,
                                 depparse_batch_size=self.batch_size)
        return parser

    def parse(self, sentences, tokenized=True, out='conllu', mw_parents=[]):
        results = []
        if not tokenized:
            print('not tokenized...')
            print(sentences[0])
            tokenized_sentences, mw_parents = self.tokenize(sentences)
        else:
            print('tokenized...')
            print(sentences[0])
            tokenized_sentences = [s.split() for s in sentences]
 
        doc = self.parser(tokenized_sentences)

        for i, sentence in enumerate(doc.sentences):
            r = []
            for word in sentence.words:
                r.append({'tid': i, 'id': word.id, "token": word.text, "head_id": word.head,
                          "head": sentence.words[word.head - 1].text if word.head > 0 else "root",
                          "deprel": word.deprel, "pos": '_'})

            results.append(r)

        assert len(results) == len(sentences)
        if out == 'conllu':
            return self.convert_to_conull(results, sentences, mw_parents)
        else:
            return results
