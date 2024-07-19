import spacy
from typing import List
from spacy.tokens import Doc, Span
from fastcoref import FCoref
from fastcoref import spacy_component
from fastcoref import LingMessCoref

class CorefReplacer():
    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')
        self.model = FCoref()
        # self.model = FCoref(device='cuda:3')

        

    def core_logic_part(self, document: Doc, coref: List[int], resolved: List[str], mention_span: Span):
        final_token = document[coref[1]]
        if final_token.tag_ in ["PRP$", "POS"]:
            resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
        else:
            resolved[coref[0]] = mention_span.text + final_token.whitespace_
        for i in range(coref[0] + 1, coref[1] + 1):
            resolved[i] = ""
        return resolved

    def get_span_noun_indices(self, doc: Doc, cluster: List[List[int]]) -> List[int]:
        spans = [doc[span[0]:span[1]+1] for span in cluster]
        spans_pos = [[token.pos_ for token in span] for span in spans]
        span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
            if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]
        return span_noun_indices

    def get_cluster_head(self, doc: Doc, cluster: List[List[int]], noun_indices: List[int]):
        head_idx = noun_indices[0]
        head_start, head_end = cluster[head_idx]
        head_span = doc[head_start:head_end+1]
        return head_span, [head_start, head_end]

    def is_containing_other_spans(self, span: List[int], all_spans: List[List[int]]):
        return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])

    def improved_replace_corefs(self, document, clusters):
        resolved = list(tok.text_with_ws for tok in document)
        all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans

        for cluster in clusters:
            noun_indices = self.get_span_noun_indices(document, cluster)
            
            if noun_indices:
                mention_span, mention = self.get_cluster_head(document, cluster, noun_indices)

                for coref in cluster:
                    if coref != mention and not self.is_containing_other_spans(coref, all_spans):
                        self.core_logic_part(document, coref, resolved, mention_span)
                        
        return "".join(resolved)

    def get_fast_cluster_spans(self, doc, clusters):
        fast_clusters = []
        for cluster in clusters:
            new_group = []
            for tuple in cluster:
                print(type(tuple), tuple)
                (start, end) = tuple
                print("start, end", start, end)
                span = doc.char_span(start, end)
                print('span', span.start, span.end)
                new_group.append([span.start, span.end-1])
            fast_clusters.append(new_group)
        return fast_clusters

    def get_fastcoref_clusters(self, doc, text):
        preds = self.model.predict(texts=[text], max_tokens_in_batch=4096)
        fast_clusters = preds[0].get_clusters(as_strings=False)
        fast_cluster_spans = self.get_fast_cluster_spans(doc, fast_clusters)
        return fast_cluster_spans


    def pronoun_replace(self, text):

        doc = self.nlp(text)
        clusters = self.get_fastcoref_clusters(doc, text)
        coref_text = self.improved_replace_corefs(doc, clusters)
        
        return coref_text

if __name__ == "__main__":
    # text ="We are so happy to see you using our coref package. This package is very fast!"
    # text = "I love my dog. Its name is Tim. It is very naughty. I love my cat. It is very calm"
    # text = "We are Tommy. We want to take our code and create a game. Let's remind ourselves how to do that."
    # text = "The host of the show, Karen Webster, sat down with John Smith to discuss the future."
#     text ="""
# Dr Xiaofang Zhou is Otto Poon Professor of Engineering and Chair Professor of Computer Science and Engineering at The Hong Kong University of Science and Technology. He is Head of Department of Computer Science and Engineering. He is the Founding Director of Jockey Club STEM Lab of Data Science Foundations (DSF), HKUST-HKPC Joint Lab on Industrial AI and Robotics Research (INAIR), HKUST-MetaX Joint Lab for Advanced AI Computing, and several other HKUST joint labs with leading industry players. He was Co-Director of HKUST Big Data Institute (BDI) from 2022 to 2024. He is a Global STEM Scholar of Hong Kong and a Fellow of IEEE.

# Professor Zhou's research focus is to find effective and efficient solutions for managing, integrating and analyzing large-scale complex data for business, scientific and personal applications. He has been working in spatiotemporal databases, data quality management, big data analytics, recommender systems, machine learning and AI, co-authored over 500 research papers. He received the Best Paper Awards of WISE 2012&2013, ICDE 2015&2019, DASFAA 2016 and ADC 2019. He was the Program Committee Chair of IEEE International Conference on Data Engineering (ICDE 2013), ACM International Conference on Information and Knowledge Management (CIKM 2016), and International Conference on Very Large Databases (PVLDB 2020). He was the General Chair of ACM Multimedia Conference (MM 2015) and ICDE 2025. He was a keynote speaker at WISE 2008, CIKM 2015, DEXA 2018, MDM 2019, ADMA 2023 and ADC 2024. 

# He was a Professor of Computer Science at the University of Queensland (UQ) from 1999 - 2020, leading its Data and Knowledge Engineering (DKE) research group and the Data Science discipline. Before joining UQ, he was a senior research scientist and leader of the Spatial Information Systems group at CSIRO. He received his Bachelor and Master in Computer Science degrees from Nanjing University and his PhD in Computer Science from UQ. He is also a Chair Professor at HKUST Shenzhen Research Institute and an Affiliate Professor at HKUST (Guangzhou). 
#     """

    replacer = CorefReplacer()

    print(replacer.pronoun_replace(text))