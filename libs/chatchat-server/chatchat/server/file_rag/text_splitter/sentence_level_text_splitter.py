import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from nltk import tokenize
logger = logging.getLogger(__name__)


def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class SentenceRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
            
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s"
        ]
        self._is_separator_regex = is_separator_regex
    
    def _split_sentence(self,text):
        return tokenize.sent_tokenize(text)
        
        
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        self.chunk_overlap=0
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)
    
        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        split_len = 0
        _separator = "" if self._keep_separator else separator
        for split in splits:
            
            for s in self._split_sentence(split):
                # print(s+)
                _good_splits.append(s+" ")
                split_len = split_len+self._length_function(s)
                if split_len > self._chunk_size:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    # print(merged_text)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                    split_len =0
                if not new_separators:
                    final_chunks.append(s)
        if len(_good_splits)>0:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
            _good_splits = []
            split_len =0
        if len(final_chunks)> 2:
            return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip()!=""]
        else:
            return final_chunks


if __name__ == "__main__":
    text_splitter = SentenceRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=100,
        chunk_overlap=0
    )
    ls = [
        """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi at pellentesque purus. Nunc viverra sapien orci, non varius lacus accumsan vel. Integer lobortis enim eget commodo sodales. Mauris bibendum accumsan turpis, a convallis sapien venenatis et. Vestibulum vitae congue orci. Aliquam erat volutpat. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Etiam tempor, tortor eu posuere dictum, ipsum ante pretium eros, in tempus justo orci in tellus. Aenean eget urna a nisi fermentum imperdiet. In eu tortor tempus diam volutpat feugiat eu et erat.

Nam rhoncus eleifend sodales. Mauris id blandit nunc, at semper tellus. Quisque a consequat sem, id cursus odio. Nunc bibendum neque non magna tincidunt, vitae interdum sapien ullamcorper. Vestibulum ullamcorper metus quis nisl faucibus, nec venenatis odio rhoncus. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Nullam ac placerat mauris, ut fringilla odio. Fusce ac nibh justo. Proin vel egestas risus. Curabitur id vestibulum sapien. Vestibulum ac leo vel tortor mattis rhoncus. Morbi pretium orci nisi, auctor tincidunt nisl dictum eget. Nullam non felis euismod, ullamcorper dolor sit amet, maximus ipsum. Proin ultrices nulla interdum augue convallis sollicitudin. Duis sit amet felis interdum, ultrices purus et, pretium leo. Curabitur dapibus imperdiet ex, sit amet aliquet odio scelerisque cursus.

Fusce rutrum magna eget pharetra ultrices. Nulla facilisi. Vivamus ullamcorper leo euismod eros sagittis, vitae scelerisque augue ultricies. Proin congue, magna eget vulputate gravida, elit massa rhoncus eros, a suscipit neque enim nec turpis. Aenean vel purus est. Duis condimentum odio interdum metus finibus, et eleifend nibh suscipit. Donec ultricies placerat eros, ac elementum odio lobortis ut. Cras pharetra, purus condimentum blandit tristique, justo felis cursus arcu, id pretium nulla lacus quis quam. Nam sapien lacus, scelerisque nec semper sed, tempus sed nunc. Praesent vitae purus et urna volutpat porttitor id eget odio.

Sed et convallis risus. Nulla ante elit, facilisis vitae ex vitae, congue placerat felis. Duis purus augue, malesuada eu ultrices sit amet, tristique a nulla. Praesent accumsan eros a enim malesuada, et sollicitudin enim tempor. Curabitur eget risus sit amet nibh aliquet interdum. Aliquam commodo tempor risus, eget ultrices erat tempor nec. Suspendisse pulvinar eros et massa tempus, suscipit eleifend arcu imperdiet. Aliquam ultrices elementum diam, ac pharetra est tincidunt in. Nam sed pharetra erat.

Sed accumsan egestas justo, quis viverra sem semper vel. In at massa a justo lobortis maximus. Nunc id neque lectus. Nam ultrices nibh in mauris lobortis tempor. Sed sodales nisl mauris, id efficitur mi vehicula consectetur. Quisque condimentum erat laoreet diam facilisis, vel congue nibh tempus. Cras fringilla, leo id egestas auctor, est nisi tincidunt tellus, id ultrices nisl magna a justo. Maecenas id pharetra ipsum, id rutrum ligula.""",
        ]
    ls =["The Dow Jones Industrial Average, an American stock index composed of 30 large companies, has changed its components 58 times since its inception, on May 26, 1896. As this is a historical listing, the names here are the full legal name of the corporation on that date, with abbreviations and ...The Dow Jones Industrial Average, an American stock index composed of 30 large companies, has changed its components 58 times since its inception, on May 26, 1896. As this is a historical listing, the names here are the full legal name of the corporation on that date, with abbreviations and punctuation according to the corporation's own usage. An up arrow ( ↑ ) indicates the company is added. A down arrow ( ↓ ) indicates the company is removed. A down arrow ( ↓ ) indicates the company is removed. A dagger ( † ) indicates a change of corporate name. The index change was prompted by DJIA constituent Walmart Inc.’s decision to split its stock 3:1 thereby reducing Walmart’s index weight due to the price weighted construction of the index. United Technologies Corporation merged with Raytheon Company and new corporation entered index as Raytheon Technologies Corporation. DowDuPont spun off DuPont and was replaced by Dow Inc. Prior to the May 26, 1896, inception of the Dow Jones Industrial Average, Charles Dow's stock average consisted of the Dow Jones Transportation Average. The average was created on July 3, 1884 by Charles Dow, co-founder of Dow Jones & Company, as part of the Customer's Afternoon Letter."]
    chunks = text_splitter.split_text(ls[0])

    for chunk in chunks:
        print((chunk))
        print("@@@"*10)
