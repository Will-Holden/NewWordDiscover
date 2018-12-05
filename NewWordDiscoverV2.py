import numpy as np
from utils import Utils
from tqdm import tqdm
import pandas as pd
from utils import Logger
import re
from collections import defaultdict
from tqdm import tqdm
from utils import Logger
import time


class NewWordDiscover:
    # data 是一个list
    def __init__(self, min_count=20, min_support={2: 5, 3: 20, 4: 40, 5: 125}, min_inf=3.5, max_sep=4, data=None):
        self.data = data
        self.result = []
        self.min_count = min_count
        self.min_support = min_support
        self.min_inf = min_inf
        self.max_sep = max_sep  # 最大字数
        self.total = 0
        self.ngrams = None
        self.pre_suffix = None
        self.after_suffix = None

    def encode_str(self, s):
        return " ".join(s)

    def decode_str(self, s):
        return s.split(" ")

    # @jit(nopython=True)
    def gen_ngrams(self):
        n = self.max_sep
        min_count = self.min_count
        ngrams = defaultdict(int)
        pre_suffix = defaultdict(dict)
        after_suffix = defaultdict(dict)
        t = self.data
        Logger.info("start generating ngrams")
        for i in tqdm(range(len(t))):
            for j in range(1, n + 1):
                if i + j <= len(t):
                    ngrams[self.encode_str(t[i:i + j])] += 1
                if i - 1 >= 0:
                    pre_suffix[self.encode_str(t[i:i + j])][t[i - 1]] = pre_suffix[self.encode_str(t[i:i + j])].get(
                        t[i - 1], 0) + 1
                if j + i + 1 <= len(t):
                    after_suffix[self.encode_str(t[i:i + j])][t[j + i]] = after_suffix[self.encode_str(t[i:i + j])].get(
                        t[j + i], 0) + 1

        self.ngrams = dict()
        self.total = 0
        for i, j in ngrams.items():
            if len(i) == 1:
                self.total += j
            if j >= min_count:
                self.ngrams[i] = j

        self.pre_suffix = pre_suffix
        self.after_suffix = after_suffix

    # @jit(nopython=True)
    def support_filter(self, min_proba={2: 5, 3: 5, 4: 5}):
        ngrams = self.ngrams
        total = self.total
        self.ngrams = {}
        Logger.info("starting compute support score")
        for word, count in tqdm(ngrams.items()):
            is_keep = False
            s = word
            if len(self.decode_str(s)) >= 2:
                numerator = []
                for index in range(len(self.decode_str(s)) - 1):
                    numerator.append(ngrams[self.encode_str(self.decode_str(s)[:index + 1])] * ngrams[
                        self.encode_str(self.decode_str(s)[index + 1:])])
                score = np.min(total * ngrams[s] / np.array(numerator))
                print(s)
                if score > min_proba[len(self.decode_str(s))]:
                    is_keep = True
            if is_keep:
                self.ngrams[word] = count

        # self.ngrams = {i: j for i, j in ngrams.items() if is_keep(i)}

    def inf_filter(self):
        min_inf = self.min_inf
        pre_suffix = self.pre_suffix
        after_suffix = self.after_suffix
        ngrams = self.ngrams
        self.ngrams = {}
        Logger.info("start computing inf scores")
        for word, count in tqdm(ngrams.items()):
            is_keep = False
            i = word
            pre_array = np.array(list(pre_suffix[i].values()), dtype=np.double)
            prob_array = pre_array / np.sum(pre_array)
            pre_info = -prob_array.dot(np.log(prob_array.transpose()))

            after_array = np.array(list(after_suffix[i].values()), dtype=np.double)
            prob_array = after_array / np.sum(after_array)
            after_info = - prob_array.dot(np.log(prob_array.transpose()))
            if pre_info > min_inf and after_info > min_inf:
                is_keep = True

            if is_keep:
                self.ngrams[word] = count

    def start_discover(self, data: object = None) -> object:
        if data:
            self.data = data
        self.gen_ngrams()
        self.support_filter(min_proba=self.min_support)
        self.inf_filter()
        return self.ngrams


if __name__ == '__main__':
    # data = 'a b c d'
    # data = data.split()
    data = \
    Utils.load_table_by_sql("select Content_processed from datasource_new_processed where Language ='cn' limit 2000")[
        'Content_processed']
    data = [word for doc in data for sentence in doc.split(';') for word in list(sentence)]
    # p = re.compile(' +')
    # data = p.sub(' ', data.replace('\n', ''))
    start_time = time.clock()
    newWordDiscover = NewWordDiscover()
    result = newWordDiscover.start_discover(data)
    end_time = time.clock()
    Logger.info("cost time {0}".format(end_time - start_time))
    print(result)
