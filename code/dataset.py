import numpy as np
import multiprocessing
import re
import pickle
from tokenizer import MosesTokenizer, MosesDetokenizer

def encode_field(*args):
    args = [str(arg) for arg in args]
    return '_'.join(args)

def decode_field(field):
    fields = field.split('_')
    fields[2] = int(fields[2])
    if fields[1] in ['en', 'fr']:
        if fields[0] == 'encoder':
            if fields[3] in ['x', 'length']:
                return fields
        if fields[0] == 'decoder':
            if fields[3] in ['x', 'y', 'length']:
                return fields
                

def tokenize_job(tokenizer, lines, i, results):
    res = []
    for line in lines:
        tokenized = tokenizer.tokenize(line)
        res.append([w.lower() for w in tokenized])
    results[i] = res
    

def tokenize(tokenizer, lines, per_thread):
    results = multiprocessing.Manager().dict()
    processes = [
        multiprocessing.Process(
            target=tokenize_job, 
            args=(tokenizer, lines[i:min(len(lines), i + per_thread)], i / per_thread, results)) 
        for i in range(0, len(lines), per_thread)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    result = []
    for i in range((len(lines) - 1) // per_thread + 1):
        result += results[i]
    return result

def unknown(token):
    if re.match(r'^[-+]?[0-9]+(\.[0-9]+)?$', token):
        return '<number>'
    else:
        return '<unk>'
    

class Dataset(object):
    def __init__(self, english, french, num_words=200000, buckets=[20, 40, 200], nthreads=4):
        self.tokenizer = {}
        self.detokenizer = {}
        self.source = {}
        self.lengths = {}
        self.vocab = {}
        self.ivocab = {}
        self.buckets_sizes = buckets
        for lang in ['en', 'fr']:
            self.tokenizer[lang] = MosesTokenizer(lang=lang)
            self.detokenizer[lang] = MosesDetokenizer(lang=lang)
            self.lengths[lang] = []
            self.vocab[lang] = {}
            self.ivocab[lang] = {}
            for i, w in enumerate(['<pad>', '<s>', '<\s>', '<unk>', '<number>']):
                self.vocab[lang][w] = i
                self.ivocab[lang][i] = w
            
        self.source['en'] = english
        self.source['fr'] = french
        
        self.num_words = num_words
        self.nthreads = nthreads
        self.per_thread = 200
            
        for lang in ['en', 'fr']:
            self.__build_vocab(lang)
        
        self.__init_tensors()
        
    
    def __build_vocab(self, lang):
        voc = {}
        with open(self.source[lang], 'r') as f:
            buffer = []
            counter = 0
            for line in f:
                buffer.append(line)
                if len(buffer) >= self.per_thread * self.nthreads:
                    results = tokenize(self.tokenizer[lang], buffer, self.per_thread)
                    for result in results:  
                        self.lengths[lang].append(len(result))
                        for w in result:
                            voc[w] = voc.get(w, 0) + 1
                    buffer = []
                            
            if len(buffer) > 0:
                results = tokenize(self.tokenizer[lang], buffer, self.per_thread)
                for result in results: 
                    self.lengths[lang].append(len(result))
                    for w in result:
                        voc[w] = voc.get(w, 0) + 1
                        
        voc_list = list(voc.items())
        voc_list.sort(key=lambda x: -x[1])
        special_words = len(self.vocab[lang])
        voc_list = voc_list[:min(self.num_words - special_words, len(voc_list))]
        for i in range(len(voc_list)):
            self.vocab[lang][voc_list[i][0]] = i + special_words
            self.ivocab[lang][i + special_words] = voc_list[i][0]
            
    def __init_tensors(self):
        lengths = np.zeros(len(self.buckets_sizes), dtype=np.int)
        for i in range(len(self.lengths['en'])):
            l = max(self.lengths['en'][i], self.lengths['fr'][i])
            bucket_id = 0
            while bucket_id < len(self.buckets_sizes) - 1 and self.buckets_sizes[bucket_id] < l:
                bucket_id += 1
            lengths[bucket_id] += 1
            
        self.encoder = {}
        self.decoder = {}
        pad = {}
        
        for lang in ['en', 'fr']:
            pad[lang] = self.vocab[lang]['<pad>']
            
        for lang in ['en', 'fr']:
            self.encoder[lang] = []
            self.decoder[lang] = []
            for i in range(len(self.buckets_sizes)):
                self.encoder[lang].append(
                    {'x': pad[lang] * np.ones((lengths[i], self.buckets_sizes[i]), dtype=np.int),
                     'length': np.zeros(lengths[i], dtype=np.int)}
                ) 
                self.decoder[lang].append(
                    {'x': pad[lang] * np.ones((lengths[i], self.buckets_sizes[i] + 1), dtype=np.int),
                     'y': pad[lang] * np.ones((lengths[i], self.buckets_sizes[i] + 1), dtype=np.int),
                     'length': np.zeros(lengths[i], dtype=np.int)}
                )
                    
    def fill_tensors(self, lang):
        with open(self.source[lang], 'r') as f:
            buffer = []
            counters = np.zeros(len(self.buckets_sizes), dtype=np.int)
            index = 0
            for line in f:
                buffer.append(line)
                if len(buffer) >= self.per_thread * self.nthreads:
                    results = tokenize(self.tokenizer[lang], buffer, self.per_thread)
                    for r in results:
                        r_ = []
                        for w in r:
                            r_.append(self.vocab[lang].get(w, self.vocab[lang][unknown(w)]))
                        bucket_id = 0
                        current_len = max(self.lengths['en'][index], self.lengths['fr'][index])
                        while bucket_id < len(self.buckets_sizes) - 1 and self.buckets_sizes[bucket_id] < current_len:
                            bucket_id += 1
                            
                        #fill encoder
                        start = max(self.buckets_sizes[bucket_id] - len(r_), 0)
                        for i in range(start, self.buckets_sizes[bucket_id]): 
                            self.encoder[lang][bucket_id]['x'][counters[bucket_id]][i] = r_[i - start]
                        self.encoder[lang][bucket_id]['length'][counters[bucket_id]] =\
                            self.buckets_sizes[bucket_id] - start
                        
                        #fill decoder
                        end = min(len(r_) + 1, self.buckets_sizes[bucket_id] + 1)
                        self.decoder[lang][bucket_id]['length'][counters[bucket_id]] = end
                        r_ = [self.vocab[lang]['<s>']] + r_ + [self.vocab[lang]['<\s>']]
                        
                        for i in range(end):
                            self.decoder[lang][bucket_id]['x'][counters[bucket_id]][i] = r_[i]
                            self.decoder[lang][bucket_id]['y'][counters[bucket_id]][i] = r_[i + 1]
                            
                        counters[bucket_id] += 1
                        index += 1
                    buffer = []
                            
            if len(buffer) > 0:
                results = tokenize(self.tokenizer[lang], buffer, self.per_thread)
                        
    
    def test_train_split(self, ratio=0.01, random_seed=42):
        np.random.seed(random_seed)
        pass
    
    def encode(self, string, lang='en'):
        tokens = self.tokenizer[lang].tokenize(string)
        result = np.zeros(len(tokens), dtype=np.int)
        for i in range(len(tokens)):
            t = tokens[i].lower()
            result[i] = self.vocab[lang].get(t, self.vocab[lang][unknown(t)])
        return result
            
    def decode(self, tensor, lang='en'):
        t = tensor.flatten()
        result = []
        for i in range(t.size):
            result.append(self.ivocab[lang][t[i]])
        return self.detokenizer[lang].detokenize(result)
    
    def save(self, filename):
        fields = {}
        for k, v in self.__dict__.items():
            if not k in ['encoder', 'decoder', 'tokenizer', 'detokenizer']:
                fields[k] = v
        
        data = {}
        for lang in ['en', 'fr']:
            for i in range(len(self.buckets_sizes)):
                for tag in ['x', 'length']:
                    data[encode_field('encoder', lang, i, tag)] = self.encoder[lang][i][tag]
                for tag in ['x', 'y', 'length']:
                    data[encode_field('decoder', lang, i, tag)] = self.decoder[lang][i][tag]
                    
        np.savez(filename, **data)
                
        with open(filename + '.pcl', 'wb') as f:
            pickle.dump(fields, f, pickle.HIGHEST_PROTOCOL)
        
            
    def load(filename):
        
        with open(filename + '.pcl', 'rb') as f:
            d_ = pickle.load(f)
            
        dataset = Dataset.__new__(Dataset)
        dataset.__dict__ = d_
        for tag in ['encoder', 'decoder']:
            dataset.__dict__[tag] = {}
            for lang in ['en', 'fr']:
                dataset.__dict__[tag][lang] = []
                for i in range(len(dataset.buckets_sizes)):
                    dataset.__dict__[tag][lang].append({})
        
        data = np.load(filename + '.npz')
        for k, v in data.items():
            fields = decode_field(k)
            dataset.__dict__[fields[0]][fields[1]][fields[2]][fields[3]] = v
        
        for tag in ['tokenizer', 'detokenizer']:
            dataset.__dict__[tag] = {}
        for lang in ['en', 'fr']:
            dataset.tokenizer[lang] = MosesTokenizer(lang=lang)
            dataset.detokenizer[lang] = MosesDetokenizer(lang=lang)
                
                    
        return dataset