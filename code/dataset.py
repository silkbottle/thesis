import numpy as np
import multiprocessing
import re
import pickle
import progressbar
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
    
def multinomial(probs):
    x = np.random.sample()
    s = probs[0]
    i = 0
    while x >= s:
        i += 1
        s += probs[i]
    return i
    

class Dataset(object):
    def __init__(self, english, french, num_words=200000, buckets=[20, 40, 200], nthreads=4, ratio=0.1):
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
        
        for lang in ['en', 'fr']:
            self.__fill_tensors(lang)
            
        self.test_train_split(ratio=ratio)
        
        
    
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
                    
    def __fill_tensors(self, lang):
        
        def bucket_id(length):
            bucket = 0
            while bucket < len(self.buckets_sizes) - 1 and self.buckets_sizes[bucket] < length:
                bucket += 1
            return bucket
        
        def fill_decoder(sentence, bucket, pos):
            end = min(len(sentence), self.buckets_sizes[bucket]) + 1
            self.decoder[lang][bucket]['length'][pos] = end
            self.decoder[lang][bucket]['x'][pos][0] = self.vocab[lang]['<s>']           
            for i in range(1, end):
                self.decoder[lang][bucket]['x'][pos][i] = sentence[i - 1]
                self.decoder[lang][bucket]['y'][pos][i - 1] = sentence[i - 1]
            if end == len(sentence) + 1:
                self.decoder[lang][bucket]['y'][pos][end - 1] = self.vocab[lang]['<\s>']
            else:
                self.decoder[lang][bucket]['y'][pos][end - 1] = sentence[end - 1]
        
        def fill_encoder(sentence, bucket, pos):
            start = max(self.buckets_sizes[bucket] - len(sentence), 0)
            for i in range(start, self.buckets_sizes[bucket]): 
                self.encoder[lang][bucket]['x'][pos][i] = sentence[i - start]
                self.encoder[lang][bucket]['length'][pos] = self.buckets_sizes[bucket] - start
        
        
        with open(self.source[lang], 'r') as f:
            buffer = []
            counters = np.zeros(len(self.buckets_sizes), dtype=np.int)
            index = 0
            for line in f:
                buffer.append(line)
                if len(buffer) >= self.per_thread * self.nthreads:
                    results = tokenize(self.tokenizer[lang], buffer, self.per_thread)
                    for res in results:
                        sentence = [self.vocab[lang].get(w, self.vocab[lang][unknown(w)]) for w in res]
                        bucket = bucket_id(max(self.lengths['en'][index], self.lengths['fr'][index])) 
                        fill_encoder(sentence, bucket, counters[bucket])
                        fill_decoder(sentence, bucket, counters[bucket])
                        counters[bucket] += 1
                        index += 1
                    buffer = []
                            
            if len(buffer) > 0:
                results = tokenize(self.tokenizer[lang], buffer, self.per_thread)
                for res in results:
                    sentence = [self.vocab[lang].get(w, self.vocab[lang][unknown(w)]) for w in res]
                    bucket = bucket_id(max(self.lengths['en'][index], self.lengths['fr'][index])) 
                    fill_encoder(sentence, bucket, counters[bucket])
                    fill_decoder(sentence, bucket, counters[bucket])
                    counters[bucket] += 1
                    index += 1
                        
    
    def test_train_split(self, ratio=0.01, seed=42):
        np.random.seed(seed)
        
        self.random_seed = seed
        self.ratio = ratio
        
        self.train = {}
        self.test = {}
        for tag in ['encoder', 'decoder']:
            self.train[tag] = {}
            self.test[tag] = {}
            for lang in ['en', 'fr']:
                self.train[tag][lang] = []
                self.test[tag][lang] = []
                
        for i in range(len(self.buckets_sizes)):
            size = self.encoder['en'][i]['x'].shape[0]
            test_size = int(ratio * size)
            perm = np.random.permutation(size)
            for lang in ['en', 'fr']:
                self.train['encoder'][lang].append({})
                self.test['encoder'][lang].append({})
                for tag in ['x', 'length']:
                    self.train['encoder'][lang][i][tag] = self.encoder[lang][i][tag][perm[test_size:]]
                    self.test['encoder'][lang][i][tag] = self.encoder[lang][i][tag][perm[:test_size]]
                self.train['decoder'][lang].append({})
                self.test['decoder'][lang].append({})
                for tag in ['x', 'y', 'length']:
                    self.train['decoder'][lang][i][tag] = self.decoder[lang][i][tag][perm[test_size:]]
                    self.test['decoder'][lang][i][tag] = self.decoder[lang][i][tag][perm[:test_size]]
    
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
    
    def iterate_minibatches(self, batch_size, mode='train', show=True):
        if mode == 'train':
            data = self.train
        if mode == 'test':
            data = self.test
            
        perms = []
        idxs = []
        sizes = []
        fullsize = 0
        for i in range(len(self.buckets_sizes)):
            sizes.append(data['encoder']['en'][i]['x'].shape[0])
            perms.append(np.random.permutation(sizes[i]))
            fullsize += data['encoder']['en'][i]['x'].size
            
        sizes = np.array(sizes)
        probs = sizes / sizes.sum()
        idxs = np.zeros(sizes.size, dtype=np.int)
        mask = np.ones(sizes.size, dtype=np.bool)
        
        if show:
            bar = progressbar.ProgressBar(max_value=fullsize)
            progress = 0
        finish = False
        while not finish:
            bucket = multinomial(probs)
            start = idxs[bucket]
            end = idxs[bucket] + batch_size
            if end > sizes[bucket]:
                mask[bucket] = 0
                if mask.sum() == 0:
                    finish = True
                else:
                    probs = np.multiply(probs, mask)
                    probs /= probs.sum()
                end = sizes[bucket]
            idxs[bucket] = end
            
            if show:
                progress += data['encoder']['en'][bucket]['x'][perms[bucket][start:end]].size
                print(progress)
                bar.update(progress)
                
            yield data['encoder']['en'][bucket]['x'][perms[bucket][start:end]], \
                data['encoder']['en'][bucket]['length'][perms[bucket][start:end]], \
                data['encoder']['fr'][bucket]['x'][perms[bucket][start:end]], \
                data['encoder']['fr'][bucket]['length'][perms[bucket][start:end]]

    def save(self, filename):
        fields = {}
        for k, v in self.__dict__.items():
            if not k in ['encoder', 'decoder', 'tokenizer', 'detokenizer', 'train', 'test']:
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
            d = pickle.load(f)
            
        dataset = Dataset.__new__(Dataset)
        dataset.__dict__ = d
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
            
        if 'random_seed' in d and 'ratio' in d:
            dataset.test_train_split(ratio=d['ratio'], seed=d['random_seed'])       
        return dataset