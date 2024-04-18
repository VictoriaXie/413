'''
for data processing, all processed files are automatically saved
'''
import pandas as pd
import os
import nltk
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec

def process_csv():
    compression_opts = dict(method='zip',
                        archive_name='out.csv')
    cols = ['target', 'ids', 'date', 'flag', 'user', 'text']
    human_preprocess = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', encoding='latin-1', names=cols, header=None)
    # process human
    human_processed = human_preprocess[human_preprocess['date'].str[-4:].astype(int) < 2017]
    human_processed = pd.DataFrame(human_processed['text'],columns=['text'])
    human_df = human_df.dropna()
    sampled_human = human_df.sample(frac=0.01)
    # process ai
    ai_df = pd.DataFrame({'text':[]})
    for filename in os.listdir('/content/ai_stuff'):
        if(filename[0] == '.'):
            continue # there's a '.ipynb_checkpoints' file in here for some reason
        temp = pd.read_csv('/content/ai_stuff/'+filename, names=['text'], header=None)
        ai_df = pd.concat([ai_df,temp])
    ai_df = ai_df.dropna()
    # add binary "did human write it" label
    human_processed['human_wrote'] = 1
    ai_df['human_wrote'] = 0
    # human csv is large so we save it to zip we save it in case want to resample later
    human_processed.to_csv('/content/human_processed.zip', index=False,
          compression=compression_opts)
    ai_df.to_csv('/content/ai_processed.csv', index=False)
    sampled_human.to_csv('/content/sampled_human.csv', index=False)
    


def baseline_tokenize():
    '''
    saves tokenized tweets into respective csvs
    return total_sentences that contains both AI and human tokenized tweets 
    '''
    nltk.download('stopwords')
    tokenizer = TweetTokenizer()
    stop_words = set(stopwords.words('english'))
    ai_df = pd.read_csv('/content/ai_processed.csv')
    sampled_human = pd.read_csv('/content/sampled_human.csv')
    ai_sentences = []
    for _, line in ai_df.iterrows():
        tokenized = tokenizer.tokenize(line['text'])
        filtered_sentence = [w for w in tokenized if not w.lower() in stop_words]
        ai_sentences.append(filtered_sentence)
    human_sentences = []
    for _, line in sampled_human.iterrows():
        tokenized = tokenizer.tokenize(line['text'])
        filtered_sentence = [w for w in tokenized if not w.lower() in stop_words]
        human_sentences.append(filtered_sentence)
    total_sentences = ai_sentences + human_sentences

    # save to csv, both indexed and non_indexed
    out_human = pd.DataFrame({'text':human_sentences, 'human_wrote':1})
    out_ai = pd.DataFrame({'text':ai_sentences, 'human_wrote':0})
    out_human.to_csv('/content/human_token.csv', index=False)
    out_ai.to_csv('/content/ai_token.csv', index=False)
    out_human.to_csv('/content/human_token_index.csv', index=True)
    out_ai.to_csv('/content/ai_token_index.csv', index=True)

    return total_sentences

def train_w2v():
    total_sentences = baseline_tokenize()
    model = gensim.models.Word2Vec(
            total_sentences,
            window=5, # context window size
            min_count=2, # Ignores all words with total frequency lower than 2.
            sg = 1, # 1 for skip-gram model
            hs = 0,
            epochs=10,
            negative = 10, # for negative sampling
            workers= 32, # no.of cores
            seed = 34
    )
    model.train(total_sentences, total_examples=len(total_sentences), epochs=model.epochs)
    model.save('/content/w2vmodel.model')