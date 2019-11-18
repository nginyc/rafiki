import nltk
from sklearn.model_selection import train_test_split
import pickle

# download nltk Treebank dataset & Universal tagset (https://universaldependencies.org/u/pos/) to your nltk package/nltk_data
nltk.download('treebank')
nltk.download('universal_tagset')

#split train & test set
tagged_sentence = nltk.corpus.treebank.tagged_sents(tagset='universal')
train_set, test_set = train_test_split(tagged_sentence,test_size=0.2,random_state=1234)

#save data set to local folder in pickle file
with open("data/ptb_train.txt", "wb") as fp:
    pickle.dump(train_set, fp)

with open("data/ptb_test.txt", "wb") as fp:
    pickle.dump(test_set, fp)

    