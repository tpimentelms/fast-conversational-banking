PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3


class LanguageDict:
    def __init__(self, name):
        self.name = name
        self.startDict()

        self.pad_idx = PAD_token
        self.sos_idx = SOS_token
        self.eos_idx = EOS_token
        self.unk_idx = UNK_token

    def startDict(self):
        self.word2index = {"PAD": PAD_token, "SOS": SOS_token, "EOS": EOS_token, "UNK": UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.n_words = 4  # Count PAD, SOS, EOS and UNK

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def removeRareWords(self, min_count):
        temp_word2count = self.word2count
        self.startDict()

        for word, count in temp_word2count.items():
            if count >= min_count:
                self.addWord(word)
                self.word2count[word] = count

    def getWordIndex(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.unk_idx
