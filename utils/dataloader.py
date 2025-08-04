import unicodedata
import re
import csv

# Special tokens
SOS_token = 0
EOS_token = 1

# Language class for vocabulary building
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": SOS_token, "EOS": EOS_token}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Normalize string (remove accents, punctuation, lowercase)
def normalizeString(s):
    s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode("utf-8")
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?¿]+", r" ", s)
    return s.lower().strip()

# Read CSV and convert to language object and pairs
def readLangs(lang1, lang2, filepath):
    print("Reading CSV lines...")

    pairs = []
    with open(filepath, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header

        for i, row in enumerate(reader):
            if len(row) < 2:
                print(f"Skipping row {i+2} — not enough columns: {row}")
                continue

            source = row[0].strip()
            target = row[1].strip()

            if not source or not target:
                print(f"Skipping row {i+2} — empty field: {row}")
                continue

            source = normalizeString(source)
            target = normalizeString(target)
            pairs.append([source, target])

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print(f"Total pairs: {len(pairs)}")
    print(f"Input language words: {input_lang.n_words}")
    print(f"Output language words: {output_lang.n_words}")

    return input_lang, output_lang, pairs
