
import torch
from utils.dataloader import normalizeString, readLangs, EOS_token, SOS_token
from models.encoder import EncoderRNN
from models.decoder import AttnDecoderRNN
from utils.config import *
import random

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data and initialize
input_lang, output_lang, pairs = readLangs("eng", "spa", "english_spanish_10000.csv")
encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)
decoder = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words).to(device)

# Load saved models
encoder.load_state_dict(torch.load("encoder.pt", map_location=device))
decoder.load_state_dict(torch.load("decoder.pt", map_location=device))
encoder.eval()
decoder.eval()

# Convert sentence to tensor
def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index.get(word, 0) for word in sentence.split()]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# Translate a single sentence
def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        sentence = normalizeString(sentence)
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden().to(device)

        encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            next_word_index = topi.item()

            if next_word_index == EOS_token:
                break

            decoded_words.append(output_lang.index2word.get(next_word_index, "<UNK>"))
            decoder_input = topi.detach().view(1, -1)

        return ' '.join(decoded_words)

# Evaluate n random sentences
def evaluateRandomly(n=5):
    for _ in range(n):
        pair = random.choice(pairs)
        print(f"\n🟡 English: {pair[0]}")
        print(f"🟢 Spanish (Expected): {pair[1]}")
        output = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        print(f"🔵 Spanish (Predicted): {output}")

# Live translation loop
if __name__ == '__main__':
    print("🌐 Translator Ready! Type an English sentence to translate.\n")
    while True:
        input_sentence = input("Enter English sentence (or 'q' to quit): ")
        if input_sentence.lower() == 'q':
            print("❌ Exiting.")
            break
        translation = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
        print("Spanish:", translation)
