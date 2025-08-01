# 📦 Import required modules and utilities
from utils.dataloader import readLangs, EOS_token, SOS_token, normalizeString
from models.encoder import EncoderRNN
from models.decoder import AttnDecoderRNN
from utils.config import *
import torch
import torch.nn as nn
from torch import optim
import random

# 🔧 Set hyperparameters
teacher_forcing_ratio = 0.5           # 50% chance to use teacher forcing
NUM_EPOCHS = 1000                     # Total number of training iterations
LEARNING_RATE = 0.001                # Learning rate for optimizers

# 🔁 Fix randomness for consistent results
torch.manual_seed(42)

# 📚 Load English-Spanish sentence pairs and language metadata
input_lang, output_lang, pairs = readLangs("eng", "spa", "english_spanish_10000.csv")

# # 🧠 Initialize the encoder and decoder models
encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE)
decoder = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words)


# # ⚡ Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
decoder = decoder.to(device)

# # ✏️ Convert sentence to tensor of word indices
def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index.get(word, 0) for word in sentence.split()]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# # 🔁 Training function for a single sentence pair
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden().to(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)

    loss = 0

    # 🔼 Encode input sentence
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # ⬇️ Start decoding
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = random.random() < teacher_forcing_ratio

    if use_teacher_forcing:
        # 👩‍🏫 Use actual target words as next inputs
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di].view(1, -1)
    else:
        # 🤖 Use model’s own predictions
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach().view(1, -1)
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    # 🔙 Backpropagation and optimization Gradients 
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# ⚙️ Set up optimizers and loss function
encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
criterion = nn.NLLLoss()

# 🚀 Start training loop
print("Starting training...\n")
for epoch in range(1, NUM_EPOCHS + 1):
    training_pair = random.choice(pairs)
    input_tensor = tensorFromSentence(input_lang, normalizeString(training_pair[0]))
    target_tensor = tensorFromSentence(output_lang, normalizeString(training_pair[1]))

    loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Loss: {loss:.4f} | Sentence: \"{training_pair[0]}\" -> \"{training_pair[1]}\"")

# 💾 Save the trained models
torch.save(encoder.state_dict(), "encoder.pt")
torch.save(decoder.state_dict(), "decoder.pt")
print("\n✅ Training complete. Models saved!")
