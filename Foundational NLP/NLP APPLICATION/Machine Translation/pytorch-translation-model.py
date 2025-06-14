import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# Create a small dummy dataset of Twi-English sentence pairs
twi_sentences = [
    "Me din de John",
    "Ɛte sɛn?",
    "Meda wo ase",
    "Me pɛ sɛ me kɔ fie",
    "Ɛhɔ yɛ fɛ",
    "Me dɔ wo",
    "Aduane no yɛ dɛ",
    "Nnipa no reba",
    "Mepɛ sɛ mesua Twi",
    "Ɛnnɛ yɛ Memeneda"
]

english_sentences = [
    "My name is John",
    "How are you?",
    "Thank you",
    "I want to go home",
    "It is beautiful there",
    "I love you",
    "The food is delicious",
    "The people are coming",
    "I want to learn Twi",
    "Today is Saturday"
]

# Tokenization functions
def tokenize(sentences):
    # Create vocabulary
    vocab = set()
    for sentence in sentences:
        for word in sentence.lower().split():
            vocab.add(word)
    
    # Add special tokens
    vocab.add('<pad>')
    vocab.add('<sos>')
    vocab.add('<eos>')
    vocab.add('<unk>')
    
    # Create word-to-index and index-to-word mappings
    word2idx = {word: idx for idx, word in enumerate(sorted(vocab))}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return word2idx, idx2word

# Create tokenizers
twi_word2idx, twi_idx2word = tokenize(twi_sentences)
eng_word2idx, eng_idx2word = tokenize(english_sentences)

# Get vocabulary sizes
twi_vocab_size = len(twi_word2idx)
eng_vocab_size = len(eng_word2idx)

# Convert sentences to indices
def sentence_to_indices(sentence, word2idx):
    return [word2idx.get(word.lower(), word2idx['<unk>']) for word in sentence.split()]

# Prepare dataset
class TranslationDataset(Dataset):
    def __init__(self, twi_sentences, eng_sentences, twi_word2idx, eng_word2idx):
        self.twi_data = []
        self.eng_data = []
        
        for twi_sentence, eng_sentence in zip(twi_sentences, eng_sentences):
            twi_indices = sentence_to_indices(twi_sentence, twi_word2idx)
            eng_indices = [eng_word2idx['<sos>']] + sentence_to_indices(eng_sentence, eng_word2idx) + [eng_word2idx['<eos>']]
            
            self.twi_data.append(twi_indices)
            self.eng_data.append(eng_indices)
    
    def __len__(self):
        return len(self.twi_data)
    
    def __getitem__(self, idx):
        return self.twi_data[idx], self.eng_data[idx]

# Pad sequences in a batch
def collate_fn(batch):
    twi_batch, eng_batch = [], []
    twi_lengths, eng_lengths = [], []
    
    for twi_item, eng_item in batch:
        twi_batch.append(twi_item)
        eng_batch.append(eng_item)
        twi_lengths.append(len(twi_item))
        eng_lengths.append(len(eng_item))
    
    # Pad sequences
    max_twi_len = max(twi_lengths)
    max_eng_len = max(eng_lengths)
    
    twi_padded = []
    eng_padded = []
    
    for twi_item, eng_item in zip(twi_batch, eng_batch):
        twi_padded.append(twi_item + [twi_word2idx['<pad>']] * (max_twi_len - len(twi_item)))
        eng_padded.append(eng_item + [eng_word2idx['<pad>']] * (max_eng_len - len(eng_item)))
    
    return torch.LongTensor(twi_padded), torch.LongTensor(eng_padded), torch.LongTensor(twi_lengths), torch.LongTensor(eng_lengths)

# Create dataset and dataloader
dataset = TranslationDataset(twi_sentences, english_sentences, twi_word2idx, eng_word2idx)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

# Define the Attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: (1, batch_size, hidden_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Repeat hidden state seq_len times
        hidden = hidden.repeat(seq_len, 1, 1).permute(1, 0, 2)
        
        # Calculate attention scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return torch.softmax(attention, dim=1)

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, attention):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.attention = attention
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(hidden_size + embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x, hidden, cell, encoder_outputs):
        # x: (batch_size, 1)
        # hidden: (1, batch_size, hidden_size)
        # cell: (1, batch_size, hidden_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        
        x = x.unsqueeze(1)  # (batch_size, 1)
        
        embedded = self.embedding(x)  # (batch_size, 1, embedding_size)
        
        # Calculate attention weights
        attention_weights = self.attention(hidden, encoder_outputs)  # (batch_size, seq_len)
        attention_weights = attention_weights.unsqueeze(1)  # (batch_size, 1, seq_len)
        
        # Apply attention weights to encoder outputs
        context = torch.bmm(attention_weights, encoder_outputs)  # (batch_size, 1, hidden_size)
        
        # Combine embedded input and context vector
        lstm_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, embedding_size + hidden_size)
        
        # Pass through LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Combine output and context for prediction
        output = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)  # (batch_size, hidden_size * 2)
        prediction = self.fc(output)  # (batch_size, output_size)
        
        return prediction, hidden, cell

# Define the Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (batch_size, src_len)
        # trg: (batch_size, trg_len)
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode source sequence
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # First decoder input is the <sos> token
        decoder_input = trg[:, 0]
        
        for t in range(1, trg_len):
            # Pass through decoder
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            
            # Store output
            outputs[:, t, :] = output
            
            # Teacher forcing: use actual target token as next input
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # If teacher forcing, use actual token; otherwise use predicted
            decoder_input = trg[:, t] if teacher_force else top1
            
        return outputs

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
EMBEDDING_SIZE = 32
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

# Initialize model components
encoder = Encoder(twi_vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE).to(device)
attention = Attention(HIDDEN_SIZE).to(device)
decoder = Decoder(eng_vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, attention).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=eng_word2idx['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg, src_len, trg_len) in enumerate(dataloader):
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # Calculate loss (ignore the first token which is <sos>)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss / len(dataloader):.4f}")

# Save the model
torch.save({
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
    'twi_word2idx': twi_word2idx,
    'twi_idx2word': twi_idx2word,
    'eng_word2idx': eng_word2idx,
    'eng_idx2word': eng_idx2word,
}, 'twi_to_english_model.pt')

# Function to translate a sentence
def translate_sentence(model, sentence, twi_word2idx, twi_idx2word, eng_word2idx, eng_idx2word, device, max_length=50):
    model.eval()
    
    # Convert sentence to indices
    tokens = sentence.split()
    indices = [twi_word2idx.get(token.lower(), twi_word2idx['<unk>']) for token in tokens]
    
    # Convert to tensor
    src_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
    
    # Encode the source sentence
    encoder_outputs, hidden, cell = model.encoder(src_tensor)
    
    # Start with <sos> token
    trg_idx = [eng_word2idx['<sos>']]
    
    for _ in range(max_length):
        trg_tensor = torch.LongTensor([trg_idx[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
        
        # Get predicted token
        pred_token = output.argmax(1).item()
        trg_idx.append(pred_token)
        
        # If <eos> token is predicted, end translation
        if pred_token == eng_word2idx['<eos>']:
            break
    
    # Convert indices to tokens (excluding <sos> and <eos>)
    translated = [eng_idx2word[idx] for idx in trg_idx if idx not in [eng_word2idx['<sos>'], eng_word2idx['<eos>'], eng_word2idx['<pad>']]]
    
    return ' '.join(translated)

# Test the model with some examples
print("\nTesting the model:")
for i in range(len(twi_sentences)):
    twi_sent = twi_sentences[i]
    eng_sent = english_sentences[i]
    
    translation = translate_sentence(model, twi_sent, twi_word2idx, twi_idx2word, eng_word2idx, eng_idx2word, device)
    
    print(f"Twi: {twi_sent}")
    print(f"Expected: {eng_sent}")
    print(f"Translated: {translation}")
    print("-" * 50)
