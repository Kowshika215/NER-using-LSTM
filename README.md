# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

The objective of this project is to develop a Bidirectional Long Short-Term Memory (BiLSTM) model for Name Recognition in sentences. The model will perform Named Entity Recognition (NER) by identifying and classifying person names within input text sequences. Given a sentence, the model should label each word using sequence tagging (e.g., B-PER, I-PER, O) to determine whether it is part of a person’s name or not. The performance of the model will be evaluated using metrics such as accuracy and validation loss.

<img width="1143" height="272" alt="image" src="https://github.com/user-attachments/assets/1270b143-e849-4adc-aaa6-0c978a3cf932" />


## DESIGN STEPS

Import the necessary libraries.

### Step 2:
Load the dataset and use DataLoader to batch the dataset

### Step 3:
Create a class to define the Long Short Term Memory Neural Network, in the class define the forward function

### Step 4:
Initialize the model and get a model summary

### STEP 5:
Initialize the loss function MSELoss and Optimizier

### STEP 6:
Create a function to train the model and call it to train the model.

### STEP 7:
Test the model using the test_loader.

### Step 8:
Display the results.

## PROGRAM
### Name:Kowshika R
### Register Number: 212224220049
```
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128):
        super(BiLSTMTagger, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=word2idx["ENDPAD"])

        # BiLSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        lstm_out, _ = self.lstm(x)
        outputs = self.fc(lstm_out)
        return outputs
        


model = BiLSTMTagger(len(word2idx), len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        # TRAINING
        model.train()
        total_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids)

            loss = loss_fn(
                outputs.view(-1, outputs.shape[-1]),
                labels.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # VALIDATION
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in test_loader:

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)

                loss = loss_fn(
                    outputs.view(-1, outputs.shape[-1]),
                    labels.view(-1)
                )

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses


```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="879" height="630" alt="image" src="https://github.com/user-attachments/assets/c357b506-8df2-4ece-a8ae-5eaca05f57b3" />


### Sample Text Prediction

<img width="444" height="610" alt="image" src="https://github.com/user-attachments/assets/b7630d00-aa56-4c62-8bc0-f90d642d5792" />


## RESULT
Thus, A Long Short Term Memory Neural Network model is implemented successfully for recognizing the named entities in the text.
