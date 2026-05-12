import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import gensim


def val(model, val_loader, criterion, device):
    """
    Inputs:
    model (torch.nn.Module): The forecasting model to evaluate.
    val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    criterion (torch.nn.Module): Loss function (e.g. MSE) to compute the validation loss.

    Outputs:
    Mean validation loss across all examples.
    """
    model.eval()
    val_running_loss = 0.0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            bs = inputs.size(0)
            val_running_loss += loss.item() * bs
            total += bs

    return val_running_loss / total

def train(model, train_loader, val_loader, criterion, epochs, optimizer, device):
    """
    Inputs:
    model (torch.nn.Module): The forecasting model to train.
    train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    criterion (torch.nn.Module): Loss function (e.g. MSE) to compute the training loss.
    epochs: Number of epochs to train for.
    optimizer: The optimizer to use during training.

    Outputs:
    Tuple of (train_loss_arr, val_loss_arr) holding per-epoch mean losses.
    """
    train_loss_arr = []
    val_loss_arr = []
    print("train start", flush=True)

    for epoch in range(epochs):
        print(
            "epoch:",
            epoch + 1,
            flush=True
        )
        model.train()
        running_loss = 0.0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx % 220 == 0:
                print(
                    f"epoch {epoch + 1}, batch {batch_idx}/{len(train_loader)}",
                    flush=True,
                )

            inputs = inputs.float().to(device, non_blocking=True)
            targets = targets.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


            bs = inputs.size(0)
            running_loss += loss.item() * bs
            total += bs

        train_loss = running_loss / total
        val_loss = val(model, val_loader, criterion, device)
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)
        print(
            "train loss:",
            round(train_loss, 6),
            "val loss:",
            round(val_loss, 6),
        )

    print("Training finished.")

    return train_loss_arr, val_loss_arr


# # TODO 8: Understand this code.
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_length):
#         """
#         Inputs:
#         d_model: The dimension of the embeddings.
#         max_seq_length: Maximum length of sequences input into the transformer.
#         """
#         super(PositionalEncoding, self).__init__()

#         pe = torch.zeros(max_seq_length, d_model)
#         position = torch.arange(0, max_seq_length, dtype=torch.float).reshape(max_seq_length, 1)
#         div_term = torch.exp( 
#               -1 * (torch.arange(0, d_model, 2).float()/d_model) * math.log(10000.0)
#         )

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         self.register_buffer("pe", pe.unsqueeze(0))

#     def forward(self, x):
#         """
#         Adds the positional encoding to the model input x.
#         """
#         return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_dropout=0.0):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: The number of attention heads to use.
        attn_dropout: Dropout probability applied to attention weights.
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads


        # TODO 9.1: define layers W_q, W_k, W_v, and W_o
        # Hint: Recall that linear layers essentially perform matrix multiplication
        #       between the layer input and layer weights

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)

        #################

    def split_heads(self, x):
        """
        Reshapes Q, K, V into multiple heads.
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)

    def compute_attention(self, Q, K, V):
        """
        Returns single-headed attention between Q, K, and V.
        """
        # TODO 9.2: compute attention using the attention equation provided above
        score = Q @ K.transpose(-2,-1)/ math.sqrt(self.d_k)
        attn = torch.softmax(score, -1)
        attn = self.attn_dropout(attn)
        attention = attn @ V
        #################
        
        return attention

    def combine_heads(self, x):
        """
        Concatenates the outputs of each attention head into a single output.
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, x):
        # TODO: 9.3 implement forward pass

        #################

        x = x.float()
        batch_size = x.size(0)

        Q = self.W_q(x)  
        K = self.W_k(x)   
        V = self.W_v(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_heads = self.compute_attention(Q,K,V)
        multi_head = self.W_o(self.combine_heads(attention_heads))

        return multi_head

        



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        d_ff: Hidden dimension size for the feed-forward network.
        dropout: Dropout probability applied between the two linear layers.
        """
        super(FeedForward, self).__init__()

        # TODO 10: define the network
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        #################

    def forward(self, x):
        # TODO 10: implement feed forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #################

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, p, attn_dropout=0.0):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        d_ff: Hidden dimension size for the feed-forward network.
        p: Dropout probability applied to the sublayer outputs and inside the FFN.
        attn_dropout: Dropout probability applied to attention weights.
        """
        super(EncoderLayer, self).__init__()

        # TODO 11: define the encoder layer
        self.self_attn = MultiHeadAttention(d_model, num_heads, attn_dropout=attn_dropout)

        self.norm1 = nn.BatchNorm1d(d_model)

        self.feed_forward = FeedForward(d_model, d_ff, dropout=p)

        self.norm2 = nn.BatchNorm1d(d_model)

        self.dropout_attn = nn.Dropout(p)
        self.dropout_ffn = nn.Dropout(p)

        #################

    def forward(self, x):

        ## TODO 11: implement the forward function based on the architecture described above
        attn_output = self.self_attn(x)
        x = self.norm1((x + self.dropout_attn(attn_output)).permute(0,2,1)).permute(0,2,1)

        ff_output = self.feed_forward(x)
        x = self.norm2((x + self.dropout_ffn(ff_output)).permute(0,2,1)).permute(0,2,1)
        #################

        return x


class PatchTST(nn.Module):
    def __init__(
        self, num_prediction, d_model, num_heads, num_layers, d_ff,
        lookback_window, patch_size, patch_overlap,
        dropout=0.2, attn_dropout=0.0,
    ):
        """
        Inputs:
        num_prediction: Number of predictions (T in paper)
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        num_layers: Number of encoder layers.
        d_ff: Hidden dimension size for the feed-forward network.
        lookback_window: Length of the lookback window (L in paper).
        self.num_patch: Number of patches.
        patch_size: Size of each patch.
        """
        super(PatchTST, self).__init__()

        # TODO 12: define the transformer
        self.num_classes = num_prediction
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.lookback_window = lookback_window
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.num_patch = ((lookback_window - patch_size) // patch_overlap) + 2
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        # self.dropout = nn.Dropout(p)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout, attn_dropout=attn_dropout)
             for _ in range(num_layers)]
            )

        # self.relu = nn.ReLU()

        self.fc = nn.Linear(d_model * self.num_patch, num_prediction)

        self.W_p = nn.Linear(patch_size, d_model)
        self.W_pos = nn.Embedding(self.num_patch, d_model)
        ################_

    def forward(self, x):

        """
        x will be of shape (batch_size,num_series,lookback_window) 
        where batch_size is batch size, num_series is the number of time series,
        and lookback_window is the length of each lookback window.

        We will first apply instance normalization to each lookback window.

        Then we divide each lookback window into patches of size patch_size. 
        This would make x of shape (batch_size, num_series, num_patch, patch_size). 
        
        We will then apply a linear transformation W_p to each patch to project it into the embedding space, 
        and add positional encoding using W_pos. This would make x of shape (batch_size, num_series, num_patch, d_model). 
        
        We will then pass the embedded patches through the encoder layers. 
        
        Finally, we will flatten the output and pass it through a fully connected layer to get the 
        final output of shape (batch_size, num_series, num_prediction).
        """

        ## TODO 12: implement the forward pass
        x = x.float()
        batch_size, num_series, _ = x.shape

        ## Instance norm
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-5
        x = (x - mean) / std

        ## Divide into patches, now x has shape (batch_size, num_series, self.num_patch, patch_size)
        last = x[:,:,-1:]
        padding = last.expand(-1,-1,self.patch_overlap)
        x = torch.cat((x, padding), dim=-1)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_overlap)


        # Project into embedding space and add positional encoding,
        # now x has shape (batch_size, num_series, self.num_patch, d_model)
        x = self.W_p(x) + self.W_pos(torch.arange(self.num_patch).to(x.device)).unsqueeze(0).unsqueeze(0)

        x = x.reshape(batch_size * num_series, self.num_patch, self.d_model)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.flatten(start_dim=1)
        x = self.fc(x)

        # Reshape back to (batch_size, num_series, num_prediction) and reverse instance norm
        x = x.reshape(batch_size, num_series, self.num_classes)
        x = x * std + mean
        #################

        return x


def process_batch(bert_model, data, criterion, device, val=False):
    """
    Inputs:
    data: The data in the batch to process.
    criterion: The loss function.
    val: True if processing a batch from the validation or test set.
         False if processing a batching from the training set.

    Outputs:
    Tuple of (outputs, losses)
        outputs: a dictionary containing the model outputs ('out') and predicted labels ('preds')
        metrics: a dictionary containing the model loss over the batch ('loss') and during validation (val = True),
                 the total number of examples in the batch ('batch_size') and the total number of examples whose
                 label the model predicted correctly ('num_correct')
    """

    outputs, metrics = dict(), dict()

    # TODO 13: process batch
    input_ids = data["source_ids"].to(device)
    attention_mask = data["source_mask"].to(device)
    labels = data["label"].to(device)

    out = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits

    preds = torch.argmax(logits, dim=1)
    loss = criterion(logits, labels)

    outputs["out"] = logits
    outputs["preds"] = preds

    metrics["loss"] = loss

    if val:
        metrics["batch_size"] = labels.size(0)
        metrics["num_correct"] = torch.eq(preds, labels).sum().item()
    # Hint: For details on what information the data from the data loader contains
    #       check the __getitem__ function defined in the CustomClassDataset implemented
    #       at the beginning of Part 5
    # Hint: Make sure to send the data to the same device that the model is on.
    #################

    return outputs, metrics

