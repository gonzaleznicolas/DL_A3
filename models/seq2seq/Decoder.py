"""
S2S Decoder model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN", attention=False):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type
        self.attention = attention

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer based on the "model_type" argument.            #
        #          Supported types (strings): "RNN", "LSTM". Instantiate the        #
        #          appropriate layer for the specified model_type.                  #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #       5) If attention is True, A linear layer to downsize concatenation   #
        #           of context vector and input                                     #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################

        # 1) Embedding layer
        self.embedding = nn.Embedding(output_size, emb_size)
        
        # 2) Recurrent layer based on model_type
        if model_type == "RNN":
            self.rnn = nn.RNN(emb_size, decoder_hidden_size, batch_first=False)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(emb_size, decoder_hidden_size, batch_first=False)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        # 3) Linear layer for output with log softmax
        self.out = nn.Linear(decoder_hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        # 4) Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # 5) Linear layer for attention (to downsize concatenation of context and input)
        if attention:
            self.attention_linear = nn.Linear(encoder_hidden_size + emb_size, emb_size)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim)
            Returns:
                attention: attention probabilities (dimension: N,1,T)
        """

        #############################################################################
        #                              BEGIN YOUR CODE                              #
        # It is recommended that you implement the cosine similarity function from  #
        # the formula given in the docstring. This exercise will build up your     #
        # skills in implementing mathematical formulas working with tensors.        #
        # Alternatively you may use nn.torch.functional.cosine_similarity or        #
        # some other similar function for your implementation.                      #
        #############################################################################

        # hidden: (1, N, hidden_dim)
        # encoder_outputs: (N, T, hidden_dim)
        
        # Remove the sequence dimension from hidden: (1, N, hidden_dim) -> (N, hidden_dim)
        hidden = hidden.squeeze(0)  # (N, hidden_dim)
        
        # Compute q @ K^T
        # hidden: (N, hidden_dim), encoder_outputs: (N, T, hidden_dim)
        # We need to compute dot product: (N, hidden_dim) @ (N, hidden_dim, T) -> (N, T)
        scores = torch.bmm(hidden.unsqueeze(1), encoder_outputs.transpose(1, 2))  # (N, 1, T)
        
        # Compute magnitudes
        # |q|: (N, 1)
        q_norm = torch.norm(hidden, dim=1, keepdim=True)  # (N, 1)
        
        # |K|: (N, T)
        k_norm = torch.norm(encoder_outputs, dim=2)  # (N, T)
        
        # Compute cosine similarity: scores / (|q| * |K|)
        # Expand q_norm to match dimensions: (N, 1) -> (N, 1, 1) then broadcast
        cosine_sim = scores / (q_norm.unsqueeze(2) * k_norm.unsqueeze(1) + 1e-8)  # (N, 1, T)
        
        # Apply softmax to get attention probabilities
        attention_prob = torch.softmax(cosine_sim, dim=2)  # (N, 1, T)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return attention_prob

    def forward(self, input, hidden, encoder_outputs=None):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden state of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the state coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
                where N is the batch size, T is the sequence length
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       1) Apply the dropout to the embedding layer                         #
        #                                                                           #
        #       2) If attention is true, compute the attention probabilities and    #
        #       use them to do a weighted sum on the encoder_outputs to determine   #
        #       the context vector. The context vector is then concatenated with    #
        #       the output of the dropout layer and is fed into the linear layer    #
        #       you created in the init section. The output of this layer is fed    #
        #       as input vector to your recurrent layer. Refer to the diagram       #
        #       provided in the Jupyter notebook for further clarifications. note   #
        #       that attention is only applied to the hidden state of LSTM.         #
        #                                                                           #
        #       3) Apply linear layer and log-softmax activation to output tensor   #
        #       before returning it.                                                #
        #                                                                           #
        #       If model_type is LSTM, the hidden variable returns a tuple          #
        #       containing both the hidden state and the cell state of the LSTM.    #
        #############################################################################

        # input: (N, 1)
        # 1a) Embed the input
        embedded = self.embedding(input)  # (N, 1, emb_size)

        # 1b) Apply dropout
        dropped_out_embedded = self.dropout(embedded) # (N, 1, emb_size)
        
        # 2) Handle attention if enabled
        if self.attention and encoder_outputs is not None:
            # Get the hidden state for attention computation
            # For LSTM, hidden is a tuple (h, c), we use h for attention
            if self.model_type == "LSTM":
                h_state = hidden[0]  # (1, N, decoder_hidden_size)
            else:
                h_state = hidden  # (1, N, decoder_hidden_size)
            
            # Compute attention probabilities
            attention_weights = self.compute_attention(h_state, encoder_outputs)  # (N, 1, T)
            
            # Compute context vector as weighted sum of encoder outputs
            # attention_weights: (N, 1, T), encoder_outputs: (N, T, encoder_hidden_size)
            context = torch.bmm(attention_weights, encoder_outputs)  # (N, 1, encoder_hidden_size)
            
            # Concatenate context with dropped_out_embedded input
            # dropped_out_embedded: (N, 1, emb_size), context: (N, 1, encoder_hidden_size)
            combined = torch.cat((context, dropped_out_embedded), dim=2)  # (N, 1, encoder_hidden_size + emb_size)
            
            # Pass through linear layer to downsize to emb_size
            rnn_input = self.attention_linear(combined)  # (N, 1, emb_size)
        else:
            rnn_input = dropped_out_embedded  # (N, 1, emb_size)
        
        # Reshape for RNN: (seq_len=1, N, emb_size)
        rnn_input = rnn_input.transpose(0, 1)  # (1, N, emb_size)
        
        # Pass through RNN/LSTM
        rnn_output, hidden = self.rnn(rnn_input, hidden)  # rnn_output: (1, N, decoder_hidden_size)
        
        # Reshape back: (1, N, decoder_hidden_size) -> (N, decoder_hidden_size)
        rnn_output = rnn_output.squeeze(0)  # (N, decoder_hidden_size)
        
        # 3) Apply linear layer and log-softmax
        output = self.out(rnn_output)  # (N, output_size)
        output = self.log_softmax(output)  # (N, output_size)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden
