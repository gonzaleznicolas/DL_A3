"""
S2S Encoder model.  (c) 2021 Georgia Tech

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


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the encoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer based on the "model_type" argument.            #
        #          Supported types (strings): "RNN", "LSTM". Instantiate the        #
        #          appropriate layer for the specified model_type.                  #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden states of the Encoder(namely, Linear - ReLU - Linear).    #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #          HINT: the size of the output of the second linear layer must     #
        #          satisfy certain constraint relevant to the decoder.              #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        
        # 1) Embedding layer
        self.embedding = nn.Embedding(input_size, emb_size)
        
        # 2) Recurrent layer
        if model_type == "RNN":
            self.rnn = nn.RNN(emb_size, encoder_hidden_size, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(emb_size, encoder_hidden_size, batch_first=True)
        else:
            raise ValueError("Unsupported model type. Use 'RNN' or 'LSTM'")
        
        # 3) Linear layers with ReLU activation
        self.linear1 = nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        
        # 4) Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder;
                hidden (tensor): the state coming out of the last hidden unit
        """

        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #                                                                           #
        #       Apply tanh activation to the hidden tensor before returning it      #
        #                                                                           #
        #       Do not apply any linear layers/Relu for the cell state when         #
        #       model_type is LSTM before returning it.                             #
        #                                                                           #
        #       If model_type is LSTM, the hidden variable returns a tuple          #
        #       containing both the hidden state and the cell state of the LSTM.    #
        #############################################################################

        # Apply embedding and dropout
        embedded = self.dropout(self.embedding(input))
        
        # Pass through RNN/LSTM
        output, hidden_states = self.rnn(embedded)
        
        # Process hidden states based on model type
        if self.model_type == "RNN":
            # Apply linear layers to hidden state
            hidden = self.linear1(hidden_states)
            hidden = self.relu(hidden)
            hidden = self.linear2(hidden)
            # Apply tanh activation
            hidden = torch.tanh(hidden)
        else:  # LSTM
            # For LSTM, hidden_states is a tuple (hidden_state, cell_state)
            hidden_state, cell_state = hidden_states
            
            # Apply linear layers only to hidden state
            processed_hidden = self.linear1(hidden_state)
            processed_hidden = self.relu(processed_hidden)
            processed_hidden = self.linear2(processed_hidden)
            # Apply tanh activation
            processed_hidden = torch.tanh(processed_hidden)
            
            # Return tuple with processed hidden state and untouched cell state
            hidden = (processed_hidden, cell_state)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden
