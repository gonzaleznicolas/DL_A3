import random

""" 			  		 			     			  	   		   	  			  	
Seq2Seq model.  (c) 2021 Georgia Tech

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

import torch
import torch.nn as nn
import torch.optim as optim


# import custom models


class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the Seq2Seq model. You should use .to(device) to make sure  #
        #    that the models are on the same device (CPU/GPU). This should take no  #
        #    more than 2 lines of code.                                             #
        #############################################################################

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, source):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
        """

        batch_size = source.shape[0]
        seq_len = source.shape[1]
        #############################################################################
        # TODO:                                                                     #
        #   Implement the forward pass of the Seq2Seq model. Please refer to the    #
        #   following steps:                                                        #
        #       1) Get the last hidden representation from the encoder. Use it as   #
        #          the first hidden state of the decoder                            #
        #       2) The first input for the decoder should be the <sos> token, which #
        #          is the first in the source sequence.                             #
        #       3) Feed this first input and hidden state into the decoder          #  
        #          one step at a time in the sequence, adding the output to the     #
        #          final outputs.                                                   #
        #       4) Update the input and hidden being fed into the decoder           #
        #          at each time step. The decoder output at the previous time step  # 
        #          will have to be manipulated before being fed in as the decoder   #
        #          input at the next time step.                                     #
        #############################################################################

        # 1) Get the last hidden representation from the encoder
        encoder_outputs, hidden = self.encoder(source)
        
        # Initialize outputs tensor with zeros (batch_size, seq_len, decoder_output_size)
        outputs = torch.zeros(batch_size, seq_len, self.decoder.output_size).to(self.device)
        
        # 2) The first input for the decoder is the <sos> token (first token in source)
        decoder_input = source[:, 0].unsqueeze(1)  # (batch_size, 1)
        
        # 3) & 4) Feed through decoder one step at a time
        for t in range(seq_len):
            # Feed input and hidden state into decoder
            if self.decoder.attention:
                output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            else:
                output, hidden = self.decoder(decoder_input, hidden)
            
            # Store output in outputs tensor
            outputs[:, t, :] = output
            
            # 4) Update decoder input for next time step
            # Get the index of the most likely next token
            most_likely_next_token = output.argmax(1)  # (batch_size,)
            decoder_input = most_likely_next_token.unsqueeze(1)  # (batch_size, 1)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs
