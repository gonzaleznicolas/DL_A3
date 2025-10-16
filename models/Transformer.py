"""
Transformer model.  (c) 2021 Georgia Tech

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

import numpy as np

import torch
from torch import nn
import random

####### Do not modify these imports.

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        :param max_length: the maximum length of the input sequences
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        
        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Don't worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        self.embeddingL = nn.Embedding(input_size, hidden_dim)      #initialize word embedding layer
        self.posembeddingL = nn.Embedding(max_length, hidden_dim)   #initialize positional embedding layer

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        
        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        # 
        # Don't forget the layer normalization.                                      #
        ##############################################################################
        self.ff_layer1 = nn.Linear(self.hidden_dim, self.dim_feedforward)
        self.ff_layer2 = nn.Linear(self.dim_feedforward, self.hidden_dim)
        self.norm_ff = nn.LayerNorm(self.hidden_dim)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        self.final = nn.Linear(self.hidden_dim, self.output_size)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling TransformerTranslator class methods here.      #
        #############################################################################
        # Step 1: Embed the inputs
        embeddings = self.embed(inputs)
        
        # Step 2: Apply multi-head attention
        attention_out = self.multi_head_attention(embeddings)
        
        # Step 3: Apply feedforward layer
        ff_out = self.feedforward_layer(attention_out)
        
        # Step 4: Apply final layer
        outputs = self.final_layer(ff_out)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T). (N=batch size, T=sequence length)
        :returns embeddings: floatTensor of shape (N,T,H). (N=batch size, T=sequence length, H=hidden_dim)
        """
        #############################################################################
        # TODO:                                                                     #
        # Deliverable 1: Return the embeddings.                                     #
        # This will take a few lines.                                               #
        #############################################################################
        N, T = inputs.shape
        # Get word embeddings (this is just a lookup operation)
        word_embed = self.embeddingL(inputs)  # (N, T, H)
        
        # Create position indices and get positional embeddings
        positions = torch.arange(T, device=inputs.device).unsqueeze(0).expand(N, T)  # (N, T)
        pos_embed = self.posembeddingL(positions)  # (N, T, H)
        
        # Add word and positional embeddings
        embeddings = word_embed + pos_embed  # (N, T, H)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        
        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################
        # Head 1
        k1 = self.k1(inputs)  # (N, T, dim_k)
        v1 = self.v1(inputs)  # (N, T, dim_v)
        q1 = self.q1(inputs)  # (N, T, dim_q)
        
        # Compute attention scores for head 1
        scores1 = torch.bmm(q1, k1.transpose(1, 2)) / np.sqrt(self.dim_k)  # (N, T, T)
        attention_weights1 = self.softmax(scores1)  # (N, T, T)
        attention1 = torch.bmm(attention_weights1, v1)  # (N, T, dim_v)
        
        # Head 2
        k2 = self.k2(inputs)  # (N, T, dim_k)
        v2 = self.v2(inputs)  # (N, T, dim_v)
        q2 = self.q2(inputs)  # (N, T, dim_q)
        
        # Compute attention scores for head 2
        scores2 = torch.bmm(q2, k2.transpose(1, 2)) / np.sqrt(self.dim_k)  # (N, T, T)
        attention_weights2 = self.softmax(scores2)  # (N, T, T)
        attention2 = torch.bmm(attention_weights2, v2)  # (N, T, dim_v)
        
        # Concatenate the two attention heads
        multi_head = torch.cat([attention1, attention2], dim=2)  # (N, T, dim_v * num_heads)
        
        # Project the concatenated heads
        projected = self.attention_head_projection(multi_head)  # (N, T, H)
        
        # Add & Norm (residual connection + layer normalization)
        outputs = self.norm_mh(inputs + projected)  # (N, T, H)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################
        ff_output = self.ff_layer1(inputs)
        ff_output = nn.ReLU()(ff_output)
        ff_output = self.ff_layer2(ff_output)
        outputs = self.norm_ff(inputs + ff_output)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code. Softmax is not needed here    #
        # as it is integrated as part of cross entropy loss function.               #
        #############################################################################
        outputs = self.final(inputs)
                
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        

class FullTransformerTranslator(nn.Module):

    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2,
                 dim_feedforward=2048, num_layers_enc=2, num_layers_dec=2, dropout=0.2, max_length=43, ignore_index=1):
        super(FullTransformerTranslator, self).__init__()

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.pad_idx=ignore_index

        seed_torch(0)

        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the Transformer Layer          #
        # You should use nn.Transformer                                              #
        ##############################################################################
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers_enc,
            num_decoder_layers=num_layers_dec,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        ##############################################################################
        # TODO:
        # Deliverable 2: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Initialize embeddings in order shown below.                                #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        # Do not change the order for these variables
        self.srcembeddingL = nn.Embedding(input_size, hidden_dim)       #embedding for src
        self.tgtembeddingL = nn.Embedding(output_size, hidden_dim)       #embedding for target
        self.srcposembeddingL = nn.Embedding(max_length, hidden_dim)    #embedding for src positional encoding
        self.tgtposembeddingL = nn.Embedding(max_length, hidden_dim)    #embedding for target positional encoding
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the final layer.               #
        ##############################################################################
        self.final = nn.Linear(hidden_dim, output_size)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, src, tgt):
        """
         This function computes the full Transformer forward pass used during training.
         Put together all of the layers you've developed in the correct order.

         :param src: a PyTorch tensor of shape (N,T) these are tokenized input sentences
                tgt: a PyTorch tensor of shape (N,T) these are tokenized translations
         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the full Transformer stack for the forward pass. #
        #############################################################################
        # shift tgt to right, add one <sos> to the beginning and shift the other tokens to right
        tgt = self.add_start_token(tgt)

        # embed src and tgt for processing by transformer
        N, T_src = src.shape
        _, T_tgt = tgt.shape
        
        # Source embeddings
        src_word_embed = self.srcembeddingL(src)
        src_positions = torch.arange(T_src, device=src.device).unsqueeze(0).expand(N, T_src)
        src_pos_embed = self.srcposembeddingL(src_positions)
        src_embed = src_word_embed + src_pos_embed
        
        # Target embeddings
        tgt_word_embed = self.tgtembeddingL(tgt)
        tgt_positions = torch.arange(T_tgt, device=tgt.device).unsqueeze(0).expand(N, T_tgt)
        tgt_pos_embed = self.tgtposembeddingL(tgt_positions)
        tgt_embed = tgt_word_embed + tgt_pos_embed

        # create target mask and target key padding mask for decoder - Both have boolean values
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_tgt).to(src.device)
        src_key_padding_mask = (src == self.pad_idx)
        tgt_key_padding_mask = (tgt == self.pad_idx)

        # invoke transformer to generate output
        transformer_out = self.transformer(
            src_embed, 
            tgt_embed,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # pass through final layer to generate outputs
        outputs = self.final(transformer_out)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def generate_translation(self, src):
        """
         This function generates the output of the transformer taking src as its input
         it is assumed that the model is trained. The output would be the translation
         of the input

         :param src: a PyTorch tensor of shape (N,T)

         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 5: You will be calling the transformer forward function to    #
        # generate the translation for the input.                                   #
        #############################################################################
        N, T = src.shape
        
        # initially set outputs as a tensor of zeros with dimensions (batch_size, seq_len, output_size)
        outputs = torch.zeros(N, T, self.output_size).to(src.device)
        
        # initially set tgt as a tensor of <pad> tokens with dimensions (batch_size, seq_len)
        tgt = torch.full((N, T), self.pad_idx, dtype=torch.long).to(src.device)
        
        # Add the <sos> tokens taken from src to tgt tensor prior to the loop
        tgt[:, 0] = src[:, 0]
        
        # Generate tokens autoregressively, starting from t=0
        for t in range(T):
            # Call forward with src and current tgt
            current_outputs = self.forward(src, tgt)
            
            # Store the output for position t
            outputs[:, t, :] = current_outputs[:, t, :]
            
            # Get the most likely token for position t and update tgt for next iteration
            if t < T - 1:
                most_likely_token = current_outputs[:, t, :].argmax(dim=1)
                tgt[:, t + 1] = most_likely_token

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def add_start_token(self, batch_sequences, start_token=2):
        """
            add start_token to the beginning of batch_sequence and shift other tokens to the right
            if batch_sequences starts with two consequtive <sos> tokens, return the original batch_sequence

            example1:
            batch_sequence = [[<sos>, 5,6,7]]
            returns:
                [[<sos>,<sos>, 5,6]]

            example2:
            batch_sequence = [[<sos>, <sos>, 5,6,7]]
            returns:
                [[<sos>, <sos>, 5,6,7]]
        """
        def has_consecutive_start_tokens(tensor, start_token):
            """
                return True if the tensor has two consecutive start tokens
            """
            consecutive_start_tokens = torch.tensor([start_token, start_token], dtype=tensor.dtype,
                                                    device=tensor.device)

            # Check if the first two tokens in each sequence are equal to consecutive start tokens
            is_consecutive_start_tokens = torch.all(tensor[:, :2] == consecutive_start_tokens, dim=1)

            # Return True if all sequences have two consecutive start tokens at the beginning
            return torch.all(is_consecutive_start_tokens).item()

        if has_consecutive_start_tokens(batch_sequences, start_token):
            return batch_sequences

        # Clone the input tensor to avoid modifying the original data
        modified_sequences = batch_sequences.clone()

        # Create a tensor with the start token and reshape it to match the shape of the input tensor
        start_token_tensor = torch.tensor(start_token, dtype=modified_sequences.dtype, device=modified_sequences.device)
        start_token_tensor = start_token_tensor.view(1, -1)

        # Shift the words to the right
        modified_sequences[:, 1:] = batch_sequences[:, :-1]

        # Add the start token to the first word in each sequence
        modified_sequences[:, 0] = start_token_tensor

        return modified_sequences

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True