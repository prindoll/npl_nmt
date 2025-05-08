import numpy as np
import torch
import math, time
import torch.nn.functional as functional
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

import modules.constants as const
from utils.misc import no_peeking_mask
from utils.data import generate_language_token
from modules.inference.decode_strategy import DecodeStrategy  # Import DecodeStrategy

class TopKSampling(DecodeStrategy):
    def __init__(self, model, max_len, device, k=50, temperature=0.7):
        """
        Args:
            model: The used model (e.g., LLaMA or similar Transformer).
            max_len: The maximum timestep to be used.
            device: The device to perform calculation.
            k: Number of top tokens to sample from (default: 50).
            temperature: Temperature parameter to control randomness (default: 0.7).
        """
        super(TopKSampling, self).__init__(model, max_len, device)
        self.k = k
        self.temperature = temperature

    def init_vars(self, src, start_token=const.DEFAULT_SOS):
        """
        Initialize the input and encoder output for translation.
        Args:
            src: The batch of sentences (tensor of shape [batch_size, src_len]).
            start_token: The start token (default: <sos>).
        Returns:
            outputs: Initial output tensor with start token [batch_size, 1].
            e_outputs: Encoder outputs [batch_size, src_len, d_model].
        """
        model = self.model
        batch_size = len(src)
        
        init_tok = self.TRG.vocab.stoi[start_token]
        src_mask = (src != self.SRC.vocab.stoi['<pad>']).unsqueeze(-2).to(self.device)
        src = src.to(self.device)

        # Encoder
        e_output = model.encode(src, src_mask)
        outputs = torch.LongTensor([[init_tok] for _ in range(batch_size)]).to(self.device)
        
        return outputs, e_output

    def top_k_sampling(self, src, src_lang=None, trg_lang=None, src_tokens=None, debug=False):
        """
        Generate sequences using top-k sampling.
        Args:
            src: The batch of sentences [batch_size, src_len].
            src_lang: Source language (optional).
            trg_lang: Target language (optional).
            src_tokens: Source tokens in string format (optional).
            debug: If True, print debug information.
        Returns:
            Translated sentences in list-of-tokens format [batch_size, tgt_len].
        """
        model = self.model
        start_token = const.DEFAULT_SOS if trg_lang is None else generate_language_token(trg_lang)
        outputs, e_outputs = self.init_vars(src, start_token=start_token)

        eos_tok = self.TRG.vocab.stoi[const.DEFAULT_EOS]
        src_mask = (src != self.SRC.vocab.stoi[const.DEFAULT_PAD]).unsqueeze(-2).to(self.device)
        batch_size = src.shape[0]

        for i in range(1, self.max_len):
            trg_mask = no_peeking_mask(i, self.device)
            
            # Decoder
            decoder_output = model.decode(outputs[:, :i], e_outputs, src_mask, trg_mask)
            out = model.out(decoder_output)
            logits = out[:, -1, :] / self.temperature  # Apply temperature
            probs = functional.softmax(logits, dim=-1)

            # Top-k sampling
            top_probs, top_indices = torch.topk(probs, self.k, dim=-1)
            top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities
            next_token = torch.multinomial(top_probs, num_samples=1)
            next_token_id = top_indices[torch.arange(batch_size), next_token.squeeze(-1)]
            
            # Append to outputs
            outputs = torch.cat([outputs, next_token_id.unsqueeze(-1)], dim=-1)

            # Check for EOS
            if (outputs[:, i] == eos_tok).all():
                break

        # Convert outputs to translated sentences
        batch_size = src.shape[0]
        outputs = outputs.cpu().numpy()
        translated_sentences = np.empty([batch_size], dtype=object)
        trim_and_itos = lambda sent: [self.TRG.vocab.itos[i] for i in sent[1:self._length(sent, eos_tok=eos_tok)]]
        for ba in range(batch_size):
            translated_sentences[ba] = trim_and_itos(outputs[ba])

        return translated_sentences

    def preprocess_batch(self, sentences, src_lang=None, field_processed=False, pad_token="<pad>", src_size_limit=None, output_tokens=False, debug=True):
        """Preprocess a batch of sentences.
        Args:
            sentences: List of input sentences (str or tensor if field_processed=True).
            src_lang: Source language (optional).
            field_processed: If True, input is already tokenized and indexed.
            pad_token: Padding token.
            src_size_limit: Maximum source length (optional).
            output_tokens: If True, return tokenized sentences alongside indices.
            debug: If True, print debug information.
        Returns:
            Processed batch (tensor) and optionally tokenized sentences.
        """
        if field_processed:
            if src_size_limit is not None:
                sentences = sentences[:, :src_size_limit]
            return sentences
        processed_sent = map(self.SRC.preprocess, sentences)
        if src_lang is not None:
            src_token = generate_language_token(src_lang)
            processed_sent = map(lambda x: [src_token] + x, processed_sent)
        if src_size_limit:
            processed_sent = map(lambda x: x[:src_size_limit], processed_sent)
        processed_sent = list(processed_sent)
        tokenized_sent = [torch.LongTensor([self._token_to_index(t) for t in s]) for s in processed_sent]
        sentences = Variable(pad_sequence(tokenized_sent, True, padding_value=self.SRC.vocab.stoi[pad_token]))
        if debug:
            print("Input batch after process: ", sentences.shape, sentences)
        
        if output_tokens:
            return sentences, processed_sent
        else:
            return sentences

    def translate_batch_sentence(self, src, src_lang=None, trg_lang=None, field_processed=False, src_size_limit=None, output_tokens=False, debug=False, replace_unk=False):
        """Translate a batch of sentences using top-k sampling.
        Args:
            src: The batch of sentences to be translated (list of str or tensor).
            src_lang: Source language (optional).
            trg_lang: Target language (optional).
            field_processed: If True, input is already processed.
            src_size_limit: Maximum source length (optional).
            output_tokens: If True, return tokens instead of joined sentences.
            debug: If True, print debug information.
            replace_unk: If True, replace unknown tokens (default: False).
        Returns:
            Translated sentences (list of str or list of tokens).
        """
        self.model.eval()
        processed_batch = self.preprocess_batch(src, src_lang=src_lang, field_processed=field_processed, src_size_limit=src_size_limit, output_tokens=True, debug=debug)
        sent_ids, sent_tokens = (processed_batch, None) if field_processed else processed_batch
        assert isinstance(sent_ids, torch.Tensor), "sent_ids is instead {}".format(type(sent_ids))

        batch_start = time.time()
        translated_sentences = self.top_k_sampling(sent_ids, trg_lang=trg_lang, src_tokens=sent_tokens, debug=debug)
        if debug:
            print("Time performed for batch {}: {:.2f}s".format(sent_ids.shape, time.time() - batch_start))

        if not output_tokens:
            translated_sentences = [' '.join(tokens) for tokens in translated_sentences]

        return translated_sentences

    def translate_batch(self, sentences, **kwargs):
        return self.translate_batch_sentence(sentences, **kwargs)

    def _length(self, tokens, eos_tok=None):
        """Retrieve the first location of eos_tok as length; else return the entire length."""
        if eos_tok is None:
            eos_tok = self.TRG.vocab.stoi[const.DEFAULT_EOS]
        eos, = np.nonzero(tokens == eos_tok)
        return len(tokens) if len(eos) == 0 else eos[0]

    def _token_to_index(self, tok):
        """Convert token to index."""
        return self.SRC.vocab.stoi[tok]