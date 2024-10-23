# N-gram Language Models and Context-Aware Spelling Correction

This repository contains an assignment for the **Text Analytics** course, part of the **MSc in Data Science** program. The assignment focuses on the development of n-gram language models (bigram and trigram) and a context-aware spelling correction system using various smoothing techniques. The primary tasks include implementing bigram and trigram models, estimating cross-entropy and perplexity, and creating a context-aware spelling corrector with a beam search decoder.

In addition, the assignment involves evaluating the models and corrector on test data, calculating Word Error Rate (WER) and Character Error Rate (CER). Detailed steps for training, testing, and evaluating the models are outlined below, along with the corresponding code implementations.

---

## 2. [Optional] 
- **Task (i)**: Implement the dynamic programming algorithm that computes the **Levenshtein distance**. You may want to compare the outputs of your implementation to those of [this link](http://www.let.rug.nl/~kleiweg/lev/).
  
- **Task (ii)**: Optionally extend your implementation to accept as input a word $w$, a vocabulary $V$ (e.g., words that occur at least 10 times in a corpus), and a maximum distance $d$, and return the words of $V$ whose Levenshtein distance to $w$ is up to $d$.

---

## 3. N-gram Language Models and Spelling Correction

### 3.1. Bigram and Trigram Language Models
- **Task (i)**: Implement a **bigram** and a **trigram** language model for sentences, using **Laplace smoothing** or optionally (if you are very keen) **Kneser-Ney smoothing**, which is much better. In practice, n-gram language models compute the sum of the logarithms of the n-gram probabilities of each sequence, instead of their product (why?) and you should do the same.
  
- **Start and End Tokens**: Assume that each sentence starts with the pseudo-token `*start*` (or two pseudo-tokens `*start1*`, `*start2*` for the trigram model) and ends with the pseudo-token `*end*`.

- **Training**: Train your models on a training subset of a corpus (e.g., a subset of a corpus included in NLTK – see [this link](http://www.nltk.org/)). Include in the vocabulary only words that occur, e.g., at least 10 times in the training subset. Use the same vocabulary in the bigram and trigram models. Replace all out-of-vocabulary (OOV) words (in the training, development, test subsets) by a special token `*UNK*`.

- **BPE Option**: Alternatively, you may want to use **Byte-Pair Encoding (BPE)** instead of words (obtaining the BPE vocabulary from your training subset) to avoid unknown words. See Section 2.4.3 (“Byte-Pair Encoding for Tokenization”) of the 3rd edition of Jurafsky & Martin’s book ([link](https://web.stanford.edu/~jurafsky/slp3/)); for more information, check [this link](https://huggingface.co/transformers/master/tokenizer_summary.html).

---

### 3.2. Cross-Entropy and Perplexity
- **Task (ii)**: Estimate the **language cross-entropy** and **perplexity** of your two models on a test subset of the corpus, treating the entire test subset as a single sequence of sentences, with `*start*` (or `*start1*`, `*start2*`) at the beginning of each sentence, and `*end*` at the end of each sentence.

- **Exclusion of Start Tokens**: Do not include probabilities of the form $P(*start* | \dots)$ or $P(*start1* | \dots), P(*start2* | \dots)$ in the computation of cross-entropy and perplexity, since we are not predicting the start pseudo-tokens. However, do include probabilities of the form $P(*end* | \dots)$, since we do want to be able to predict if a word will be the last one of a sentence.

- **Length Counting**: You must also count `*end*` tokens (but not `*start*`, `*start1*`, `*start2*`) in the total length $N$ of the test corpus.

---

### 3.3. Context-aware Spelling Correction
- **Task (iii)**: Develop a **context-aware spelling corrector** (for both types of errors) using your **bigram language model**, a **beam search decoder**.

- **Levenshtein Distance**: You can use the **inverse of the Levenshtein distance** between $w_i$ and $t_i$ as $P(w_i | t_i)$.

- **Trigram Model Option**: If you are very keen, you can also try using your **trigram model**.

- **Better Probability Estimates**: You may also want to use better estimates of $P(w_i | t_i)$ that satisfy $\sum P(w_i | t_i) = 1$.

- **Lambda Tuning**: Use the following formula to control (by tuning the hyper-parameters $\lambda_1, \lambda_2$) the importance of the language model score $\log P(t_1^m)$ vs. the importance of $\log P(w_1^m | t_1^m)$:

$$
\hat{w_1^m} = \arg\max_{w_1^m} \left( \lambda_1 \log P(t_1^m) + \lambda_2 \log P(w_1^m | t_1^m) \right)
$$

---

### 3.4. Artificial Test Dataset
- **Task (iv)**: Create an artificial test dataset to evaluate your context-aware spelling corrector. You may use, for example, the test dataset that you used to evaluate your language models, but now replace with a small probability each non-space character of each test sentence with another random (or visually or acoustically similar) non-space character.

- **Example**: “This is an interesting course.” may become “Tais is an imterestieg kourse.”

---

### 3.5. Spelling Corrector Evaluation
- **Task (v)**: Evaluate your context-aware spelling corrector in terms of **Word Error Rate (WER)** and **Character Error Rate (CER)**, using the original (before character replacements) form of your test dataset from question (iv) as the **ground truth** (reference sentences), and averaging WER (or CER) over the test sentences.

- **Metrics**: 
  - WER is similar to CER but operates at the word level.
  - CER operates at the character level.
  
  See the following references:
  - [WER Reference](https://huggingface.co/spaces/evaluate-metric/wer)
  - [CER Reference](https://huggingface.co/spaces/evaluate-metric/cer)

---

## Libraries and Tools
- You are allowed to use **NLTK** ([NLTK website](http://www.nltk.org/)) or other tools and libraries for:
  - Sentence splitting
  - Tokenization (including BPE tokenizers)
  - Counting n-grams
  - Computing Levenshtein distances, WER, and CER.
  
- You **must write your own code** for:
  - Estimating probabilities
  - Computing cross-entropy and perplexity
  - Beam search decoding.
  
- **Comparison**: Compare the cross-entropy and perplexity results of your implementation to results obtained by using existing code (e.g., from NLTK or other toolkits).

---
