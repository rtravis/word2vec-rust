# word2vec-rust
This project aims to port Google's published [word2vec implementation in C](https://code.google.com/archive/p/word2vec/) to the Rust programming language.

Currently it can only train using the CBOW (Continuous Bag of Words) architecture and saves the output vectors in a format compatible with the original C implementation.


Code comments in `nnet.rs` are taken from [word2vec_commented](https://github.com/chrisjmccormick/word2vec_commented) repository.

## word2vec Model Training

word2vec training is called from the main executable main.rs to run training you can for example run:

```shell
cargo run --release -- -t TRAINING_FILE -o OUTPUT_VECTORS_FILE
```

where `TRAINING_FILE` is the file that contains newline delimited sentences and `OUTPUT_VECTORS_FILE` will be a binary vectors file accepted by the word2vec tools such as the `distance` utility.

### Text Parsing

The word2vec C project does not include code for parsing and tokenizing your text. It simply accepts a training file with words separated by whitespace (spaces, tabs, or newlines). This means that you'll need to handle the removal of things like punctuation separately.

The code expects the text to be divided into sentences (with a default maximum length of 1,000 words). The end of a sentence is marked by a single newline character "\n"--that is, there should be one sentence per line in the file.

