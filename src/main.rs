//  Copyright 2026 Robert Zavalczki
//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
use std::fs::metadata;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::thread;

use word2vec_rust::distance::print_close_words;
use word2vec_rust::nnet::{NeuralNet, TrainigParams, TrainigProgress, train_model_thread};
use word2vec_rust::vocab::Vocabulary;

fn train(params: &mut TrainigParams) -> Result<(), Box<dyn std::error::Error>> {
    params.training_file_size = metadata(&params.training_file)?.len();
    let vocab: Vocabulary = if params.vocab_file.is_empty() {
        Vocabulary::learn_vocabulary_from_training_file(&params.training_file, params.min_count)?
    } else {
        Vocabulary::load_from_file(&params.vocab_file)?
    };

    if params.debug_mode > 0 {
        vocab.debug_print_summary();
    }

    if !params.save_vocab_file.is_empty() {
        if params.debug_mode > 0 {
            println!("Saving vocabulary to file: '{}'", &params.save_vocab_file);
        }
        vocab.save_to_file(&params.save_vocab_file)?;
    }

    if params.output_file.is_empty() {
        println!("No output file specified, skipping training");
        return Ok(());
    }

    let progress = TrainigProgress {
        word_count_actual: AtomicU64::new(0),
    };

    let net = NeuralNet::new(vocab.len(), params.vectors_size);
    let net = Arc::new(net);

    thread::scope(|scope| {
        // we don't need these "moved", but "thread_id" has to be moved
        let vocab = &vocab;
        let params = &params;
        let progress = &progress;

        for thread_id in 0..params.num_threads {
            let net = Arc::clone(&net);
            scope.spawn(move || train_model_thread(net, vocab, thread_id, params, progress));
        }
    });

    net.save_word_vecs(&vocab, &params.output_file, params.binary)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut params = TrainigParams {
        training_file: String::new(),
        training_file_size: 0,
        vocab_file: String::new(),
        save_vocab_file: String::new(),
        output_file: String::new(),
        vectors_size: 100,
        window: 5,
        total_iter: 5,
        negative_samples: 5,
        num_threads: 1,
        starting_alpha: 0.025,
        debug_mode: 2,
        min_count: 5,
        binary: false,
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match &arg[..] {
            "-t" | "--train" => {
                // Use text data from <file> to train the model
                if let Some(arg_file) = args.next() {
                    params.training_file = arg_file;
                } else {
                    panic!("No value specified for parameter --train.");
                }
            }
            "-o" | "--output" => {
                // Use <file> to save the resulting word vectors / word clusters
                if let Some(arg_file) = args.next() {
                    params.output_file = arg_file;
                } else {
                    panic!("No value specified for parameter --output.");
                }
            }
            "-s" | "--size" => {
                // Set size of word vectors; default is 100
                if let Some(val) = args.next().and_then(|x| x.parse().ok()) {
                    params.vectors_size = val;
                } else {
                    panic!("No valid value specified for parameter --size, must be >= 1");
                }
            }
            "-w" | "--window" => {
                // Set max skip length between words; default is 5
                if let Some(val) = args.next().and_then(|x| x.parse().ok()) {
                    params.window = val;
                } else {
                    panic!("No valid value specified for parameter --window, must be >= 1");
                }
            }
            "-ns" | "--negative-samples" => {
                // Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
                if let Some(val) = args.next().and_then(|x| x.parse().ok()) {
                    params.negative_samples = val;
                } else {
                    panic!("No valid value specified for parameter --window, must be >= 0");
                }
            }
            "--threads" => {
                // Use <usize> threads (default 1)
                if let Some(val) = args.next().and_then(|x| x.parse().ok()) {
                    params.num_threads = val;
                } else {
                    panic!("No valid value specified for parameter --threads, must be >= 1");
                }
            }
            "--iter" => {
                // Run more training iterations or epochs (default 5)
                if let Some(val) = args.next().and_then(|x| x.parse().ok()) {
                    params.total_iter = val;
                } else {
                    panic!("No valid value specified for parameter --iter, must be >= 1");
                }
            }
            "--alpha" => {
                // Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
                if let Some(val) = args.next().and_then(|x| x.parse().ok()) {
                    params.starting_alpha = val;
                } else {
                    panic!(
                        "No valid value specified for parameter --alpha, must be a fraction in (0, 1)"
                    );
                }
            }
            "--binary" => {
                // Save the resulting vectors in binary moded; default is 0 (off)
                if let Some(val) = args.next().and_then(|x| x.parse::<u32>().ok()) {
                    params.binary = val != 0;
                } else {
                    panic!("No valid value specified for parameter --binary, must be 0 or 1");
                }
            }
            "-v" | "--read-vocab" => {
                // The vocabulary will be read from <file>, not constructed from the training data
                if let Some(arg_file) = args.next() {
                    params.vocab_file = arg_file;
                } else {
                    panic!("No value specified for parameter --read-vocab.");
                }
            }
            "--save-vocab" => {
                // The vocabulary will be saved to <file>
                if let Some(arg_file) = args.next() {
                    params.save_vocab_file = arg_file;
                } else {
                    panic!("No value specified for parameter --save-vocab.");
                }
            }
            "-d" | "--debug" => {
                // Set the debug mode (default = 2 = more info during training)
                if let Some(val) = args.next().and_then(|x| x.parse().ok()) {
                    params.debug_mode = val;
                } else {
                    panic!("No valid value specified for parameter --debug, must be 0, 1, 2, ...");
                }
            }
            "-m" | "--min-count" => {
                // This will discard words that appear less than <int> times; default is 5
                if let Some(val) = args.next().and_then(|x| x.parse().ok()) {
                    params.min_count = val;
                } else {
                    panic!("No valid value specified for parameter --min-count, must be >= 1");
                }
            }
            "--distance" => {
                // Example "distance" program that will load word projection vectors from
                // `word_vecs_file`, ask the user to input words and print out all words that
                // are most close to the input words. `word_vecs_file` is expected to be in the
                // binary format.
                if let Some(word_vecs_file) = args.next() {
                    return print_close_words(&word_vecs_file);
                } else {
                    panic!("No valid value specified for parameter --min-count, must be >= 1");
                }
            }
            _ => {
                if arg.starts_with('-') {
                    println!("Unkown argument {}", arg);
                } else {
                    println!("Unkown positional argument {}", arg);
                }
            }
        }
    }

    if !params.training_file.is_empty()
        && (!params.output_file.is_empty() || !params.vocab_file.is_empty())
    {
        train(&mut params)
    } else {
        Err("Wrong usage, read main.rs for program parameters.")?
    }
}
