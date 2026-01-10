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

use word2vec_rust::nnet::{NeuralNet, TrainigParams, TrainigProgress, train_model_thread};
use word2vec_rust::vocab::Vocabulary;

fn train(
    training_file: &str,
    vocab_file: &str,
    output_file: &str,
    save_vocab_file: &str,
    debug_mode: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let training_file_size = metadata(training_file)?.len();
    let vocab: Vocabulary = if vocab_file.is_empty() {
        Vocabulary::learn_vocabulary_from_training_file(training_file, 2)?
    } else {
        Vocabulary::load_from_file(vocab_file)?
    };

    if debug_mode > 0 {
        vocab.debug_print_summary();
    }

    if !save_vocab_file.is_empty() {
        if debug_mode > 0 {
            println!("Saving vocabulary to file: '{save_vocab_file}'");
        }
        vocab.save_to_file(save_vocab_file)?;
    }

    if output_file.is_empty() {
        println!("No output file specified, skipping training");
        return Ok(());
    }

    let params = TrainigParams {
        training_file,
        training_file_size,
        num_threads: 2,
        window: 5,
        total_iter: 1,
        negative_samples: 4,
        starting_alpha: 0.025,
        debug_mode,
    };

    let progress = TrainigProgress {
        word_count_actual: AtomicU64::new(0),
    };

    let net = NeuralNet::new(vocab.len(), 100);
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

    net.save(&vocab, output_file, true)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut training_file: String = String::new();
    let mut vocab_file: String = String::new();
    let mut output_file: String = String::new();
    let mut save_vocab_file: String = String::new();
    let mut debug_mode: i32 = 2;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match &arg[..] {
            "-t" | "--train" => {
                if let Some(arg_file) = args.next() {
                    training_file = arg_file;
                } else {
                    panic!("No value specified for parameter --train.");
                }
            }
            "-v" | "--read-vocab" => {
                if let Some(arg_file) = args.next() {
                    vocab_file = arg_file;
                } else {
                    panic!("No value specified for parameter --read-vocab.");
                }
            }
            "-o" | "--output" => {
                if let Some(arg_file) = args.next() {
                    output_file = arg_file;
                } else {
                    panic!("No value specified for parameter --output.");
                }
            }
            "--save-vocab" => {
                if let Some(arg_file) = args.next() {
                    save_vocab_file = arg_file;
                } else {
                    panic!("No value specified for parameter --save-vocab.");
                }
            }
            "-d" | "--debug" => {
                if let Some(val) = args.next().and_then(|x| x.parse().ok()) {
                    debug_mode = val;
                } else {
                    panic!("No valid value specified for parameter --debug, must be 0, 1, 2, ...");
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
    train(
        &training_file,
        &vocab_file,
        &output_file,
        &save_vocab_file,
        debug_mode,
    )
}
