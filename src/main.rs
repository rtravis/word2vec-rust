use std::fs::metadata;

use w2v_rs::nnet::{NeuralNet, train_model_thread};
use w2v_rs::vocab::Vocabulary;

fn train(
    training_file: &str,
    vocab_file: &str,
    output_file: &str,
    save_vocab_file: &str,
    debug_mode: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let file_size = metadata(training_file)?.len();
    let vocab: Vocabulary;
    if vocab_file.is_empty() {
        vocab = Vocabulary::learn_vocabulary_from_training_file(training_file, 2)?;
    } else {
        vocab = Vocabulary::load_from_file(vocab_file)?;
    }

    if debug_mode > 0 {
        vocab.debug_print_summary();
    }

    if !save_vocab_file.is_empty() {
        if debug_mode > 0 {
            println!("Saving vocabulary to file: '{save_vocab_file}'");
        }
        vocab.save_to_file(&save_vocab_file)?;
    }

    if output_file.is_empty() {
        println!("No output file specified, skipping training");
        return Ok(());
    }

    let mut net = NeuralNet::new(vocab.len(), 10);
    train_model_thread(&mut net, training_file, &vocab, 0, 1, file_size)?;
    net.save(&vocab, &output_file, true)?;
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
