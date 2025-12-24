use std::fs::metadata;

use w2v_rs::nnet::{NeuralNet, train_model_thread};
use w2v_rs::vocab::Vocabulary;

fn train(training_file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file_size = metadata(training_file)?.len();
    let vocab = Vocabulary::learn_vocabulary_from_training_file(training_file, 2);
    let mut net = NeuralNet::new(vocab.len(), 10);
    let res = train_model_thread(training_file, &vocab, &mut net, 0, 1, file_size);
    Ok(res?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut training_file: String = String::from("text8");
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
            _ => {
                if arg.starts_with('-') {
                    println!("Unkown argument {}", arg);
                } else {
                    println!("Unkown positional argument {}", arg);
                }
            }
        }
    }
    train(&training_file)
}
