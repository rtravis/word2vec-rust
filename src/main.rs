use w2v_rs::nnet::NeuralNet;
use w2v_rs::vocab::Vocabulary;

fn main() {
    let vocab = Vocabulary::learn_vocabulary_from_training_file("bigdoc.txt");
    let _net = NeuralNet::new(vocab.size(), 10);
}
