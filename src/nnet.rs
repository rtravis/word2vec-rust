pub struct NeuralNet {
    vocab_size: usize,
    layer1_size: usize,
    syn0: Vec<f32>,
    syn1neg: Vec<f32>,
}

struct LcRandomGen {
    state: i64,
}

impl LcRandomGen {
    fn new(seed: i64) -> LcRandomGen {
        LcRandomGen { state: seed }
    }
    fn next_rand(&mut self) -> i64 {
        self.state = self.state.wrapping_mul(25214903917).wrapping_add(11);
        self.state
    }
}

impl NeuralNet {
    pub fn new(vocab_size: usize, layer1_size: usize) -> NeuralNet {
        let size = vocab_size * layer1_size;
        let mut net = NeuralNet {
            vocab_size,
            layer1_size,
            syn0: Vec::with_capacity(size),
            syn1neg: Vec::with_capacity(size),
        };

        let mut lc_rand = LcRandomGen::new(1);
        let rand_gen =
            || (((lc_rand.next_rand() & 0xffff) as f32 / 65536.0) - 0.5) / layer1_size as f32;
        net.syn0.resize_with(size, rand_gen);
        net.syn1neg.resize(size, 0.0);
        net
    }
}

// pub fn train_model() {
//     loop {
//         break;
//     }
// }
