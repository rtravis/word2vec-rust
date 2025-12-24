use super::tokenizer::read_file_by_tokens;
use std::fs::File;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufWriter, Write};

struct WordInfo {
    word: String,
    count: u32,
}

const VOCAB_HASH_TABLE_SIZE: i32 = 30_000_000;
const UNIGRAM_TABLE_SIZE: usize = 100_000_000;

pub struct Vocabulary {
    words: Vec<WordInfo>,
    hash_table: Vec<i32>,
    train_words: u64,
    min_reduce: u32,
    unigram_table: Vec<i32>,
}

fn get_word_hash_index(word: &str) -> usize {
    let mut hasher = DefaultHasher::new();
    word.hash(&mut hasher);
    (hasher.finish() % VOCAB_HASH_TABLE_SIZE as u64) as usize
}

impl Vocabulary {
    pub fn learn_vocabulary_from_training_file(file_name: &str, min_count: u32) -> Vocabulary {
        let mut vocab = Vocabulary::new();
        let mut word_callback = |word: &[u8]| {
            let word_str =
                String::from_utf8(Vec::from(word)).unwrap_or_else(|_| String::from("<INV>"));
            let _ = vocab.add_word(word_str);
        };

        // ensure the document/sentence/line separator represented by "</s>" has index 0, as
        // expected by other functions
        word_callback(b"</s>");
        let _ = read_file_by_tokens(file_name, word_callback);
        vocab.sort_vocab(min_count);

        init_unigram_table(&mut vocab);

        vocab
    }

    pub fn save_vocab(&self, vocab_file: &str) -> std::io::Result<()> {
        let mut buf_writer: BufWriter<File> = BufWriter::new(File::create(vocab_file)?);
        for w in self.words.iter() {
            writeln!(buf_writer, "{} {}", w.word, w.count)?;
        }
        Ok(())
    }

    pub fn print_vocab(&self) {
        for (idx, w) in self.words.iter().enumerate() {
            print!("{} {}, ", idx, w.word);
        }
        println!();
    }

    pub fn search_word(&self, word: &str) -> i32 {
        let mut hidx = get_word_hash_index(word);
        loop {
            if self.hash_table[hidx] == -1 {
                return -1;
            }
            let widx = self.hash_table[hidx];
            if self.words[widx as usize].word == word {
                return widx;
            }
            hidx = (hidx + 1) % (VOCAB_HASH_TABLE_SIZE as usize);
        }
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn train_words(&self) -> u64 {
        self.train_words
    }

    // Pick a random word to use as a 'negative sample'; do this using
    // the unigram table.
    pub fn sample_random_word(&self, rand_seed: i64) -> i32 {
        let idx = (rand_seed >> 16) as usize % self.unigram_table.len();
        let mut target = self.unigram_table[idx];
        // If the target is the special end of sentence token, then just
        // pick a random word from the vocabulary instead.
        if target == 0 {
            target = (rand_seed as usize % (self.words.len() - 1) + 1) as i32;
        }
        target
    }

    fn new() -> Self {
        let mut vocab = Vocabulary {
            words: Vec::new(),
            hash_table: Vec::with_capacity(VOCAB_HASH_TABLE_SIZE as usize),
            train_words: 0,
            min_reduce: 1,
            unigram_table: Vec::new(),
        };
        vocab.hash_table.resize(VOCAB_HASH_TABLE_SIZE as usize, -1);
        vocab
    }

    fn add_word(&mut self, word: String) -> i32 {
        let mut hidx = get_word_hash_index(&word);
        let mut widx: i32 = -1;
        loop {
            if self.hash_table[hidx] == -1 {
                break;
            }
            let wi = self.hash_table[hidx];
            if self.words[wi as usize].word == word {
                widx = wi;
                break;
            }
            hidx = (hidx + 1) % (VOCAB_HASH_TABLE_SIZE as usize);
        }

        if widx == -1 {
            widx = self.words.len() as i32;
            self.words.push(WordInfo { word, count: 1 });
            self.hash_table[hidx] = widx;
        } else {
            self.words[widx as usize].count += 1;
        }
        self.train_words += 1;

        if self.words.len() as f64 > (0.7 * VOCAB_HASH_TABLE_SIZE as f64) {
            self.reduce_vocab();
            // widx is no longer valid at this point, set it to -1
            widx = -1;
        }
        widx
    }

    fn rebuild_hashtable(&mut self) {
        self.hash_table.fill(-1);
        self.train_words = 0;

        for (widx, w) in self.words.iter().enumerate() {
            let mut hidx = get_word_hash_index(&w.word);
            while self.hash_table[hidx] != -1 {
                hidx = (hidx + 1) % (VOCAB_HASH_TABLE_SIZE as usize);
            }
            self.hash_table[hidx] = widx as i32;
            self.train_words += w.count as u64;
        }
    }

    fn reduce_vocab(&mut self) {
        let mut idx: usize = 1;
        loop {
            if idx >= self.words.len() {
                break;
            }

            unsafe {
                if self.words.get_unchecked(idx).count <= self.min_reduce {
                    let mut last = self.words.len() - 1;
                    while last > idx && self.words.get_unchecked(last).count <= self.min_reduce {
                        last -= 1;
                    }
                    self.words.truncate(last + 1);
                    self.words.swap_remove(idx);
                } else {
                    idx += 1;
                }
            }
        }
        self.min_reduce += 1;
        self.rebuild_hashtable();
    }

    fn sort_vocab(&mut self, min_count: u32) {
        let f = |x: &WordInfo| u32::MAX - x.count;
        // self.words[1..].sort_unstable_by_key(f);
        self.words[1..].sort_by_key(f);
        let idx = self
            .words
            .partition_point(|x: &WordInfo| x.count >= min_count);
        self.words.truncate(idx);
        self.rebuild_hashtable();
    }
}

fn init_unigram_table(vocab: &mut Vocabulary) {
    // initialize the unigram table according to the word count distribution
    const WORD_POWER: f64 = 0.75;
    let train_words_pow: f64 = vocab.words.iter().fold(0.0f64, |acc, word| {
        acc + f64::powf(word.count as f64, WORD_POWER)
    });

    let mut frac: f64 = f64::powf(vocab.words[0].count as f64, WORD_POWER) / train_words_pow;

    vocab.unigram_table.reserve(UNIGRAM_TABLE_SIZE);
    unsafe {
        vocab.unigram_table.set_len(UNIGRAM_TABLE_SIZE);
    }

    let mut word_idx: usize = 0;
    for (idx, tab_val) in vocab.unigram_table.iter_mut().enumerate() {
        *tab_val = word_idx as i32;
        if (idx as f64 / UNIGRAM_TABLE_SIZE as f64) > frac {
            word_idx += 1;
            frac += f64::powf(vocab.words[word_idx].count as f64, WORD_POWER) / train_words_pow;
        }
    }
}
