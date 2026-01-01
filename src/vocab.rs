use super::tokenizer::read_file_by_tokens;
use core::str;
use std::fs::File;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Error, Write};

struct WordInfo {
    word: String,
    count: u32,
}

const VOCAB_HASH_TABLE_SIZE: i32 = 30_000_000;
const UNIGRAM_TABLE_SIZE: usize = 100_000_000;

fn get_word_hash_index(word: &str) -> usize {
    let mut hasher = DefaultHasher::new();
    word.hash(&mut hasher);
    (hasher.finish() % VOCAB_HASH_TABLE_SIZE as u64) as usize
}

pub struct Vocabulary {
    words: Vec<WordInfo>,
    hash_table: Vec<i32>,
    train_words: u64,
    min_reduce: u32,
    unigram_table: Vec<i32>,
}

impl Vocabulary {
    pub fn learn_vocabulary_from_training_file(
        file_name: &str,
        min_count: u32,
    ) -> std::io::Result<Vocabulary> {
        let mut vocab = Vocabulary::new();
        let mut word_callback = |word: &[u8]| {
            let word_str =
                String::from_utf8(Vec::from(word)).unwrap_or_else(|_| String::from("<INV>"));
            let _ = vocab.add_word(word_str);
        };

        // ensure the document/sentence/line separator represented by "</s>" has index 0, as
        // expected by other functions
        word_callback(b"</s>");
        read_file_by_tokens(file_name, word_callback)?;
        vocab.sort_vocab(min_count);
        vocab.init_unigram_table();
        Ok(vocab)
    }

    pub fn save_to_file(&self, vocab_file: &str) -> std::io::Result<()> {
        let mut buf_writer: BufWriter<File> = BufWriter::new(File::create(vocab_file)?);
        for w in self.words.iter() {
            writeln!(buf_writer, "{} {}", w.word, w.count)?;
        }
        Ok(())
    }

    pub fn debug_print_summary(&self) {
        println!("Vocab size: {}", self.words.len());
        println!("Words in train file: {}", self.train_words);
    }

    pub fn load_from_file(vocab_file: &str) -> std::io::Result<Vocabulary> {
        let mut buf_reader = BufReader::new(File::open(vocab_file)?);
        let mut vocab = Vocabulary::new();

        let mut line_buf: Vec<u8> = vec![];
        loop {
            line_buf.clear();
            let res = buf_reader.read_until(b'\n', &mut line_buf);
            let p = match res {
                Ok(0) => break,
                Ok(_size) => {
                    let mut parts = line_buf.split(|b| b.is_ascii_whitespace());
                    let (Some(p1), Some(p2)) = (parts.next(), parts.next()) else {
                        break;
                    };
                    (str::from_utf8(p1), str::from_utf8(p2))
                }
                Err(e) => return Err(e),
            };

            let (Ok(word), Ok(count)) = p else {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Encountered invalid line",
                ));
            };

            let Ok(count): Result<u32, _> = count.parse() else {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Count is not a positive integer",
                ));
            };

            if word.is_empty() {
                return Err(Error::new(std::io::ErrorKind::InvalidData, "Word is empty"));
            }

            vocab.add_word_with_count(word.to_string(), count);
        }

        if vocab.is_empty() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Empty vocabulary",
            ));
        }

        vocab.init_unigram_table();
        Ok(vocab)
    }

    /// return word index (or word ID), -1 is returned if not found
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

    /// return a pair: (hashtable_index, word_array_index), where
    /// hashtable_index is the hashtable index for an existing word or the hashtable
    /// insertion index for a new word
    /// word_array_index the index in the word array for an existing word or -1 for
    /// a new word
    fn get_word_indices(&self, word: &str) -> (usize, i32) {
        // index in the hashtable
        let mut hash_idx = get_word_hash_index(word);
        // index in the words array
        let mut word_idx: i32 = -1;
        loop {
            if self.hash_table[hash_idx] == -1 {
                break;
            }
            let wi = self.hash_table[hash_idx];
            if self.words[wi as usize].word == word {
                word_idx = wi;
                break;
            }
            hash_idx = (hash_idx + 1) % (VOCAB_HASH_TABLE_SIZE as usize);
        }
        (hash_idx, word_idx)
    }

    fn add_word(&mut self, word: String) -> i32 {
        let (hash_idx, mut word_idx) = self.get_word_indices(&word);

        if word_idx == -1 {
            word_idx = self.words.len() as i32;
            self.words.push(WordInfo { word, count: 1 });
            self.hash_table[hash_idx] = word_idx;
        } else {
            self.words[word_idx as usize].count += 1;
        }
        self.train_words += 1;

        if self.words.len() as f64 > (0.7 * VOCAB_HASH_TABLE_SIZE as f64) {
            self.reduce_vocab();
            // word_idx is no longer valid at this point, set it to -1
            word_idx = -1;
        }
        word_idx
    }

    fn add_word_with_count(&mut self, word: String, count: u32) -> i32 {
        let (hash_idx, mut word_idx) = self.get_word_indices(&word);
        assert!(word_idx == -1);

        if word_idx == -1 {
            word_idx = self.words.len() as i32;
            self.words.push(WordInfo { word, count });
            self.hash_table[hash_idx] = word_idx;
        } else {
            self.words[word_idx as usize].count += count;
        }
        self.train_words += count as u64;
        word_idx
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

    fn init_unigram_table(&mut self) {
        assert!(!self.words.is_empty());

        // initialize the unigram table according to the word count distribution
        const WORD_POWER: f64 = 0.75;
        let train_words_pow: f64 = self.words.iter().fold(0.0f64, |acc, word| {
            acc + f64::powf(word.count as f64, WORD_POWER)
        });

        let mut frac: f64 = f64::powf(self.words[0].count as f64, WORD_POWER) / train_words_pow;

        self.unigram_table.reserve(UNIGRAM_TABLE_SIZE);
        unsafe {
            self.unigram_table.set_len(UNIGRAM_TABLE_SIZE);
        }

        let mut word_idx: usize = 0;
        for (idx, tab_val) in self.unigram_table.iter_mut().enumerate() {
            *tab_val = word_idx as i32;
            if (idx as f64 / UNIGRAM_TABLE_SIZE as f64) > frac {
                word_idx += 1;
                frac += f64::powf(self.words[word_idx].count as f64, WORD_POWER) / train_words_pow;
            }
        }
    }
}

pub struct VocabularyIter<'a> {
    vocab: &'a Vocabulary,
    i: usize,
}

impl<'a> Iterator for VocabularyIter<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.vocab.words.len() {
            None
        } else {
            self.i += 1;
            Some(&self.vocab.words[self.i - 1].word)
        }
    }
}

impl<'a> IntoIterator for &'a Vocabulary {
    type Item = &'a str;
    type IntoIter = VocabularyIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        VocabularyIter { vocab: self, i: 0 }
    }
}
