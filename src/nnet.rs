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

use std::fs::File;
use std::io::{BufWriter, Write};
use std::ptr::slice_from_raw_parts_mut;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::mem_block_lock::MemBlockLocker;
use crate::tokenizer::FileTokenIterator;
use crate::vocab::Vocabulary;


pub struct TrainigParams {
    pub training_file: String,
    pub training_file_size: u64,
    pub vocab_file: String,
    pub save_vocab_file: String,
    pub output_file: String,
    pub vectors_size: usize,
    pub window: usize,         // the train window parameter
    pub total_iter: u64,       // number of training epochs
    pub negative_samples: i32, // number of negative samples
    pub num_threads: usize,    // the total number of training threads
    pub starting_alpha: f32,   // the starting learning rate
    pub debug_mode: i32,
    pub min_count: u32,
    pub binary: bool,
}

pub struct NeuralNet {
    vocab_size: usize,
    layer1_size: usize,
    syn0: Vec<f32>,
    syn1neg: Vec<f32>,
    locker: MemBlockLocker,
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
            locker: MemBlockLocker::new(),
        };

        let mut lc_rand = LcRandomGen::new(1);
        let rand_gen =
            || (((lc_rand.next_rand() & 0xffff) as f32 / 65536.0) - 0.5) / layer1_size as f32;
        net.syn0.resize_with(size, rand_gen);
        net.syn1neg.resize(size, 0.0);
        net
    }

    pub fn save(
        &self,
        vocab: &Vocabulary,
        output_file_name: &str,
        binary: bool,
    ) -> Result<(), std::io::Error> {
        let mut buf_writer: BufWriter<File> = BufWriter::new(File::create(output_file_name)?);
        writeln!(buf_writer, "{} {}", self.vocab_size, self.layer1_size)?;
        let syn0 = &self.syn0;
        for (idx, word) in vocab.into_iter().enumerate() {
            write!(buf_writer, "{word} ")?;
            let word_vec = &syn0[idx * self.layer1_size..(idx + 1) * self.layer1_size];
            if binary {
                unsafe {
                    let data = std::slice::from_raw_parts(
                        word_vec.as_ptr() as *const u8,
                        word_vec.len() * std::mem::size_of_val(&word_vec[0]),
                    );
                    buf_writer.write_all(data)?;
                }
            } else {
                for f in word_vec {
                    write!(buf_writer, "{f:.06} ")?;
                }
            }
            writeln!(buf_writer)?;
        }

        Ok(())
    }
}

/// @return None on EOF, Some(-1) if token is not in the vocabulary, Some(token_index) otherwise
fn read_word_index(fi: &mut FileTokenIterator, vocab: &Vocabulary) -> Option<i32> {
    fi.read_token().map(|t| vocab.search_word(&t))
}

/// @return the dot product of 2 f32 vectors
fn dot_product(vec1: &[f32], vec2: &[f32]) -> f32 {
    debug_assert!(vec1.len() == vec2.len());
    vec1.iter()
        .zip(vec2)
        .fold(0.0, |acc, cur| acc + cur.0 * cur.1)
}

// /// y <- a * x + y, named after Fortran's axpy
// fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
//     x.iter()
//         .zip(y.iter_mut())
//         .for_each(|(src, dest)| *dest += a * src);
// }

pub struct TrainigProgress {
    pub word_count_actual: AtomicU64,
}

const MAX_SENTENCE_LENGTH: usize = 1024;

/// train the word2vec neural net `net` with training data found in `training_file`
pub fn train_model_thread(
    net: Arc<NeuralNet>,
    vocab: &Vocabulary,
    thread_id: usize,
    params: &TrainigParams,
    progress: &TrainigProgress,
) -> Result<(), std::io::Error> {
    assert!(net.vocab_size == vocab.len());
    assert!(net.vocab_size * net.layer1_size == net.syn0.len());
    assert!(net.syn0.len() == net.syn1neg.len());

    let offset = params.training_file_size / params.num_threads as u64 * thread_id as u64;
    let mut fi = FileTokenIterator::new(&params.training_file, offset)?;
    let mut eof_reached: bool = false;
    let layer1_size = net.layer1_size;

    let mut neu1: Vec<f32> = Vec::with_capacity(layer1_size);
    neu1.resize(layer1_size, 0.0);
    let mut neu1e: Vec<f32> = Vec::with_capacity(layer1_size);
    neu1e.resize(layer1_size, 0.0);

    let mut rand_gen = LcRandomGen::new(thread_id as i64);
    // progress tracking
    let mut word_count: u64 = 0;
    let mut last_word_count: u64 = 0;
    let start: Instant = Instant::now();

    let mut sentence = [-1; MAX_SENTENCE_LENGTH + 1];
    let mut sentence_length: usize = 0;
    let mut sentence_position: usize = 0;
    let mut local_iter = params.total_iter;
    let mut alpha: f32 = params.starting_alpha;

    'thread_loop: loop {
        // This block prints a progress update, and also adjusts the training
        // 'alpha' parameter.
        if word_count - last_word_count > 10000 {
            progress
                .word_count_actual
                .fetch_add(word_count - last_word_count, Ordering::Relaxed);
            last_word_count = word_count;

            let wc = progress.word_count_actual.load(Ordering::Relaxed) as f64;

            // The percentage complete is based on the total number of passes we are
            // doing and not just the current pass.
            if params.debug_mode > 1 {
                print!(
                    "\rAlpha: {alpha:.06} Progress: {:.02}%  Words/sec: {:.02}k ",
                    wc / (params.total_iter * vocab.train_words() + 1) as f64 * 100_f64,
                    (wc / 1000_f64) / start.elapsed().as_secs_f64()
                );
            }

            std::io::stdout().flush().unwrap_or_default();

            // Update alpha to: [initial alpha] * [percent of training remaining]
            // This means that alpha will gradually decrease as we progress through
            // the training text.
            alpha = params.starting_alpha
                * (1_f32 - wc as f32 / (params.total_iter * vocab.train_words() + 1) as f32);

            // Don't let alpha go below [initial alpha] * 0.0001.
            if alpha < params.starting_alpha * 0.0001 {
                alpha = params.starting_alpha * 0.0001;
            }
        }

        // Retrieve the next sentence from the training set and store it in `sentence`
        if sentence_length == 0 {
            loop {
                let idx = match read_word_index(&mut fi, vocab) {
                    Some(x) if x < 0 => continue,
                    Some(x) if x as usize >= net.vocab_size => continue,
                    Some(x) => x,
                    None => {
                        eof_reached = true;
                        break;
                    }
                };

                word_count += 1;

                // word 0 is the special token "</s>" which indicates the end of a sentence
                if idx == 0 {
                    // an empty sentence, or one consisting only of out-of-vocabulary words
                    if sentence_length == 0 {
                        continue;
                    }
                    break;
                }

                sentence[sentence_length] = idx;
                sentence_length += 1;
                if sentence_length > MAX_SENTENCE_LENGTH {
                    break;
                }
                sentence_position = 0;
            }
        }

        if (sentence_length == 0 && eof_reached)
            || (word_count > vocab.train_words() / params.num_threads as u64)
        {
            local_iter -= 1;
            if local_iter == 0 {
                break 'thread_loop;
            }
            word_count = 0;
            last_word_count = 0;
            sentence_length = 0;
            // alpha *= 0.75;
            fi.reset(offset)?;
            eof_reached = false;
            continue 'thread_loop;
        }

        let word = sentence[sentence_position];
        // assertion taken care of when filling sentence
        debug_assert!(word >= 0 && (word as usize) < net.vocab_size);

        neu1.fill(0.0);
        // `cw` stores the context word count
        let mut cw = 0;
        let b = rand_gen.next_rand() as usize % params.window;

        for a in b..params.window * 2 + 1 - b {
            if a == params.window {
                continue;
            }
            let c: isize = sentence_position as isize - params.window as isize + a as isize;
            if c < 0 || c >= sentence_length as isize {
                continue;
            }

            let last_word = sentence[c as usize] as usize;

            // sum all the context word vectors and store the result in neu1
            let net_word_index = last_word * layer1_size;
            let word_vec = unsafe {
                net.syn0
                    .get_unchecked(net_word_index..net_word_index + layer1_size)
            };
            for i in 0..neu1.len() {
                neu1[i] += word_vec[i];
            }
            cw += 1;
        }

        // if there were any context words
        if cw > 0 {
            // neu1e is used in this block only
            neu1e.fill(0.0);
            // `neu1` is the sum of the context word vectors, and now
            // becomes their average.
            for n in &mut neu1 {
                *n /= cw as f32;
            }

            // NEGATIVE SAMPLING
            // Rather than performing backpropagation for every word in our
            // vocabulary, we only perform it for the positive sample and a few
            // negative samples (the number of words is given by 'negative').
            // These negative words are selected using a "unigram" distribution,
            // which is generated in the function InitUnigramTable.
            for d in 0..params.negative_samples + 1 {
                let target: i32;
                let label: f32;

                if d == 0 {
                    // On the first iteration, we're going to train the positive sample.
                    target = word;
                    label = 1.0;
                } else {
                    // On the other iterations, we'll train the negative samples.
                    // Pick a random word to use as a 'negative sample'; do this using
                    // the unigram table.
                    target = vocab.sample_random_word(rand_gen.next_rand());
                    // Don't use the positive sample as a negative sample!
                    if target == word {
                        continue;
                    }
                    // this condition allows us to use unsafe code to index the nets
                    if target < 0 || target as usize >= net.vocab_size {
                        continue;
                    }
                    // Mark this as a negative example.
                    label = 0.0;
                }

                // At this point, target might either be the positive sample or a
                // negative sample, depending on the value of `label`.

                // Get the index of the target word in the output layer.
                let l2 = target as usize * layer1_size;
                let target_output_weights =
                    unsafe { net.syn1neg.get_unchecked(l2..l2 + layer1_size) };

                // Calculate the dot product between:
                //   neu1 - The average of the context word vectors.
                //   syn1neg[l2] - The output weights for the target word.
                let f: f32 = dot_product(&neu1, target_output_weights);

                // This block does two things:
                //   1. Calculates the output of the network for this training
                //      pair, using the expTable to evaluate the output layer
                //      activation function.
                //   2. Calculate the error at the output, stored in 'g', by
                //      subtracting the network output from the desired output,
                //      and finally multiply this by the learning rate.

                // activation function: 1 / (1 + e^(-x)) = e^x / (e^x + 1)
                let expx = f64::exp(f as f64);
                let output = expx / (expx + 1.0);
                let err = (label - output as f32) * alpha;

                // Multiply the error by the output layer weights.
                // (I think this is the gradient calculation?)
                // Accumulate these gradients over all of the negative samples.
                for i in 0..layer1_size {
                    neu1e[i] += err * target_output_weights[i];
                }

                // Update the output layer weights by multiplying the output error
                // by the average of the context word vectors.
                unsafe {
                    let target_output_weights_mut = slice_from_raw_parts_mut(
                        target_output_weights.as_ptr().cast_mut(),
                        target_output_weights.len(),
                    );

                    net.locker.lock(target as usize);
                    for (i, n) in neu1.iter().enumerate() {
                        (*target_output_weights_mut)[i] += n;
                    }
                    net.locker.unlock(target as usize);
                }
            }

            // hidden -> in
            // Backpropagate the error to the hidden layer (the word vectors).
            // This code is used both for heirarchical softmax and for negative
            // sampling.
            //
            // Loop over the positions in the context window (skipping the word at
            // the center). 'a' is just the offset within the window, it's not
            // the index relative to the beginning of the sentence.
            for a in b..params.window * 2 + 1 - b {
                if a == params.window {
                    continue;
                }
                // Convert the window offset 'a' into an index 'c' into the sentence
                // array.
                let c: isize = sentence_position as isize - params.window as isize + a as isize;

                // Verify c isn't outisde the bounds of the sentence.
                if c < 0 || c >= sentence_length as isize {
                    continue;
                }

                // Get the context word. That is, get the id of the word (its index in
                // the vocab table).
                let last_word = sentence[c as usize] as usize;

                // Add the gradient in the vector `neu1e` to the word vector for
                // the current context word.
                // syn0[last_word * layer1_size] <-- Accesses the word vector.
                let word_vector = unsafe {
                    net.syn0
                        .get_unchecked(last_word * layer1_size..(last_word + 1) * layer1_size)
                };
                unsafe {
                    let mutable_unsafe_slice = slice_from_raw_parts_mut(
                        word_vector.as_ptr().cast_mut(),
                        word_vector.len(),
                    );

                    net.locker.lock(last_word);
                    for (i, err) in neu1e.iter().enumerate() {
                        (*mutable_unsafe_slice)[i] += err;
                    }
                    net.locker.unlock(last_word);
                }
            }
        }

        // Advance to the next word in the sentence.
        sentence_position += 1;

        // Check if we've reached the end of the sentence. If so, set sentence_length
        // to 0 and we'll read a new sentence at the beginning of this loop.
        if sentence_position >= sentence_length {
            sentence_length = 0;
            continue;
        }
    }

    Ok(())
}
