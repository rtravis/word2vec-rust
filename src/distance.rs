use crate::nnet::NeuralNet;
use std::io;

fn normalize_vec(v: &[f32]) -> Vec<f32> {
    let norm = v
        .iter()
        .fold(0.0, |acc, y| acc + (*y as f64 * *y as f64))
        .sqrt() as f32;

    let mut result: Vec<f32> = Vec::with_capacity(v.len());
    v.iter().for_each(|x| result.push(*x / norm));
    result
}

fn dot_product(vec1: &[f32], vec2: &[f32]) -> f32 {
    debug_assert!(vec1.len() == vec2.len());
    vec1.iter()
        .zip(vec2)
        .fold(0.0, |acc, cur| acc + cur.0 * cur.1)
}

fn read_user_input() -> Result<String, Box<dyn std::error::Error>> {
    println!("Input words separated by space then press Enter, type q or quit to stop.");
    let mut input_words = String::new();
    io::stdin().read_line(&mut input_words)?;
    input_words.pop(); // strip newline
    Ok(input_words)
}

pub fn print_close_words(word_vecs_file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let word_vecs = NeuralNet::load_word_vecs(word_vecs_file, true, true)?;

    loop {
        let input_words = match read_user_input() {
            Err(e) => break Err(e),
            Ok(s) => match &s[..] {
                "q" | "quit" => break Ok(()),
                _ => s,
            },
        };

        let search_words = input_words.split(' ');

        let mut search_vec = Vec::new();
        search_vec.resize(word_vecs.vec_size, 0.0_f32);
        let mut found_words: usize = 0;

        for w in search_words {
            if let Some(v) = word_vecs.get_word_vec(w) {
                search_vec
                    .iter_mut()
                    .zip(v)
                    .for_each(|(dest, src)| *dest += src);
                found_words += 1;
            }
        }

        if found_words == 0 {
            Err("No words found in the vocabulary!")?
        }

        const MAX_SHOWN_WORD_COUNT: usize = 20;
        let wv = normalize_vec(&search_vec);
        let mut close_words = vec![(f32::INFINITY, usize::MAX); 1];

        for (idx, v) in word_vecs.into_iter().enumerate() {
            let dist = 1.0 - dot_product(&wv, v);

            if dist >= close_words.last().unwrap().0 {
                continue;
            }

            let insert_idx = close_words.partition_point(|(d, _)| d < &dist);
            close_words.insert(insert_idx, (dist, idx));
            if close_words.len() > MAX_SHOWN_WORD_COUNT {
                close_words.pop();
            }
        }

        for (d, word_idx) in close_words {
            println!("{:.05} {}", d, word_vecs.words[word_idx]);
        }
    }
}
