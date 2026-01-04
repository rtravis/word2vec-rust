//  Copyright 2026 Robert Zavalczki
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
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};

#[inline]
fn is_token_separator(byte: &u8) -> bool {
    *byte == b'\n' || *byte == b' ' || *byte == b'\t' || *byte == b'\r'
}

fn is_doc_separator(byte: &u8) -> bool {
    *byte == b'\n'
}

/// Read file and invoke `token_callback` for each token (word)
pub fn read_file_by_tokens<T: FnMut(&[u8])>(
    file_name: &str,
    mut token_callback: T,
) -> std::io::Result<()> {
    let mut buf_reader: BufReader<File> = BufReader::new(File::open(file_name)?);
    let mut rest: Vec<u8> = Vec::new();

    loop {
        let data: &[u8] = buf_reader.fill_buf()?;
        if data.is_empty() {
            if !rest.is_empty() {
                token_callback(&rest);
            }
            break;
        }

        let data_len = data.len();
        let mut skip_bytes: usize = 0;

        if !rest.is_empty() {
            if let Some(pos) = data.iter().position(is_token_separator) {
                rest.extend_from_slice(&data[..pos]);
                token_callback(&rest);
                rest.clear();
                if is_doc_separator(&data[pos]) {
                    token_callback(b"</s>")
                }
                skip_bytes = pos + 1;
            }
        }

        let mut token_start: usize = skip_bytes;
        let mut token_end: usize = token_start;

        for byte in &data[skip_bytes..] {
            if is_token_separator(byte) {
                if token_end > token_start {
                    token_callback(&data[token_start..token_end]);
                }
                token_end += 1;
                token_start = token_end;

                if is_doc_separator(byte) {
                    token_callback(b"</s>");
                }
            } else {
                token_end += 1;
            }
        }

        if (token_end - token_start) != 0 {
            rest.extend_from_slice(&data[token_start..token_end]);
        }

        buf_reader.consume(data_len);
    }

    Ok(())
}

/// Iterator over file tokens (words)
pub struct FileTokenIterator {
    file: File,
    start_pos: usize,
    end_pos: usize,
    read_buffer: Vec<u8>,
    rest: Vec<u8>,
    output_separator: bool,
}

impl Iterator for FileTokenIterator {
    type Item = String;
    fn next(&mut self) -> Option<Self::Item> {
        self.read_token()
    }
}

fn vec_to_string_opt(v: &[u8]) -> Option<String> {
    Some(String::from_utf8(Vec::from(v)).unwrap_or_else(|_| String::from("<INV>")))
}

const READ_BUFFER_SIZE: usize = 8192;
const MAX_TOKEN_LEN: usize = 64;

impl FileTokenIterator {
    /// Construct a FileTokenIterator, iteration begins at byte `offset` in the file
    pub fn new(file_name: &str, offset: u64) -> std::io::Result<FileTokenIterator> {
        let mut result = FileTokenIterator {
            file: File::open(file_name)?,
            start_pos: 0,
            end_pos: 0,
            read_buffer: Vec::with_capacity(READ_BUFFER_SIZE),
            rest: Vec::new(),
            output_separator: false,
        };
        result.read_buffer.resize(READ_BUFFER_SIZE, 0);
        result.file.seek(SeekFrom::Start(offset))?;

        Ok(result)
    }

    /// Re-start iteration from the given offset
    pub fn reset(&mut self, offset: u64) -> std::io::Result<()> {
        self.file.seek(SeekFrom::Start(offset))?;
        self.start_pos = 0;
        self.end_pos = 0;
        self.rest.clear();
        self.output_separator = false;
        Ok(())
    }

    /// Read and return the next token from the file
    pub fn read_token(&mut self) -> Option<String> {
        if self.output_separator {
            self.output_separator = false;
            return Some(String::from("</s>"));
        }

        let mut output: Option<String> = None;

        'readloop: loop {
            // Read data if read buffer is empty
            if self.start_pos == self.end_pos {
                self.start_pos = 0;
                self.end_pos = self.file.read(&mut self.read_buffer[..]).unwrap_or(0);
                if self.end_pos == 0 {
                    if !self.rest.is_empty() {
                        output = vec_to_string_opt(&self.rest);
                        self.rest.clear();
                    }
                    break 'readloop output;
                }
            }

            // Check if we have trailing data from a previous read. If we can form a
            // token together with the current read, return it
            if !self.rest.is_empty() {
                if let Some(pos) = self.read_buffer[self.start_pos..self.end_pos]
                    .iter()
                    .position(is_token_separator)
                {
                    self.rest
                        .extend_from_slice(&self.read_buffer[self.start_pos..self.start_pos + pos]);

                    if is_doc_separator(&self.read_buffer[self.start_pos + pos]) {
                        self.output_separator = true;
                    }

                    self.start_pos += pos + 1;
                    output = vec_to_string_opt(&self.rest);
                    self.rest.clear();
                    break 'readloop output;
                }
            }

            let mut token_start: usize = self.start_pos;
            let mut token_end: usize = token_start;

            for byte in &self.read_buffer[self.start_pos..self.end_pos] {
                if !is_token_separator(byte) {
                    token_end += 1;
                    continue;
                }

                if is_doc_separator(byte) {
                    self.output_separator = true;
                }

                if token_end == token_start {
                    // empty token, skip it
                    token_end += 1;
                    token_start = token_end;
                    continue;
                }

                output = vec_to_string_opt(&self.read_buffer[token_start..token_end]);
                self.start_pos = token_end + 1;
                break 'readloop output;
            }

            if token_end > token_start {
                self.rest
                    .extend_from_slice(&self.read_buffer[token_start..token_end]);
                self.start_pos = self.end_pos;
                if self.rest.len() < MAX_TOKEN_LEN {
                    continue 'readloop;
                }

                output = vec_to_string_opt(&self.rest);
                self.rest.clear();
                break 'readloop output;
            }

            // set start_pos == end_pos to force reading in the next iteration
            self.start_pos = token_end;
            self.end_pos = token_end;
        }
    }
}
