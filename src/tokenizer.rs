use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn read_file_by_tokens<T: FnMut(&[u8])>(
    file_name: &str,
    mut token_callback: T,
) -> std::io::Result<()> {
    let is_separator =
        |byte: &u8| *byte == b'\n' || *byte == b' ' || *byte == b'\t' || *byte == b'\r';
    let mut buf_reader: BufReader<File> = BufReader::new(File::open(file_name)?);
    let mut rest: Vec<u8> = Vec::new();

    loop {
        let data: &[u8] = buf_reader.fill_buf()?;
        if data.is_empty() {
            break;
        }

        let data_len = data.len();
        let mut skip_bytes: usize = 0;

        if !rest.is_empty() {
            if let Some(pos) = data.iter().position(is_separator) {
                rest.extend_from_slice(&data[..pos]);
                token_callback(&rest);
                rest.clear();
                if data[pos] == b'\n' {
                    token_callback(b"</s>")
                }
                skip_bytes = pos + 1;
            }
        }

        let mut token_start: usize = skip_bytes;
        let mut token_end: usize = token_start;

        for byte in &data[skip_bytes..] {
            if is_separator(byte) {
                if token_end > token_start {
                    token_callback(&data[token_start..token_end]);
                }
                token_end += 1;
                token_start = token_end;

                if *byte == b'\n' {
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
