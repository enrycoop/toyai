/// Dataset struct is specific for iris dataset actually
///
use std::fs;

pub struct Dataset {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
}

impl Dataset {
    pub fn from(file_path: &str) -> Self {
        let contents = fs::read_to_string(file_path)
            .expect(format!("Impossibile leggere il file: {file_path}").as_str());
        let lines: Vec<&str> = contents.lines().collect();
        let mut x: Vec<Vec<f64>> = Vec::new(); // TODO: add capacity allocation
        let mut y: Vec<f64> = Vec::new();

        for row_index in 0..lines.len() {
            let tokens: Vec<&str> = lines[row_index].split(",").collect();
            // Extract and parsing features
            let mut row: Vec<f64> = Vec::new();
            for col_index in 0..tokens.len() - 1 {
                let value: f64 = match tokens[col_index].parse() {
                    Ok(num) => num,
                    Err(e) => panic!(
                        "Error during parsing of value in row [{row_index}] col [{col_index}]: {e}"
                    ),
                };
                row.push(value);
            }
            x.push(row);

            // Extract and parsing labels
            y.push(match tokens[tokens.len() - 1].parse() {
                Ok(num) => num,
                Err(e) => panic!(
                    "Error during parsing of last value in row [{row_index}]: {e}"
                ),
            });
        }
        Dataset { x, y }
    }
}
