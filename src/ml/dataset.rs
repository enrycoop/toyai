/// Dataset struct is specific for iris dataset actually
/// 

use std::fs;

pub struct Dataset {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
}

impl Dataset {

    pub fn read_file(file_path: &str) -> Vec<Vec<String>> {
        let contents = fs::read_to_string(file_path)
            .expect(format!("Impossible to read file: {file_path}").as_str());        

        let mut tokens: Vec<Vec<String>> = Vec::new();
        for line in contents.lines() {
            let mut row: Vec<String> = Vec::new();
            for token in line.split(",") {
                row.push(String::from(token));
                print!("{} ", token);
            }
            tokens.push(row);
            println!();
        }
        tokens
    }

    pub fn build(file_path: &str, categorical: Vec<bool>){
        let tokens = Dataset::read_file(file_path);
        for line in tokens {
           
        }
    }

    pub fn from(file_path: &str) -> Self {
        let contents = fs::read_to_string(file_path)
            .expect(format!("Impossibile leggere il file: {file_path}").as_str());
        let lines: Vec<&str> = contents.lines().collect();
        let mut x: Vec<Vec<f64>> = Vec::new(); // TODO: aggiungere allocazione della capacita'
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
            y.push(match tokens[tokens.len() - 1] {
                "Iris-setosa" => 1.0,
                "Iris-versicolor" => -1.0,
                _ => panic!("Valore non gestito"),
            });
        }
        Dataset { x, y }
    }
}