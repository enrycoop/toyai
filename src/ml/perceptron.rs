//! Perceptron classifier
//!
//! Parameters
//! -------------
//! - `eta`: Learning rate (between 0.0 and 1.0)
//! - `n_iter`: Passes over the training dataset.
//!
//! Attributes
//! -------------
//! - `weights`: weights after fitting
//! - `errors`: number of misclassifications in every epoch
//! 
use std::error::Error;

pub struct Perceptron {
    eta: f64,
    n_iter: u32,
    weights: Vec<f64>,
    pub errors: Vec<u32>,
}

impl Perceptron {
    pub fn new(_eta: f64, _n_iter: u32) -> Self {
        Perceptron {
            eta: _eta,
            n_iter: _n_iter,
            weights: vec![],
            errors: vec![],
        }
    }

    fn feature_size_check(&self, x: &Vec<Vec<f64>>, y: &Vec<f64>) -> Result<bool, Box<dyn Error>> {
        if x.is_empty() {
            return Err("X is empty".into());
        }

        if x.len() != y.len() {
            let x_len = x.len();
            let y_len = y.len();
            return Err(format!("Size of X [{x_len}] don't match with y[{y_len}] size").into());
        }

        let feature_size = x.get(0).unwrap().len();
        let mut i = 1;
        for row in x {
            if row.len() != feature_size {
                return Err(format!("Size of X[{i}] is inconsistent").into());
            }
            i = i + 1;
        }

        Ok(true)
    }

    fn net_input(&self, x: &Vec<f64>) -> f64 {
        let mut result: f64 = 0.0;
        if self.weights.len() - 1 != x.len() {
            panic!("size of w and x mismatch");
        }

        for i in 0..x.len() {
            result += x[i] * self.weights[i + 1];
        }
        result + self.weights[0]
    }

    /// Return class label after unit step
    pub fn predict(&self, x: &Vec<f64>) -> f64 {
        if self.net_input(x) >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    /// Fit training data.
    /// Parameters
    /// ------------
    /// - `X`(n_samples, n_features):
    ///         Training vectors, where n_samples
    ///         is the number of samples and
    ///         n_features is the number of features.
    /// - `y`(n_samples): 
    ///         Target values.
    ///
    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) -> Result<bool, Box<dyn Error>> {
        match self.feature_size_check(x, y) {
            Ok(_) => (),
            Err(e) => return Err(e),
        }

        self.weights = vec![0.0; x[0].len() + 1];
        self.errors = Vec::new();

        for epoch in 0..self.n_iter {
            // Epoche
            print!("Epoch: {epoch}");
            let mut errors: u32 = 0;
            for i_sample in 0..x.len() {
                // Calcolo l'aggiornamento con la distanza tra la classe
                // e il valore predetto per un certo coeff.
                let update = self.eta * (y[i_sample] - self.predict(&x[i_sample]));

                // aggiorno i pesi in weights
                for i_weight in 1..self.weights.len() {
                    self.weights[i_weight] += update * x[i_sample][i_weight - 1];
                }
                self.weights[0] += update;

                errors += if update != 0.0 { 1 } else { 0 };
            }
            println!(" Errors: {errors}");
            self.errors.push(errors);
        }

        Ok(true)
    }
}
