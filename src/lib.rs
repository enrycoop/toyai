//! # TOYAI
//! 
//! `toyai` is a small collection of ai algorithms to perform some simple prediction
//! on structured data.
//! 
//! 


/// IRIS DATASET for tests 
/// Relevant Information:
///    --- This is perhaps the best known database to be found in the pattern
///        recognition literature.  Fisher's paper is a classic in the field
///        and is referenced frequently to this day.  (See Duda & Hart, for
///        example.)  The data set contains 3 classes of 50 instances each,
///        where each class refers to a type of iris plant.  One class is
///        linearly separable from the other 2; the latter are NOT linearly
///        separable from each other.
///    --- Predicted attribute: class of iris plant.
///    --- This is an exceedingly simple domain.
///    --- This data differs from the data presented in Fishers article
/// 	(identified by Steve Chadwick,  spchadwick@espeedaz.net )
/// 	The 35th sample should be: 4.9,3.1,1.5,0.2,"Iris-setosa"
/// 	where the error is in the fourth feature.
/// 	The 38th sample: 4.9,3.6,1.4,0.1,"Iris-setosa"
/// 	where the errors are in the second and third features. 
/// Number of Instances: 150 (50 in each of three classes)
/// 
/// Number of Attributes: 4 numeric, predictive attributes and the class
/// 
/// Attribute Information:
///    1. sepal length in cm
///    2. sepal width in cm
///    3. petal length in cm
///    4. petal width in cm
///    5. class:
///       -- Iris Setosa
///       -- Iris Versicolour
///       -- Iris Virginica
/// 
/// Missing Attribute Values: None
/// 
/// Summary Statistics:
///            Min  Max   Mean    SD   Class Correlation
///    sepal length: 4.3  7.9   5.84  0.83    0.7826
///     sepal width: 2.0  4.4   3.05  0.43   -0.4194
///    petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
///     petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)
/// 
/// Class Distribution: 33.3% for each of 3 classes.
/// 
/// 
/// NOTE: for the experiments Iris virginica was removed

mod ml;
pub use crate::ml::{dataset, perceptron};

#[cfg(test)]
mod perceptron_tests {
    use perceptron::Perceptron;

    use super::*;

    #[test]
    fn it_works() {
        let file_path = "data/iris.data";

        let dataset = dataset::Dataset::from(&file_path);
        let mut model = Perceptron::new(0.1, 10);

        model
            .fit(&(dataset.x), &(dataset.y))
            .expect("Error during learning model.");
        assert!(model.errors[9] == 0);
    }
}
