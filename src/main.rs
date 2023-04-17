use crate::linear_regression::{linear_regression, linear_regression_fn_generator};
use crate::maths::{covariance, variance};

mod linear_regression;
mod maths;

fn main() {
    let x: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];

    let predictor = linear_regression_fn_generator(&x, &y);

    println!("value predicted : {:?}", predictor(10.0));

    let linear_regression = linear_regression(&x, &y);
    println!(
        "value predicted : {:?}",
        linear_regression.predict_value(10.0)
    );
    println!(
        "vector prediction : {:?}",
        linear_regression.predict_vector(&x)
    );
}
