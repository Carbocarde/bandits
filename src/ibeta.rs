extern crate test;
#[cfg(test)]
use float_cmp::approx_eq;
#[cfg(test)]
use test::{black_box, Bencher};

#[test]
fn beta_inverse() {
    let a = 2;
    let b = 1;
    let p = 0.5;

    let result = puruspe::invbetai(p, a as f64, b as f64);

    println!(
        "Total percentage of area at point {}: {:.2}%",
        result,
        p * 100.0
    );

    assert!(approx_eq!(f64, result, 0.707106781186548, ulps = 100_000));
}

#[test]
fn beta_inverse_half() {
    let a = 1000;
    let b = 1000;
    let p = 0.5;

    let result = puruspe::invbetai(p, (a + 1) as f64, (b + 1) as f64);

    println!(
        "Total percentage of area at point {}: {:.2}%",
        result * 100.0,
        p * 100.0
    );

    let inv = puruspe::betai((a + 1) as f64, (b + 1) as f64, result);

    println!("Inverse {}: {:.2}%", result * 100.0, inv * 100.0);

    assert!(approx_eq!(f64, result, 0.5, ulps = 100_000));
}

#[bench]
fn basic_benchmark(ben: &mut Bencher) {
    let a = 2.0;
    let b = 1.0;
    let p = 0.5;

    ben.iter(|| {
        let result = puruspe::invbetai(p, a, b);
        black_box(result);
    });
}
