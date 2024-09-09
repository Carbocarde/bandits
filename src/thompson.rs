use log::debug;
use ordered_float::NotNan;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ThompsonInfo {
    pub interesting: u64,
    pub uninteresting: u64,
}

pub fn skew_percentile(
    sampled_point: NotNan<f64>,
    runtime: &Option<NotNan<f64>>,
    user_bias: &NotNan<f64>,
) -> NotNan<f64> {
    if let Some(runtime) = runtime {
        let time_scaler = NotNan::new(100.0).unwrap() / runtime;

        // A script with bias of 5 is weighted to be equal to an equivalent script that runs 5x as fast.
        sampled_point * time_scaler * user_bias
    } else {
        // If we don't know a runtime, return the max score so we sample the runtime at least once.
        NotNan::new(f64::MAX).unwrap()
    }
}

/// Prefer entries with low runtime.
/// Entries without a specified runtime will always be run first.
pub fn thompson_sampling_bias_runtime(
    entries: &[&ThompsonInfo],
    runtimes: &[&Option<NotNan<f64>>],
    user_biases: &[&NotNan<f64>],
) -> Option<usize> {
    let mut selected_entry_index: Option<usize> = None;
    let mut selected_entry_percentile: NotNan<f64> = NotNan::new(-1.0).unwrap();
    for (index, entry) in entries.iter().enumerate() {
        let skewed_percentile = thompson_step_bias_runtime(
            entry.interesting,
            entry.uninteresting,
            runtimes[index],
            user_biases[index],
        );

        if skewed_percentile > selected_entry_percentile {
            selected_entry_index = Some(index);
            selected_entry_percentile = skewed_percentile;
        }
    }
    debug!("Selected entry: {:?}", selected_entry_index);

    selected_entry_index
}

/// Returns a vector mapping the nth selected entry to its index.
///
/// Ex. [0, 2, 1]: The first element was ranked first, the third second, and second third.
pub fn thompson_ranking_bias_runtime(
    entries: &[&ThompsonInfo],
    runtimes: &[&Option<NotNan<f64>>],
    user_biases: &[&NotNan<f64>],
) -> Vec<usize> {
    let mut percentiles_index_mapping = entries
        .iter()
        .enumerate()
        .map(|(idx, entry)| {
            (
                idx,
                thompson_step_bias_runtime(
                    entry.interesting,
                    entry.uninteresting,
                    runtimes[idx],
                    user_biases[idx],
                ),
            )
        })
        .collect::<Vec<_>>();

    percentiles_index_mapping.sort_by_key(|&(_, percentile)| percentile);

    percentiles_index_mapping
        .iter()
        .rev()
        .map(|(index, _percentile)| *index)
        .collect()
}

/// Map a single entry into a score comparable to other entries.
fn thompson_step_bias_runtime(
    interesting: u64,
    uninteresting: u64,
    runtime: &Option<NotNan<f64>>,
    user_bias: &NotNan<f64>,
) -> NotNan<f64> {
    let mut rng = rand::thread_rng();
    // Random number from 0.0 to 1.0 inclusive
    let random_float = rng.gen_range(0.0..1.0);

    let percentile = puruspe::invbetai(
        random_float,
        (interesting + 1) as f64,
        (uninteresting + 1) as f64,
    );

    let skewed_percentile = skew_percentile(NotNan::new(percentile).unwrap(), runtime, user_bias);

    debug!(
        "Total percentage of area at point {:.4}: {:.2}% B({}, {}) Skewed area: {:.2}",
        random_float * 100.0,
        percentile,
        (uninteresting + 1) as f64,
        (interesting + 1) as f64,
        skewed_percentile
    );

    skewed_percentile
}

/// Perform thompson sampling and pick a single entry. Ignores runtime.
pub fn thompson_sampling(entries: &[&ThompsonInfo], user_biases: &[&NotNan<f64>]) -> Option<usize> {
    let mut selected_entry_index: Option<usize> = None;
    let mut selected_entry_percentile: NotNan<f64> = NotNan::new(-1.0).unwrap();
    for (index, entry) in entries.iter().enumerate() {
        let mut percentile = thompson_step(entry.interesting, entry.uninteresting);
        debug!(
            "Total percentage of area at random point {:.2}%",
            percentile * 100.,
        );
        percentile *= f64::from(*user_biases[index]);

        if percentile > selected_entry_percentile {
            selected_entry_index = Some(index);
            selected_entry_percentile = percentile
        }
    }
    selected_entry_index
}

/// Returns a vector mapping the nth selected entry to its index.
///
/// Ex. [0, 2, 1]: The first element was ranked first, the third second, and second third.
pub fn thompson_ranking(entries: &[&ThompsonInfo]) -> Vec<usize> {
    let mut percentiles_index_mapping = entries
        .iter()
        .enumerate()
        .map(|(idx, entry)| (idx, thompson_step(entry.interesting, entry.uninteresting)))
        .collect::<Vec<_>>();

    percentiles_index_mapping.sort_by_key(|&(_, percentile)| percentile);

    percentiles_index_mapping
        .iter()
        .rev()
        .map(|(index, _percentile)| *index)
        .collect()
}

fn thompson_step(interesting: u64, uninteresting: u64) -> NotNan<f64> {
    let mut rng = rand::thread_rng();
    // Random number from 0.0 to 1.0 inclusive
    let random_float: f64 = rng.gen_range(0.0..1.0);
    debug!("Percentile to sample: {}", random_float);
    let percentile = puruspe::invbetai(
        random_float,
        (interesting + 1) as f64,
        (uninteresting + 1) as f64,
    );
    debug!(
        "Total percentage of area at point {:.4}: {:.2}% B({}, {})",
        random_float * 100.0,
        percentile,
        (interesting + 1) as f64,
        (uninteresting + 1) as f64
    );
    NotNan::new(percentile).unwrap()
}

/// Returns the nth percentile of the beta distribution.
pub fn dist_area_at_percentile(entry: &ThompsonInfo, area: f64) -> f64 {
    puruspe::invbetai(
        area,
        (entry.interesting + 1) as f64,
        (entry.uninteresting + 1) as f64,
    )
}

#[test]
fn test_thompson_sampling_none() {
    assert_eq!(thompson_sampling(&vec![], &vec![]), None);
}

#[test]
fn test_thompson_sampling_one() {
    assert_eq!(
        thompson_sampling(
            &[&ThompsonInfo {
                interesting: 0,
                uninteresting: 0
            }],
            &[&NotNan::new(1.0).unwrap(), &NotNan::new(1.0).unwrap()]
        ),
        Some(0)
    );
}

#[test]
fn test_thompson_sampling_prefer_interesting() {
    assert_eq!(
        thompson_sampling(
            &[
                &ThompsonInfo {
                    interesting: 0,
                    uninteresting: 100,
                },
                &ThompsonInfo {
                    interesting: 100,
                    uninteresting: 0
                }
            ],
            &[&NotNan::new(1.0).unwrap(), &NotNan::new(1.0).unwrap()]
        ),
        Some(1)
    );
}

#[test]
fn test_thompson_sampling_bias_prefer_fast() {
    assert_eq!(
        thompson_sampling_bias_runtime(
            &[
                &ThompsonInfo {
                    interesting: 100,
                    uninteresting: 100
                },
                &ThompsonInfo {
                    interesting: 100,
                    uninteresting: 100
                }
            ],
            &[
                &Some(NotNan::new(1.0).unwrap()),
                &Some(NotNan::new(100.0).unwrap())
            ],
            &[&NotNan::new(1.0).unwrap(), &NotNan::new(1.0).unwrap()]
        ),
        Some(0)
    );
}

#[test]
fn test_thompson_sampling_bias_prefer_unknown() {
    assert_eq!(
        thompson_sampling_bias_runtime(
            &[
                &ThompsonInfo {
                    interesting: 100,
                    uninteresting: 0
                },
                &ThompsonInfo {
                    interesting: 0,
                    uninteresting: 0
                }
            ],
            &[&Some(NotNan::new(1.0).unwrap()), &None],
            &[&NotNan::new(1.0).unwrap(), &NotNan::new(1.0).unwrap()]
        ),
        Some(1)
    );
}
