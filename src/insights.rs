use colored::Colorize;
use log::error;
use ordered_float::NotNan;
use rgb::RGB8;
use textplots::ColorPlot;

use crate::{
    thompson::{
        dist_area_at_percentile, skew_percentile, thompson_ranking, thompson_ranking_bias_runtime,
        ThompsonInfo,
    },
    Script,
};

pub fn plot_top_3(scripts: &[Script]) {
    if scripts.len() < 3 {
        error!("Cannot plot top 3 with less than 3 scripts.");
        return;
    }

    let mut scripts = scripts.to_owned();

    scripts.sort_by(|a, b| b.runcount.partial_cmp(&a.runcount).unwrap());

    let most_run_scripts: Vec<&Script> = scripts.iter().take(3).collect();

    use textplots::{Chart, Shape};

    println!("Plot of top 3 run scripts. Interesting cases (area under curve).");

    let colors = [
        RGB8 {
            r: 100,
            g: 250,
            b: 200,
        },
        RGB8 {
            r: 200,
            g: 250,
            b: 100,
        },
        RGB8 {
            r: 200,
            g: 100,
            b: 250,
        },
    ];

    Chart::new(120, 60, 0.0, 1.0)
        .linecolorplot(
            &Shape::Continuous(Box::new(|x| {
                puruspe::betai(
                    (most_run_scripts[2].results.uninteresting + 1) as f64,
                    (most_run_scripts[2].results.interesting + 1) as f64,
                    x.into(),
                ) as f32
            })),
            colors[2],
        )
        .linecolorplot(
            &Shape::Continuous(Box::new(|x| {
                puruspe::betai(
                    (most_run_scripts[1].results.uninteresting + 1) as f64,
                    (most_run_scripts[1].results.interesting + 1) as f64,
                    x.into(),
                ) as f32
            })),
            colors[1],
        )
        .linecolorplot(
            &Shape::Continuous(Box::new(|x| {
                puruspe::betai(
                    (most_run_scripts[0].results.uninteresting + 1) as f64,
                    (most_run_scripts[0].results.interesting + 1) as f64,
                    x.into(),
                ) as f32
            })),
            colors[0],
        )
        .display();

    println!("Top 3 run scripts:");
    println!(
        "1: {} {} {}ms",
        most_run_scripts[0].runcount,
        most_run_scripts[0]
            .name
            .truecolor(colors[0].r, colors[0].g, colors[0].b),
        most_run_scripts[0]
            .avgruntime_ms
            .unwrap_or(NotNan::new(-1.0).unwrap())
    );
    println!(
        "2: {} {} {}ms",
        most_run_scripts[1].runcount,
        most_run_scripts[1]
            .name
            .truecolor(colors[1].r, colors[1].g, colors[1].b),
        most_run_scripts[1]
            .avgruntime_ms
            .unwrap_or(NotNan::new(-1.0).unwrap())
    );
    println!(
        "3: {} {} {}ms",
        most_run_scripts[2].runcount,
        most_run_scripts[2]
            .name
            .truecolor(colors[2].r, colors[2].g, colors[2].b),
        most_run_scripts[2]
            .avgruntime_ms
            .unwrap_or(NotNan::new(-1.0).unwrap())
    );
}

pub fn plot_top_3_inverses(scripts: &[Script]) {
    if scripts.len() < 3 {
        println!("Cannot plot the top 3 inverses with less than 3 scripts.");
        return;
    }

    let mut scripts = scripts.to_owned();

    scripts.sort_by(|a, b| b.runcount.partial_cmp(&a.runcount).unwrap());

    let most_run_scripts: Vec<&Script> = scripts.iter().take(3).collect();

    use textplots::{Chart, Shape};

    println!(
        "Plot of inverse 3 run scripts. Minimizing time per interesting case (area under curve)."
    );

    let colors = [
        RGB8 {
            r: 100,
            g: 250,
            b: 200,
        },
        RGB8 {
            r: 200,
            g: 250,
            b: 100,
        },
        RGB8 {
            r: 200,
            g: 100,
            b: 250,
        },
    ];

    Chart::new(120, 60, 0.0, 1.0)
        .linecolorplot(
            &Shape::Continuous(Box::new(|x| {
                f32::from(
                    skew_percentile(
                        NotNan::new(puruspe::invbetai(
                            x.into(),
                            (most_run_scripts[2].results.interesting + 1) as f64,
                            (most_run_scripts[2].results.uninteresting + 1) as f64,
                        ))
                        .unwrap(),
                        &most_run_scripts[2].avgruntime_ms,
                        &most_run_scripts[2].bias,
                    )
                    .as_f32(),
                )
            })),
            colors[2],
        )
        .linecolorplot(
            &Shape::Continuous(Box::new(|x| {
                f32::from(
                    skew_percentile(
                        NotNan::new(puruspe::invbetai(
                            x.into(),
                            (most_run_scripts[1].results.interesting + 1) as f64,
                            (most_run_scripts[1].results.uninteresting + 1) as f64,
                        ))
                        .unwrap(),
                        &most_run_scripts[1].avgruntime_ms,
                        &most_run_scripts[1].bias,
                    )
                    .as_f32(),
                )
            })),
            colors[1],
        )
        .linecolorplot(
            &Shape::Continuous(Box::new(|x| {
                f32::from(
                    skew_percentile(
                        NotNan::new(puruspe::invbetai(
                            x.into(),
                            (most_run_scripts[0].results.interesting + 1) as f64,
                            (most_run_scripts[0].results.uninteresting + 1) as f64,
                        ))
                        .unwrap(),
                        &most_run_scripts[0].avgruntime_ms,
                        &most_run_scripts[0].bias,
                    )
                    .as_f32(),
                )
            })),
            colors[0],
        )
        .display();

    println!("Top 3 run scripts:");
    println!(
        "1: {} {} {}ms",
        most_run_scripts[0].runcount,
        most_run_scripts[0]
            .name
            .truecolor(colors[0].r, colors[0].g, colors[0].b),
        most_run_scripts[0]
            .avgruntime_ms
            .unwrap_or(NotNan::new(-1.0).unwrap())
    );
    println!(
        "2: {} {} {}ms",
        most_run_scripts[1].runcount,
        most_run_scripts[1]
            .name
            .truecolor(colors[1].r, colors[1].g, colors[1].b),
        most_run_scripts[1]
            .avgruntime_ms
            .unwrap_or(NotNan::new(-1.0).unwrap())
    );
    println!(
        "3: {} {} {}ms",
        most_run_scripts[2].runcount,
        most_run_scripts[2]
            .name
            .truecolor(colors[2].r, colors[2].g, colors[2].b),
        most_run_scripts[2]
            .avgruntime_ms
            .unwrap_or(NotNan::new(-1.0).unwrap())
    );
}

pub fn print_ranking_bias_runtime(
    scripts: &[Script],
    runtimes: &[&Option<NotNan<f64>>],
    user_biases: &[&NotNan<f64>],
    verbose: bool,
) {
    let items = scripts
        .iter()
        .filter(|x| x.limit.is_none() || x.limit.unwrap() < x.results.interesting)
        .map(|x| &x.results)
        .collect::<Vec<_>>();
    let entries: &[&ThompsonInfo] = items.as_slice();
    let ranking = thompson_ranking_bias_runtime(entries, runtimes, user_biases);

    if verbose {
        println!("Ranking (biased by runtime):");

        for (i, script) in ranking.iter().enumerate() {
            println!("{}: {}", i + 1, scripts[*script].name,);
            println!(
                "- 50th percentile: {:.4}",
                dist_area_at_percentile(&scripts[*script].results, 0.5)
            );
            println!("- Runs: {}", &scripts[*script].runcount);
            println!(
                "- Observed percent {:.5}%",
                scripts[*script].results.interesting as f64 / scripts[*script].runcount as f64
                    * 100.
            )
        }
    } else {
        ranking.iter().for_each(|script| {
            println!("{}", scripts[*script].name);
        });
    }
}

pub fn print_ranking(scripts: &[Script], verbose: bool) {
    let items = scripts
        .iter()
        .filter(|x| x.limit.is_none() || x.limit.unwrap() < x.results.interesting)
        .map(|x| &x.results)
        .collect::<Vec<_>>();
    let entries: &[&ThompsonInfo] = items.as_slice();
    let ranking = thompson_ranking(entries);

    if verbose {
        println!("Ranking (raw):");

        for (i, script) in ranking.iter().enumerate() {
            println!("{}: {}", i + 1, scripts[*script].name,);
            println!(
                "- 50th percentile: {:.4}",
                dist_area_at_percentile(&scripts[*script].results, 0.5)
            );
            println!("- Runs: {}", &scripts[*script].runcount);
        }
    } else {
        ranking.iter().for_each(|script| {
            println!("{}", scripts[*script].name);
        });
    }
}
