#![feature(test)]
mod config;
mod ibeta;
mod insights;
mod thompson;

use argh::FromArgs;
use config::{parse_config, save_config, Config, Script};
use insights::{plot_top_3, plot_top_3_inverses, print_ranking, print_ranking_bias_runtime};
use log::{debug, trace, warn};
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, process::Command, time::Instant};
use thompson::{thompson_sampling, thompson_sampling_bias_runtime, ThompsonInfo};

fn choose_script(config: &Config, ignore_runtime: bool) -> usize {
    let items = config
        .scripts
        .iter()
        .filter(|x| x.limit.is_none() || x.limit.unwrap() < x.results.interesting)
        .map(|x| &x.results)
        .collect::<Vec<_>>();
    let entries: &[&ThompsonInfo] = items.as_slice();
    let runtime = config
        .scripts
        .iter()
        .filter(|x| x.limit.is_none() || x.limit.unwrap() < x.results.interesting)
        .map(|x| &x.avgruntime_ms)
        .collect::<Vec<_>>();
    let runtimes: &[&Option<NotNan<f64>>] = runtime.as_slice();

    let user_biases = config.scripts.iter().map(|x| &x.bias).collect::<Vec<_>>();
    let user_biases: &[&NotNan<f64>] = user_biases.as_slice();

    if ignore_runtime {
        thompson_sampling(entries, user_biases).unwrap()
    } else {
        thompson_sampling_bias_runtime(entries, runtimes, user_biases).unwrap()
    }
}

fn run_script(script: &Script) -> ScriptResult {
    let mut parts = script.command.split_whitespace();
    // Get the command (first part)
    let command = parts.next().expect("No command provided");

    // Get command arguments
    let args: Vec<&str> = parts.collect();

    // Execute the command
    let start = Instant::now();
    let output = Command::new(command)
        .args(&args)
        .output()
        .expect("Failed to execute command");
    let duration = start.elapsed();

    // Print the output
    if output.status.success() {
        trace!("Command executed successfully!");
        trace!("Output: {}", String::from_utf8_lossy(&output.stdout));
        ScriptResult {
            interesting: 0,
            uninteresting: 1,
            runtime_ms: duration.as_millis(),
        }
    } else if output.status.code() == Some(1) {
        warn!("Command failed with exit status 1, logging as interesting.");
        warn!("Error: {}", String::from_utf8_lossy(&output.stderr));
        ScriptResult {
            interesting: 1,
            uninteresting: 0,
            runtime_ms: duration.as_millis(),
        }
    } else {
        println!("Command failed with unknown exit status: {}", output.status);
        println!("Error: {}", String::from_utf8_lossy(&output.stderr));
        println!("Unrecognized exit status. Adding to avg iteration runtime but not logging as interesting/uninteresting");
        ScriptResult {
            interesting: 0,
            uninteresting: 0,
            runtime_ms: duration.as_millis(),
        }
    }
}

fn update_state(existing_results: &mut Script, result: ScriptResult) {
    let results = ThompsonInfo {
        interesting: existing_results.results.interesting + result.interesting,
        uninteresting: existing_results.results.uninteresting + result.uninteresting,
    };

    let total_runtime = existing_results
        .avgruntime_ms
        .unwrap_or(NotNan::new(0.0).unwrap())
        * existing_results.runcount as f64;
    existing_results.runcount += 1;
    existing_results.avgruntime_ms =
        Some((total_runtime + result.runtime_ms as f64) / existing_results.runcount as f64);
    existing_results.results = results;
}

fn reset_state(config: &mut Config, script_name: Option<String>) {
    if let Some(script_name) = script_name {
        if config
            .scripts
            .iter()
            .any(|script| script.name == script_name)
        {
            panic!("Could not find specified script {script_name} to reset")
        }

        // Reset only the specified script
        config.scripts = config
            .scripts
            .clone()
            .into_iter()
            .map(|mut script| {
                if script.name == script_name {
                    let results = ThompsonInfo {
                        interesting: 0,
                        uninteresting: 0,
                    };

                    script.runcount = 0;
                    script.results = results;
                    script.avgruntime_ms = None;
                    script
                } else {
                    // Leave untouched
                    script
                }
            })
            .collect();
    } else {
        // Reset all scripts
        config.scripts = config
            .scripts
            .clone()
            .into_iter()
            .map(|mut script| {
                let results = ThompsonInfo {
                    interesting: 0,
                    uninteresting: 0,
                };

                script.runcount = 0;
                script.results = results;
                script.avgruntime_ms = None;
                script
            })
            .collect();
    }
}

#[derive(Debug)]
struct ScriptResult {
    interesting: u64,
    uninteresting: u64,
    runtime_ms: u128,
}

fn step(config: &mut Config, ignore_runtime: bool) {
    if config.scripts.is_empty() {
        debug!("ERROR: No scripts to execute. Exiting...");
        return;
    }

    let script_index = choose_script(config, ignore_runtime);

    debug!("Running script {}...", script_index);

    let result = run_script(&config.scripts[script_index]);

    debug!("Script {} finished. Result: {:?}", script_index, result);

    update_state(config.scripts.get_mut(script_index).unwrap(), result);
}

#[derive(FromArgs, Debug)]
/**
Biased Thompson Sampling for Multi Armed Bandit.
Prioritize bandits according to their likelihood to be interesting and runtime.
*/
struct Args {
    #[argh(subcommand)]
    subcommand: SubCommands,
}

#[derive(Debug, Serialize, Deserialize, FromArgs, PartialEq)]
#[argh(subcommand)]
enum SubCommands {
    New(NewOptions),
    Run(RunOptions),
    Rank(RankOptions),
    Reset(ResetOptions),
    Summarize(SummarizeOptions),
    Lint(LintOptions),
}

#[derive(Debug, Serialize, Deserialize, FromArgs, PartialEq)]
/// Repeatedly prioritize & run bandits according to their likelihood to discover interesting cases.
#[argh(subcommand, name = "run")]
struct RunOptions {
    /// list of scripts to execute
    #[argh(positional)]
    config: PathBuf,

    /// output location for updated config
    #[argh(option, short = 'o', default = "PathBuf::from(\"./new-config.json\")")]
    output: PathBuf,

    /// number of command invocations to perform
    #[argh(option, default = "10")]
    steps: usize,

    /// ignore runtime when ranking scripts
    #[argh(switch, short = 'i')]
    ignore_runtime: bool,
}

#[derive(Debug, Serialize, Deserialize, FromArgs, PartialEq)]
/// Summarize the config file
#[argh(subcommand, name = "summarize")]
struct SummarizeOptions {
    /// list of scripts to execute
    #[argh(positional)]
    config: PathBuf,

    /// ignore runtime when ranking scripts
    #[argh(switch, short = 'i')]
    ignore_runtime: bool,
}

#[derive(Debug, Serialize, Deserialize, FromArgs, PartialEq)]
/// Repeatedly prioritize & run bandits according to their likelihood to discover interesting cases.
#[argh(subcommand, name = "rank")]
struct RankOptions {
    /// list of scripts to rank
    #[argh(positional)]
    config: PathBuf,

    /// ignore runtime when ranking scripts
    #[argh(switch, short = 'i')]
    ignore_runtime: bool,

    /// verbose
    #[argh(switch, short = 'v')]
    verbose: bool,
}

#[derive(Debug, Serialize, Deserialize, FromArgs, PartialEq)]
/// Reset a config file to clear runtime data.
#[argh(subcommand, name = "reset")]
struct ResetOptions {
    /// list of scripts to rank
    #[argh(positional)]
    config: PathBuf,

    /// only reset this script
    #[argh(option)]
    script: Option<String>,

    /// output location for reset config
    #[argh(option, short = 'o', default = "PathBuf::from(\"./new-config.json\")")]
    output: PathBuf,
}

#[derive(Debug, Serialize, Deserialize, FromArgs, PartialEq)]
/// Create a new config file for the given list of scripts
#[argh(subcommand, name = "new")]
struct NewOptions {
    /// output location for new config
    #[argh(positional)]
    path: PathBuf,

    /// test=command mapping
    #[argh(option, short = 't', from_str_fn(parse_mapping))]
    tests: Vec<(String, String)>,
}

fn parse_mapping(s: &str) -> Result<(String, String), String> {
    let parts: Vec<&str> = s.split('=').collect();
    if parts.len() == 2 {
        Ok((parts[0].to_string(), parts[1].to_string()))
    } else {
        Err("Mapping should be in the format key=value".to_string())
    }
}

#[derive(Debug, Serialize, Deserialize, FromArgs, PartialEq)]
/// Lint an existing config file to ensure it is valid
#[argh(subcommand, name = "lint")]
struct LintOptions {
    /// config to lint
    #[argh(positional)]
    config: PathBuf,
}

fn main() {
    let args: Args = argh::from_env();

    match args.subcommand {
        SubCommands::New(new_opts) => {
            let config = Config {
                scripts: new_opts
                    .tests
                    .iter()
                    .map(|(test_name, test_command)| Script {
                        name: test_name.to_string(),
                        command: test_command.to_string(),
                        results: ThompsonInfo {
                            interesting: 0,
                            uninteresting: 0,
                        },
                        runcount: 0,
                        avgruntime_ms: None,
                        bias: NotNan::new(1.0).unwrap(),
                        limit: None,
                    })
                    .collect(),
            };

            save_config(&config, &new_opts.path);
        }
        SubCommands::Run(run_opts) => {
            let mut config = parse_config(&run_opts.config);

            for _ in 0..run_opts.steps {
                step(&mut config, run_opts.ignore_runtime);
            }

            save_config(&config, &run_opts.output);

            let config = parse_config(&run_opts.output);

            plot_top_3(&config.scripts);
            if !run_opts.ignore_runtime {
                plot_top_3_inverses(&config.scripts);
            }
        }
        SubCommands::Rank(rank_opts) => {
            let config = parse_config(&rank_opts.config);

            if rank_opts.ignore_runtime {
                print_ranking(&config.scripts, rank_opts.verbose);
            } else {
                if rank_opts.verbose {
                    plot_top_3_inverses(&config.scripts);
                }

                let runtime = config
                    .scripts
                    .iter()
                    .map(|x| &x.avgruntime_ms)
                    .collect::<Vec<_>>();
                let runtimes: &[&Option<NotNan<f64>>] = runtime.as_slice();

                let user_biases = config.scripts.iter().map(|x| &x.bias).collect::<Vec<_>>();
                let user_biases: &[&NotNan<f64>] = user_biases.as_slice();

                print_ranking_bias_runtime(
                    &config.scripts,
                    runtimes,
                    user_biases,
                    rank_opts.verbose,
                );
            }
        }
        SubCommands::Reset(reset_opts) => {
            let mut config = parse_config(&reset_opts.config);

            if config.scripts.is_empty() {
                println!("No scripts to reset. Exiting...");
                return;
            }

            reset_state(&mut config, reset_opts.script);

            save_config(&config, &reset_opts.output);
        }
        SubCommands::Summarize(summarize_opts) => {
            let config = parse_config(&summarize_opts.config);

            plot_top_3(&config.scripts);

            if summarize_opts.ignore_runtime {
                print_ranking(&config.scripts, true);
            } else {
                plot_top_3_inverses(&config.scripts);

                let runtime = config
                    .scripts
                    .iter()
                    .map(|x| &x.avgruntime_ms)
                    .collect::<Vec<_>>();
                let runtimes: &[&Option<NotNan<f64>>] = runtime.as_slice();

                let user_biases = config.scripts.iter().map(|x| &x.bias).collect::<Vec<_>>();
                let user_biases: &[&NotNan<f64>] = user_biases.as_slice();

                print_ranking_bias_runtime(&config.scripts, runtimes, user_biases, true);
            }
        }
        SubCommands::Lint(lint_opts) => {
            let config = parse_config(&lint_opts.config);
            let mut seen_zero = false;
            for script in config.scripts {
                if script.bias == 0. {
                    println!("{} Warning: A bias of 0 will only run after all other scripts reach their limit.", script.name);
                    if seen_zero {
                        println!("{} ERROR: Multiple scripts with bias zero will not be ranked relative to each other. YOU PROBABLY DON'T WANT THIS.", script.name);
                        println!("{:1$}They will always be randomly run with equal probability regardless of interestingness/runtime.", "", script.name.len() + 8);
                    }
                    seen_zero = true;
                }

                if script.bias < NotNan::new(0.).unwrap() {
                    println!("{} ERROR: A negative bias rewards tests that take more time to find an interesting case.", script.name);
                }

                if script.limit == Some(0) {
                    println!("{} Warning: Limit of 0. This will stop this script from ever running. Leave undefined to have no limit.", script.name)
                }
            }
        }
    }
}
