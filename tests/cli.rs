use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::process::Command; // Run programs

#[test]
fn command_doesnt_exist() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("bandits")?;

    cmd.arg("notcommand");
    cmd.assert().failure().stderr(predicate::str::contains(
        "Unrecognized argument: notcommand",
    ));

    Ok(())
}

#[test]
fn simple_2_scripts() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("bandits")?;

    cmd.arg("run")
        .arg("./config-tests/simple-2-scripts.json")
        .arg("--steps")
        .arg("2")
        .arg("--output")
        .arg("./tests/temp/temp.json");

    cmd.assert().success();

    let mut cmd = Command::cargo_bin("bandits")?;

    cmd.arg("summarize").arg("./tests/temp/temp.json");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("50th percentile: 0.2929"))
        .stdout(predicate::str::contains("50th percentile: 0.7071"));

    Ok(())
}

#[test]
fn simple_2_scripts_no_runs() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("bandits")?;

    cmd.arg("run")
        .arg("./config-tests/simple-2-scripts.json")
        .arg("--steps")
        .arg("0")
        .arg("--output")
        .arg("./tests/temp/no_runs.json");

    cmd.assert().success();

    let mut cmd = Command::cargo_bin("bandits")?;

    cmd.arg("summarize").arg("./tests/temp/no_runs.json");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("NaN%"))
        .stdout(predicate::str::contains("Runs: 0"));

    Ok(())
}

#[test]
fn lots_of_scripts() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("bandits")?;

    cmd.arg("run")
        .arg("./config-tests/lots-of-scripts.json")
        .arg("--steps")
        .arg("10")
        .arg("--output")
        .arg("./tests/temp/lots-of-scripts.json");

    cmd.assert().success();

    Ok(())
}

#[test]
fn prefer_02_ffast() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("bandits")?;

    cmd.arg("run")
        .arg("./config-tests/prefer-02-ffast.json")
        .arg("--steps")
        .arg("200")
        .arg("--output")
        .arg("./tests/temp/02-ffast.json");

    cmd.assert().success();

    let mut cmd = Command::cargo_bin("bandits")?;

    cmd.arg("summarize").arg("./tests/temp/02-ffast.json");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("1: test 0.2 ffast"));

    Ok(())
}

//#[ignore]
#[test]
fn prefer_09_slow() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("bandits")?;

    cmd.arg("run")
        .arg("./config-tests/prefer-09-slow.json")
        .arg("--steps")
        .arg("200")
        .arg("--output")
        .arg("./tests/temp/09-slow.json");

    cmd.assert().success();

    let mut cmd = Command::cargo_bin("bandits")?;

    cmd.arg("summarize").arg("./tests/temp/09-slow.json");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("1: test 0.9 slow"));

    Ok(())
}

#[test]
fn lint_negative_bias() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("bandits")?;

    cmd.arg("lint").arg("./tests/lint/bias-negative.json");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Test 1 ERROR: A negative bias rewards tests that take more time to find an interesting case."));

    Ok(())
}

#[test]
fn lint_zero_multiple_bias() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("bandits")?;

    cmd.arg("lint").arg("./tests/lint/bias-zero-multiple.json");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Test 1 Warning: A bias of 0 will only run after all other scripts reach their limit."))
        .stdout(predicate::str::contains("Test 2 Warning: A bias of 0 will only run after all other scripts reach their limit."))
        .stdout(predicate::str::contains("Test 2 ERROR: Multiple scripts with bias zero will not be ranked relative to each other. YOU PROBABLY DON\'T WANT THIS."))
        .stdout(predicate::str::contains("They will always be randomly run with equal probability regardless of interestingness/runtime."));

    Ok(())
}

#[test]
fn lint_zero_bias() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("bandits")?;

    cmd.arg("lint").arg("./tests/lint/bias-zero.json");

    cmd.assert().success().stdout(predicate::str::contains(
        "Test 1 Warning: A bias of 0 will only run after all other scripts reach their limit.",
    ));

    Ok(())
}

#[test]
fn lint_zero_limit() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("bandits")?;

    cmd.arg("lint").arg("./tests/lint/limit-zero.json");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Test 1 Warning: Limit of 0. This will stop this script from ever running. Leave undefined to have no limit."));

    Ok(())
}
