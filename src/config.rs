use std::{
    fs::File,
    io::{Read, Write},
    path::PathBuf,
};

use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

use crate::thompson::ThompsonInfo;

pub fn parse_config(config_path: &PathBuf) -> Config {
    let mut file = File::open(config_path).unwrap();
    let mut data = String::new();
    file.read_to_string(&mut data).unwrap();

    let config: Config = serde_json::from_str(&data).unwrap();
    config
}

pub fn save_config(config: &Config, path: &PathBuf) {
    let data = serde_json::to_string_pretty(config).unwrap();

    let prefix = path.parent().unwrap();
    std::fs::create_dir_all(prefix).unwrap();
    let mut file = File::create(path).unwrap();

    file.write_all(data.as_bytes()).unwrap();
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Config {
    pub scripts: Vec<Script>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Script {
    pub name: String,
    pub command: String,
    pub results: ThompsonInfo,
    pub runcount: u64,
    pub avgruntime_ms: Option<NotNan<f64>>,
    pub bias: NotNan<f64>,
    pub limit: Option<u64>,
}
