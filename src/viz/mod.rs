pub mod html;
pub mod json;
pub mod png;
pub mod terminal;

use serde::{Deserialize, Serialize};

/// Output format requested by the user.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Terminal,
    Png,
    Json,
    Html,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "terminal" => Ok(OutputFormat::Terminal),
            "png" => Ok(OutputFormat::Png),
            "json" => Ok(OutputFormat::Json),
            "html" => Ok(OutputFormat::Html),
            other => Err(format!(
                "Unknown output format: '{other}'. Use terminal|png|json|html"
            )),
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Terminal => write!(f, "terminal"),
            OutputFormat::Png => write!(f, "png"),
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::Html => write!(f, "html"),
        }
    }
}
