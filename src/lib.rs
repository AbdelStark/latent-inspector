pub mod analysis;
pub mod dataset;
pub mod errors;
pub mod extract;
pub mod models;
pub mod tui;
pub mod validation;
pub mod viz;

#[cfg(test)]
pub(crate) static TEST_PROCESS_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
