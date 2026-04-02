use crate::errors::AnalysisError;
use ndarray::{Array1, Array2, Array4};

pub(crate) fn ensure_finite_1d(data: &Array1<f32>, context: &str) -> Result<(), AnalysisError> {
    if let Some((index, value)) = data.indexed_iter().find(|(_, value)| !value.is_finite()) {
        return Err(AnalysisError::NonFiniteValues {
            context: format!("{context} contains non-finite value {value} at [{index}]"),
        });
    }

    Ok(())
}

pub(crate) fn ensure_finite_2d(data: &Array2<f32>, context: &str) -> Result<(), AnalysisError> {
    if let Some(((row, col), value)) = data.indexed_iter().find(|(_, value)| !value.is_finite()) {
        return Err(AnalysisError::NonFiniteValues {
            context: format!("{context} contains non-finite value {value} at [{row}, {col}]"),
        });
    }

    Ok(())
}

pub(crate) fn ensure_finite_4d(data: &Array4<f32>, context: &str) -> Result<(), AnalysisError> {
    if let Some(((layer, head, row, col), value)) =
        data.indexed_iter().find(|(_, value)| !value.is_finite())
    {
        return Err(AnalysisError::NonFiniteValues {
            context: format!(
                "{context} contains non-finite value {value} at [{layer}, {head}, {row}, {col}]"
            ),
        });
    }

    Ok(())
}

/// Compute the integer side length of a square patch grid.
///
/// Returns an error if `patch_count` is not a perfect square.
pub fn square_grid_side(patch_count: usize, context: &str) -> Result<usize, AnalysisError> {
    let grid = (patch_count as f64).sqrt().round() as usize;
    if grid > 0 && grid.checked_mul(grid) == Some(patch_count) {
        Ok(grid)
    } else {
        Err(AnalysisError::InvalidPatchGrid {
            context: context.to_string(),
            patch_count,
        })
    }
}
