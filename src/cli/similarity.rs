use crate::analysis::{cls_cosine_similarity, knn_overlap, linear_cka};
use crate::errors::Error;
use crate::extract::ExtractedFeatures;
use crate::models::ModelSession;
use clap::Args;
use ndarray::Array2;
use std::path::PathBuf;
use tracing::info;

#[derive(Args, Debug)]
pub struct SimilarityArgs {
    /// First model.
    #[arg(long)]
    pub model_a: String,

    /// Second model.
    #[arg(long)]
    pub model_b: String,

    /// Dataset directory.
    #[arg(short, long)]
    pub dataset: PathBuf,

    /// Similarity metric to use.
    #[arg(short, long, default_value = "cka", value_parser = ["cka", "knn", "cosine", "all"])]
    pub metric: String,
}

pub fn run(args: SimilarityArgs) -> Result<(), Error> {
    info!(
        "Measuring similarity {} vs {} on {:?}",
        args.model_a, args.model_b, args.dataset
    );

    let mut session_a = ModelSession::load(&args.model_a)?;
    let mut session_b = ModelSession::load(&args.model_b)?;

    let dataset = crate::dataset::DatasetIterator::new(&args.dataset, true)?;
    let total = dataset.len();
    info!("Dataset: {total} images");

    let mut cls_a: Vec<ndarray::Array1<f32>> = Vec::new();
    let mut cls_b: Vec<ndarray::Array1<f32>> = Vec::new();

    let mut patch_rows_a: Vec<ndarray::Array1<f32>> = Vec::new();
    let mut patch_rows_b: Vec<ndarray::Array1<f32>> = Vec::new();

    for result in dataset {
        let (_, img) = result?;

        let out_a = session_a.infer(&img)?;
        let out_b = session_b.infer(&img)?;

        let feat_a = ExtractedFeatures::from_output(out_a)?;
        let feat_b = ExtractedFeatures::from_output(out_b)?;

        let mean_a = feat_a.mean_patch();
        let mean_b = feat_b.mean_patch();

        if let (Some(ca), Some(cb)) = (feat_a.cls_token, feat_b.cls_token) {
            cls_a.push(ca);
            cls_b.push(cb);
        }

        patch_rows_a.push(mean_a);
        patch_rows_b.push(mean_b);
    }

    if patch_rows_a.is_empty() {
        println!("No valid images processed.");
        return Ok(());
    }

    let n = patch_rows_a.len();
    let da = patch_rows_a[0].len();
    let db = patch_rows_b[0].len();

    let mut mat_a = Array2::<f32>::zeros((n, da));
    let mut mat_b = Array2::<f32>::zeros((n, db));
    for i in 0..n {
        mat_a.row_mut(i).assign(&patch_rows_a[i]);
        mat_b.row_mut(i).assign(&patch_rows_b[i]);
    }

    println!(
        "\nRepresentation similarity: {} vs {}",
        args.model_a, args.model_b
    );
    println!("Dataset: {} images", n);
    println!("{}", "═".repeat(55));

    match args.metric.as_str() {
        "cka" | "all" => {
            let cka = linear_cka(&mat_a, &mat_b)?;
            println!("  Linear CKA:          {:.4}", cka);
        }
        _ => {}
    }

    match args.metric.as_str() {
        "knn" | "all" => {
            let overlap = knn_overlap(&mat_a, &mat_b, 10)?;
            println!("  k-NN overlap (k=10): {:.4}", overlap);
        }
        _ => {}
    }

    match args.metric.as_str() {
        "cosine" | "all" => {
            if !cls_a.is_empty() {
                let same_width = cls_a.iter().zip(&cls_b).all(|(a, b)| a.len() == b.len());

                if same_width {
                    let mut total_sim = 0.0f32;
                    for i in 0..cls_a.len() {
                        total_sim += cls_cosine_similarity(&cls_a[i], &cls_b[i]);
                    }
                    let mean_sim = total_sim / cls_a.len() as f32;
                    println!("  Mean CLS cosine sim: {:.4}", mean_sim);
                } else {
                    println!(
                        "  Mean CLS cosine sim: N/A (embedding dims differ: {} vs {})",
                        cls_a[0].len(),
                        cls_b[0].len()
                    );
                }
            }
        }
        _ => {}
    }

    Ok(())
}
