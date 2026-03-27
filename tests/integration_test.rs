use latent_inspector::analysis::{compute_metrics, knn_overlap, linear_cka, patch_entropy};
use latent_inspector::extract::ExtractedFeatures;
use latent_inspector::models::registry::{find, ModelInfo, SSLMethod};
use latent_inspector::models::{ModelOutput, ModelSession, OutputTensorMetadata};
use latent_inspector::validation::{fixtures::load_fixture_set, validate_model};
use ndarray::{Array1, Array2};

fn make_output(model_name: &str, n_patches: usize, embed_dim: usize) -> ModelOutput {
    ModelOutput {
        cls_token: Some(Array1::from_elem(embed_dim, 0.5_f32)),
        patch_tokens: Array2::from_shape_fn((n_patches, embed_dim), |(i, j)| {
            ((i + j) as f32) / (n_patches + embed_dim) as f32
        }),
        attention_weights: None,
        model_info: ModelInfo {
            name: model_name.to_string(),
            architecture: "ViT-L/14".to_string(),
            patch_size: 14,
            embed_dim: embed_dim as u32,
            num_layers: 24,
            num_heads: 16,
            method: SSLMethod::DINO,
            input_size: 224,
            params_m: 304,
        },
        tensor_metadata: OutputTensorMetadata {
            input_name: "pixel_values".into(),
            input_shape: vec![1, 3, 224, 224],
            output_name: "last_hidden_state".into(),
            output_shape: vec![1, n_patches + 1, embed_dim],
            sequence_has_cls: true,
            observed_patch_count: n_patches,
            embedding_dim: embed_dim,
        },
    }
}

#[test]
fn test_full_analysis_pipeline() {
    let output = make_output("dinov2-vit-l14", 256, 64);
    let features = ExtractedFeatures::from_output(output).unwrap();
    let metrics = compute_metrics(&features, "dinov2-vit-l14").unwrap();

    assert_eq!(metrics.model_name, "dinov2-vit-l14");
    assert!(metrics.effective_rank >= 1);
    assert!(metrics.effective_rank <= 64);
    assert!(metrics.patch_entropy >= 0.0);
    assert!(metrics.top10_variance_pct >= 0.0 && metrics.top10_variance_pct <= 100.0 + 1e-3);
}

#[test]
fn test_cka_self_similarity() {
    let data = Array2::from_shape_fn((32, 16), |(i, j)| (i + j) as f32);
    let cka = linear_cka(&data, &data).unwrap();
    approx::assert_relative_eq!(cka, 1.0, epsilon = 1e-4);
}

#[test]
fn test_knn_overlap_self() {
    let data = Array2::from_shape_fn((20, 8), |(i, j)| (i * 3 + j) as f32);
    let overlap = knn_overlap(&data, &data, 5).unwrap();
    approx::assert_relative_eq!(overlap, 1.0, epsilon = 1e-4);
}

#[test]
fn test_patch_entropy_positive() {
    let data = Array2::from_shape_fn((64, 32), |(i, j)| ((i * 7 + j * 3) % 11) as f32);
    let e = patch_entropy(&data, 8, 30).unwrap();
    assert!(e >= 0.0);
}

#[test]
fn test_registry_has_all_models() {
    let names = latent_inspector::models::registry::model_names();
    assert!(names.contains(&"dinov2-vit-l14".to_string()));
    assert!(names.contains(&"clip-vit-l14".to_string()));
    assert!(names.contains(&"mae-vit-l16".to_string()));
}

#[test]
fn test_preprocess_shape() {
    use latent_inspector::models::preprocess::{preprocess, PreprocessConfig};
    let img = image::DynamicImage::new_rgb8(400, 300);
    let cfg = PreprocessConfig::new(224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    let tensor = preprocess(&img, &cfg).unwrap();
    assert_eq!(tensor.shape(), &[1, 3, 224, 224]);
}

#[test]
fn test_validation_fixture_manifest_available() {
    let fixture_set = load_fixture_set(None).unwrap();
    assert_eq!(fixture_set.manifest.fixture_set, "standard");
    assert!(fixture_set.manifest.models.contains_key("dinov2-vit-l14"));
}

#[test]
fn test_validate_model_returns_structured_summary() {
    let summary = validate_model("dinov2-vit-l14", None, false).unwrap();
    assert_eq!(summary.model, "dinov2-vit-l14");
    assert_eq!(
        summary.status,
        latent_inspector::validation::ValidationStatus::Validated
    );
    assert!(summary.caveats.is_empty());
}

#[test]
fn test_loader_stub_tracks_contract_semantics() {
    let entry = find("mae-vit-l16").unwrap();
    let session = ModelSession::load(&entry.info.name).unwrap();
    let output = session
        .infer(&image::DynamicImage::new_rgb8(224, 224))
        .unwrap();

    assert!(!output.tensor_metadata.sequence_has_cls);
    assert_eq!(output.tensor_metadata.output_shape, vec![1, 196, 1024]);
}
