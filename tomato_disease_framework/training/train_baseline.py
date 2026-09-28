if __name__ == "__main__":
    import argparse
    import yaml

    from preprocessing.dataset import load_phase2_splits, TomatoLeafDataset
    from preprocessing.transforms import build_transforms
    from models.efficientnet_baseline import build_efficientnet_b0

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_reproducibility(config["runtime"]["seed"])

    device = resolve_device(config)
    print("Device:", device)

    split_dir = config["data"]["splits_dir"]
    splits = load_phase2_splits(split_dir)

    classes = sorted({row["class_name"] for row in splits["train"]})
    class_to_idx = {name: i for i, name in enumerate(classes)}

    dataset_root = os.environ["TDF_DATASET_ROOT"]

    transforms = build_transforms(config["model"]["image_size"])

    train_dataset = TomatoLeafDataset(
        splits["train"],
        dataset_root,
        class_to_idx,
        transforms["train"],
    )

    validation_dataset = TomatoLeafDataset(
        splits["validation"],
        dataset_root,
        class_to_idx,
        transforms["validation"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["model"]["batch_size"],
        shuffle=True,
        num_workers=config["model"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["model"]["batch_size"],
        shuffle=False,
        num_workers=config["model"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = build_efficientnet_b0(
        pretrained=config["model"]["pretrained"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"],
    )

    epochs = config["model"]["epochs"]

    output_dir = Path(config["paths"]["results_dir"]) / "phase3" / "efficientnet_b0_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_metrics = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device,
            classes,
        )

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Validation Loss: {validation_metrics['loss']:.6f}")
        print(f"Validation Accuracy: {validation_metrics['accuracy']:.6f}")
        print(f"Validation Macro F1: {validation_metrics['macro_f1']:.6f}")

        if validation_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = validation_metrics["macro_f1"]

            save_checkpoint(
                output_dir / "best_checkpoint.pt",
                model,
                optimizer,
                epoch,
                config,
            )

            print("Best checkpoint saved.")

    save_experiment_config(output_dir, config)

    print("\nTraining completed.")
    print("Best validation Macro F1:", best_val_f1)
    print("Output directory:", output_dir)