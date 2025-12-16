import os
from predict_pokemon import (
    load_encoder,
    build_database,
    predict,
)

# =============================
#   Verification script
# =============================
def verify_shiny_predictions(
    shiny_folder="shinies",
    db_folder="sprites",
    encoder_path="encoder.pth",
    metric="l2",
):
    print("Loading encoder...")
    encoder, device = load_encoder(encoder_path)

    print("Building embedding database...")
    database = build_database(db_folder, encoder, device)
    print(f"Loaded {len(database)} reference sprites.\n")

    correct = 0
    total = 0
    mismatches = []

    for fname in os.listdir(shiny_folder):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        true_label = os.path.splitext(fname)[0]
        img_path = os.path.join(shiny_folder, fname)

        pred_label, score = predict(
            img_path,
            database,
            encoder,
            device,
            metric=metric,
        )

        total += 1

        if pred_label == true_label:
            correct += 1
            status = "✅"
        else:
            status = "❌"
            mismatches.append((true_label, pred_label, score))

        print(
            f"{status} {true_label:15s} → predicted: {pred_label:15s} "
            f"(distance={score:.4f})"
        )

    print("\n===============================")
    print(f" Accuracy: {correct}/{total} ({100 * correct / max(total, 1):.2f}%)")
    print("===============================")

    if mismatches:
        print("\nMismatches:")
        for true_label, pred_label, score in mismatches:
            print(
                f"  {true_label} → {pred_label} (distance={score:.4f})"
            )


# =============================
#   Main
# =============================
if __name__ == "__main__":
    verify_shiny_predictions(
        shiny_folder="sprites",
        db_folder="sprites",
        encoder_path="encoder.pth",
        metric="l2",
    )
