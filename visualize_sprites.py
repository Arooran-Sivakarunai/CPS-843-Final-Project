import os
import matplotlib.pyplot as plt

from predict_pokemon import (
    load_encoder,
    build_database,
    predict,
)


def plot_closest_match_distances(
    shiny_folder="shinies",
    db_folder="sprites",
    encoder_path="encoder.pth",
    metric="l2",
):
    print("Loading encoder...")
    encoder, device = load_encoder(encoder_path)

    print("Building embedding database...")
    database = build_database(db_folder, encoder, device)

    labels = []
    distances = []

    shiny_files = sorted(
        f for f in os.listdir(shiny_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    print(f"Running predictions on {len(shiny_files)} shinies...\n")

    for fname in shiny_files:
        true_label = os.path.splitext(fname)[0]
        img_path = os.path.join(shiny_folder, fname)

        pred_label, score = predict(
            img_path,
            database,
            encoder,
            device,
            metric=metric,
        )

        labels.append(true_label)
        distances.append(score)

    print(f"Average Distance: {sum(distances) / len(distances):2f}")

    # =============================
    #   Bar graph
    # =============================
    plt.figure(figsize=(max(10, len(labels) * 0.4), 5))
    plt.bar(labels, distances)
    plt.xticks(rotation=90)
    plt.ylabel("Closest-match distance")
    plt.xlabel("Shiny Pokémon")
    plt.title("Distance to Closest Match for Shiny Pokémon")
    plt.tight_layout()
    plt.show()


# =============================
#   Main
# =============================
if __name__ == "__main__":
    plot_closest_match_distances(
        shiny_folder="sprites",
        db_folder="sprites",
        encoder_path="encoder.pth",
        metric="l2",
    )

    plot_closest_match_distances(
        shiny_folder="shinies",
        db_folder="sprites",
        encoder_path="encoder.pth",
        metric="l2",
    )
