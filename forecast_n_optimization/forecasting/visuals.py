import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_seaborn_comparison(y_true, y_pred, title):
    """
    Elegant Seaborn time series plot comparing actual and predicted values.
    Uses the 'husl' palette.
    Shows the last 168 observations for clarity.
    """
    # Ensure predictions are aligned with y_true's index
    y_pred_series = pd.Series(y_pred, index=y_true.index)
    df_plot = pd.DataFrame(
        {
            "Time": y_true.tail(168).index,
            "Actual": y_true.tail(168).values,
            "Predicted": y_pred_series.tail(168).values,
        }
    ).melt(
        id_vars=["Time"],
        value_vars=["Actual", "Predicted"],
        var_name="Type",
        value_name="Price",
    )

    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=df_plot,
        x="Time",
        y="Price",
        hue="Type",
        linewidth=2.5,
        palette="husl",
    )
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Price (EUR/MWh)", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="", fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_seaborn_scatter(y_true, y_pred, title):
    """
    Elegant Seaborn scatter with regression, using husl colors.
    Plots actual vs. predicted and a regression fit; overlays the y=x "perfect fit" line.
    """
    plt.figure(figsize=(8, 8))
    husl_colors = sns.color_palette("husl", 8)
    scatter_color = husl_colors[0]
    regression_color = husl_colors[0]

    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, s=50, color=scatter_color)
    sns.regplot(
        x=y_true,
        y=y_pred,
        scatter=False,
        color=regression_color,
        line_kws={"linewidth": 2},
    )

    min_val, max_val = y_true.min(), y_true.max()
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "--k",
        alpha=0.7,
        linewidth=2,
        label="Perfect Prediction",
    )

    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Actual Price (EUR/MWh)", fontsize=12)
    plt.ylabel("Predicted Price (EUR/MWh)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()


def feature_importance(model, X, title, top_n=15):
    """
    Single-color feature importance plot with husl palette.
    Displays the top_n features as a horizontal bar chart.
    """
    husl_color = sns.color_palette("husl", 1)[0]  # Single husl color
    # Prepare importance data
    importance_df = (
        pd.DataFrame(
            {"Feature": X.columns, "Importance": model.feature_importances_}
        )
        .nlargest(top_n, "Importance")
        .reset_index(drop=True)
    )

    plt.figure(figsize=(10, 7))
    ax = sns.barplot(
        data=importance_df,
        y="Feature",
        x="Importance",
        color=husl_color,
        orient="h",
    )
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Feature Importance", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    for i, v in enumerate(importance_df["Importance"]):
        ax.text(
            v + max(importance_df["Importance"]) * 0.01,
            i,
            f"{v:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="semibold",
            color="black",
        )
    sns.despine()
    plt.tight_layout()
    plt.show()
