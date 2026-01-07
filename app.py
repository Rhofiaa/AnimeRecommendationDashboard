import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="Anime Recommender", page_icon="🎯", layout="wide")

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_USERS = 5000
MIN_INTERACTIONS = 5
MEAN_CENTER = True
EVAL_SAMPLE_SIZE = 600


def _locate_default_file(filename: str) -> Optional[Path]:
    candidate = Path(__file__).resolve().parent / filename
    if candidate.exists():
        return candidate
    downloads_candidate = Path.home() / "Downloads" / filename
    if downloads_candidate.exists():
        return downloads_candidate
    return None


@st.cache_data(show_spinner=False)
def load_datasets(
    anime_bytes: Optional[bytes],
    rating_bytes: Optional[bytes],
    anime_path: Optional[str],
    rating_path: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if anime_bytes is not None:
        anime_df = pd.read_csv(io.BytesIO(anime_bytes))
    elif anime_path:
        anime_df = pd.read_csv(anime_path)
    else:
        raise FileNotFoundError("Anime metadata CSV tidak ditemukan.")

    if rating_bytes is not None:
        rating_df = pd.read_csv(io.BytesIO(rating_bytes))
    elif rating_path:
        rating_df = pd.read_csv(rating_path)
    else:
        raise FileNotFoundError("Rating CSV tidak ditemukan.")

    return anime_df, rating_df


@st.cache_data(show_spinner=False)
def prepare_cf_frame(anime_df: pd.DataFrame, rating_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = rating_df.merge(anime_df, on="anime_id", how="left", suffixes=("_user", "_anime"))
    rating_col = "rating_user" if "rating_user" in merged.columns else "rating"
    merged[rating_col] = pd.to_numeric(merged[rating_col], errors="coerce")

    cf_ratings = (
        merged[["user_id", "anime_id", rating_col]]
        .rename(columns={rating_col: "rating"})
        .dropna(subset=["rating"])
    )
    cf_ratings = cf_ratings[cf_ratings["rating"] != -1]

    meta_cols = [
        col
        for col in ["anime_id", "name", "genre", "type", "episodes", "rating_anime"]
        if col in merged.columns
    ]
    anime_meta = merged[meta_cols].drop_duplicates("anime_id") if meta_cols else anime_df

    return cf_ratings, anime_meta


@st.cache_data(show_spinner=False)
def split_train_test(cf_ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        cf_ratings,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    test_df = test_df[test_df["user_id"].isin(train_df["user_id"])]
    test_df = test_df[test_df["anime_id"].isin(train_df["anime_id"])]
    return train_df, test_df


@st.cache_resource(show_spinner=False)
def build_user_similarity(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = train_df["user_id"].value_counts()
    active_users = counts[counts >= MIN_INTERACTIONS].index
    filtered = train_df[train_df["user_id"].isin(active_users)]

    if filtered["user_id"].nunique() > MAX_USERS:
        top_users = counts.head(MAX_USERS).index
        filtered = filtered[filtered["user_id"].isin(top_users)]

    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    user_item = filtered.pivot_table(
        index="user_id",
        columns="anime_id",
        values="rating",
        aggfunc="mean",
    )

    centered = user_item.sub(user_item.mean(axis=1), axis=0) if MEAN_CENTER else user_item.copy()
    filled = centered.fillna(0)
    similarity = cosine_similarity(filled)
    similarity_df = pd.DataFrame(similarity, index=user_item.index, columns=user_item.index)

    return user_item, centered, similarity_df


def predict_rating(
    user_id: int,
    anime_id: int,
    top_n_neighbors: int,
    user_item: pd.DataFrame,
    centered: pd.DataFrame,
    similarity: pd.DataFrame,
    global_mean: float,
) -> float:
    if user_id not in user_item.index or anime_id not in user_item.columns:
        return np.nan

    sim_series = similarity.loc[user_id].drop(user_id)
    neighbors = sim_series.nlargest(top_n_neighbors)
    neighbors = neighbors[neighbors > 0]
    if neighbors.empty:
        return np.nan

    neighbor_centered = centered.loc[neighbors.index, anime_id].fillna(0)
    weights = neighbors.values
    denom = np.abs(weights).sum()
    if denom == 0:
        return np.nan

    user_mean = user_item.loc[user_id].mean(skipna=True)
    if np.isnan(user_mean):
        user_mean = global_mean

    contribution = float(np.dot(neighbor_centered.values, weights) / denom)
    return float(np.clip(user_mean + contribution, 1, 10))


def recommend_for_user(
    user_id: int,
    top_n_neighbors: int,
    top_n_recs: int,
    user_item: pd.DataFrame,
    centered: pd.DataFrame,
    similarity: pd.DataFrame,
    global_mean: float,
) -> pd.Series:
    if user_id not in user_item.index:
        return pd.Series(dtype=float)

    sim_series = similarity.loc[user_id].drop(user_id)
    neighbors = sim_series.nlargest(top_n_neighbors)
    neighbors = neighbors[neighbors > 0]
    if neighbors.empty:
        return pd.Series(dtype=float)

    neighbor_matrix = centered.loc[neighbors.index].fillna(0)
    weighted = neighbor_matrix.mul(neighbors, axis=0)
    denom = np.abs(neighbors).sum()
    if denom == 0:
        return pd.Series(dtype=float)

    weighted_sum = weighted.sum(axis=0) / denom
    user_mean = user_item.loc[user_id].mean(skipna=True)
    if np.isnan(user_mean):
        user_mean = global_mean

    preds = user_mean + weighted_sum
    preds = preds.clip(lower=1, upper=10)
    already_rated = user_item.loc[user_id].dropna().index
    preds = preds.drop(already_rated, errors="ignore").sort_values(ascending=False).head(top_n_recs)
    return preds


@st.cache_data(show_spinner=False)
def evaluate_model(
    eval_df: pd.DataFrame,
    top_n_neighbors: int,
    user_item: pd.DataFrame,
    centered: pd.DataFrame,
    similarity: pd.DataFrame,
    global_mean: float,
    seed: int,
) -> dict[str, float]:
    if eval_df.empty:
        return {"rmse": np.nan, "mae": np.nan, "n_samples": 0}

    sample = eval_df.sample(
        n=min(EVAL_SAMPLE_SIZE, len(eval_df)),
        random_state=seed,
    )
    preds = sample.apply(
        lambda row: predict_rating(
            int(row["user_id"]),
            int(row["anime_id"]),
            top_n_neighbors,
            user_item,
            centered,
            similarity,
            global_mean,
        ),
        axis=1,
    )
    valid = sample[~preds.isna()].copy()
    if valid.empty:
        return {"rmse": np.nan, "mae": np.nan, "n_samples": 0}

    preds = preds.loc[valid.index]
    mse = mean_squared_error(valid["rating"], preds)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(valid["rating"], preds)),
        "n_samples": int(len(valid)),
    }


st.title("🎯 Anime Recommendation Dashboard")
st.caption("Analisis singkat dan rekomendasi anime berbasis collaborative filtering.")

with st.sidebar:
    st.header("Pengaturan")
    data_option = st.radio(
        "Sumber data",
        ("Gunakan dataset bawaan", "Upload manual"),
        index=0,
    )

    anime_bytes: Optional[bytes] = None
    rating_bytes: Optional[bytes] = None
    anime_path: Optional[Path] = None
    rating_path: Optional[Path] = None

    if data_option == "Gunakan dataset bawaan":
        anime_path = _locate_default_file("anime.csv")
        rating_path = _locate_default_file("rating.csv")
        if not anime_path or not rating_path:
            st.warning("File default belum ditemukan. Upload manual di bawah.")
            data_option = "Upload manual"

    if data_option == "Upload manual":
        anime_file = st.file_uploader("anime.csv", type="csv", key="anime")
        rating_file = st.file_uploader("rating.csv", type="csv", key="rating")
        if anime_file is not None:
            anime_bytes = anime_file.getvalue()
        if rating_file is not None:
            rating_bytes = rating_file.getvalue()

    st.divider()
    st.subheader("Model")
    top_neighbors = st.slider("Jumlah pengguna serupa", 5, 40, 20, step=1)
    top_recs = st.slider("Jumlah rekomendasi", 5, 15, 10, step=1)

anime_path_str = str(anime_path) if anime_path else None
rating_path_str = str(rating_path) if rating_path else None

try:
    anime_df, rating_df = load_datasets(anime_bytes, rating_bytes, anime_path_str, rating_path_str)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.exception(exc)
    st.stop()

cf_ratings, anime_meta = prepare_cf_frame(anime_df, rating_df)
if cf_ratings.empty:
    st.error("Dataset tidak memiliki rating valid setelah pembersihan.")
    st.stop()

train_df, test_df = split_train_test(cf_ratings)
if train_df["user_id"].nunique() < 2:
    st.error("Dataset train belum cukup variasi user.")
    st.stop()

user_item, centered, similarity = build_user_similarity(train_df)
if user_item.empty:
    st.error("Tidak ada user yang memenuhi filter default. Gunakan dataset lebih besar.")
    st.stop()

global_mean = cf_ratings["rating"].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total user", f"{cf_ratings['user_id'].nunique():,}")
col2.metric("Total anime", f"{cf_ratings['anime_id'].nunique():,}")
col3.metric("Interaksi", f"{len(cf_ratings):,}")

recommend_tab, eval_tab, explore_tab = st.tabs(["Rekomendasi", "Evaluasi", "Eksplorasi"])

with recommend_tab:
    st.subheader("Rekomendasi personal")
    user_options = user_item.index.to_series().sort_values()
    target_user = st.selectbox(
        "Pilih user",
        options=user_options,
        index=0,
        format_func=lambda uid: f"User {uid}",
    )

    user_history = train_df[train_df["user_id"] == target_user]
    profile_cols = st.columns(3)
    profile_cols[0].metric("Interaksi dimiliki", f"{len(user_history):,}")
    avg_rating = user_history["rating"].mean() if not user_history.empty else np.nan
    profile_cols[1].metric("Rata-rata rating", f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "-")
    favourite_genre = "-"
    if not user_history.empty and "genre" in anime_meta.columns:
        genre_series = user_history.merge(
            anime_meta[["anime_id", "genre"]], on="anime_id", how="left"
        )["genre"].dropna()
        expanded_genre = (
            genre_series.astype(str)
            .str.split(",")
            .explode()
            .str.strip()
            .replace("", np.nan)
            .dropna()
        )
        if not expanded_genre.empty:
            favourite_genre = ", ".join(expanded_genre.value_counts().head(2).index.tolist())
    profile_cols[2].metric("Genre favorit", favourite_genre)

    if not user_history.empty:
        top_history = (
            user_history.merge(anime_meta[["anime_id", "name"]], on="anime_id", how="left")
            .sort_values("rating", ascending=False)
            .head(5)
        )
        if not top_history.empty:
            st.markdown("**5 rating tertinggi user**")
            st.dataframe(
                top_history[["anime_id", "name", "rating"]],
                width="stretch",
                column_config={
                    "rating": st.column_config.NumberColumn("Rating", format="%.2f"),
                },
            )

    recommendations = recommend_for_user(
        target_user,
        top_neighbors,
        top_recs,
        user_item,
        centered,
        similarity,
        global_mean,
    )

    if recommendations.empty:
        st.info("Belum ada rekomendasi untuk user ini.")
    else:
        rec_df = (
            recommendations.rename("predicted_rating")
            .reset_index()
            .rename(columns={"index": "anime_id"})
            .merge(anime_meta, on="anime_id", how="left")
        )
        st.dataframe(
            rec_df,
            width="stretch",
            column_config={
                "predicted_rating": st.column_config.NumberColumn("Prediksi", format="%.2f"),
            },
        )
        st.download_button(
            label="Unduh rekomendasi (CSV)",
            data=rec_df.to_csv(index=False).encode("utf-8"),
            file_name=f"recommendations_user_{target_user}.csv",
            mime="text/csv",
        )

with eval_tab:
    st.subheader("Kualitas model")
    eval_df = test_df[
        test_df["user_id"].isin(user_item.index)
        & test_df["anime_id"].isin(user_item.columns)
    ]
    metrics = evaluate_model(
        eval_df,
        top_neighbors,
        user_item,
        centered,
        similarity,
        global_mean,
        seed=RANDOM_STATE,
    )

    if metrics["n_samples"] == 0:
        st.info("Belum ada pasangan user-item di test set yang bisa dihitung.")
    else:
        met_cols = st.columns(3)
        met_cols[0].metric("RMSE", f"{metrics['rmse']:.4f}")
        met_cols[1].metric("MAE", f"{metrics['mae']:.4f}")
        met_cols[2].metric("Sampel", f"{metrics['n_samples']:,}")
        st.caption("Nilai lebih kecil menandakan prediksi lebih akurat.")

        metrics_df = pd.DataFrame([metrics])
        st.download_button(
            label="Unduh ringkasan evaluasi (CSV)",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="cf_evaluation_summary.csv",
            mime="text/csv",
        )

        preview_df = eval_df.sample(
            n=min(200, len(eval_df)),
            random_state=RANDOM_STATE,
        ).copy()
        preview_df["pred_rating"] = preview_df.apply(
            lambda row: predict_rating(
                int(row["user_id"]),
                int(row["anime_id"]),
                top_neighbors,
                user_item,
                centered,
                similarity,
                global_mean,
            ),
            axis=1,
        )
        preview_df = preview_df.dropna(subset=["pred_rating"])
        if not preview_df.empty:
            st.markdown("**Contoh hasil prediksi pada data uji**")
            preview_display = preview_df.merge(
                anime_meta[["anime_id", "name"]],
                on="anime_id",
                how="left",
            )
            st.dataframe(
                preview_display.head(50),
                width="stretch",
                column_config={
                    "pred_rating": st.column_config.NumberColumn("Prediksi", format="%.2f"),
                },
            )
            st.download_button(
                label="Unduh sampel prediksi (CSV)",
                data=preview_display.to_csv(index=False).encode("utf-8"),
                file_name="cf_prediction_samples.csv",
                mime="text/csv",
            )

        with st.expander("Distribusi rating train vs test", expanded=False):
            train_counts = train_df["rating"].value_counts().sort_index()
            test_counts = test_df["rating"].value_counts().sort_index()
            dist_df = pd.DataFrame({
                "Train": train_counts,
                "Test": test_counts,
            }).fillna(0)
            st.bar_chart(dist_df, width="stretch")

with explore_tab:
    st.subheader("Eksplorasi anime")
    search_term = st.text_input("Cari judul")
    explore_df = anime_meta.copy()
    if search_term:
        explore_df = explore_df[explore_df["name"].str.contains(search_term, case=False, na=False)]

    st.dataframe(explore_df.head(50), width="stretch")
    st.download_button(
        label="Unduh tampilan saat ini (CSV)",
        data=explore_df.to_csv(index=False).encode("utf-8"),
        file_name="anime_metadata_preview.csv",
        mime="text/csv",
    )

    if "genre" in anime_meta.columns:
        with st.expander("Distribusi genre teratas", expanded=False):
            genre_series = anime_meta["genre"].dropna().astype(str)
            genre_counts = (
                genre_series.str.split(",")
                .explode()
                .str.strip()
                .replace("", np.nan)
                .dropna()
                .value_counts()
            )
            if not genre_counts.empty:
                st.bar_chart(genre_counts.head(15), width="stretch")

