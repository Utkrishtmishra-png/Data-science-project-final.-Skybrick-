"""Streamlit UI for the content-based movie recommender.  Run: streamlit run app.py"""
import streamlit as st

from src.recommender import DEFAULT_DATA, MovieRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
    .card {background:#161a23;border:1px solid #2a3040;border-radius:14px;padding:16px 18px;margin-bottom:14px;}
    .card h4 {margin:0 0 4px 0;color:#fff;}
    .meta {color:#9aa4b8;font-size:.85rem;margin-bottom:8px;}
    .pill {display:inline-block;background:#2b3a67;color:#cfd8ff;border-radius:20px;padding:2px 10px;font-size:.75rem;margin-right:6px;}
    .match {float:right;color:#5eead4;font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Building similarity model...")
def get_model(path: str) -> MovieRecommender:
    return MovieRecommender.from_csv(path)


def render(recs):
    if not recs:
        st.info("No matches found. Try a different title, genre or description.")
    for r in recs:
        pills = "".join(f'<span class="pill">{g}</span>' for g in r.genres.split(", ") if g)
        st.markdown(
            f"""<div class="card"><span class="match">{r.score:.0%} match</span>
            <h4>{r.title} <span style="color:#9aa4b8;font-weight:400">({r.year})</span></h4>
            <div class="meta">Directed by {r.director or 'Unknown'}</div>
            {pills}<p style="margin-top:10px;color:#d5dbe8">{r.overview}</p></div>""",
            unsafe_allow_html=True,
        )


with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Use your own dataset (CSV)", type="csv",
                                help="Needs columns: title, genres, overview (optional: year, keywords, cast, director). Use | to separate list values.")
    n = st.slider("Number of recommendations", 3, 15, 6)

data_path = DEFAULT_DATA
if uploaded is not None:
    data_path = "uploaded.csv"
    with open(data_path, "wb") as f:
        f.write(uploaded.getbuffer())

try:
    model = get_model(str(data_path))
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load dataset: {e}")
    st.stop()

st.title("🎬 Content-Based Movie Recommender")
st.caption(f"{len(model.df)} movies loaded · TF-IDF on genres, keywords, cast, director & plot · cosine similarity")

tab1, tab2 = st.tabs(["Find similar movies", "Describe what you want"])

with tab1:
    c1, c2 = st.columns([3, 1])
    title = c1.selectbox("Pick a movie you like", model.titles, index=None, placeholder="Start typing a title...")
    genre = c2.selectbox("Filter by genre", ["Any"] + model.all_genres())
    if title:
        st.subheader(f"Because you liked *{title}*")
        render(model.recommend(title, n=n, genre=None if genre == "Any" else genre))

with tab2:
    query = st.text_input("Describe a movie", placeholder="e.g. astronaut survival on a lonely planet")
    if query:
        render(model.search(query, n=n))
