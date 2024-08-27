from streamlit_echarts import st_echarts
import streamlit as st
from tracking_graph import hex_leicolors
import pandas as pd

@st.cache_data
def cache_compute_discarted_dfs(gpath, min_track):

    data = []
    for c in st.session_state['_discarted']:
        data.append({'segment':c.segment,'unit':c.unit,'N': st.session_state._sortings[c.segment][c.unit]['N']})
    discarted_df = pd.DataFrame(data)
    discarted_df_grouped = discarted_df.groupby('segment').agg({'unit': 'count', 'N': 'sum'}).reset_index()
    segments = range(1, max(st.session_state._sortings.keys()) + 1)
    discarted_df_grouped = discarted_df_grouped.set_index('segment').reindex(segments).fillna(0).reset_index()
    discarted_df_grouped.rename(columns={'unit': 'Discarted units', 'N': 'Discarted spikes'}, inplace=True)
    return discarted_df,discarted_df_grouped

if __name__ == "__main__":
    st.set_page_config(
        page_title="Tracking_Graph: Simplify Graph",
        page_icon="↔️",
        layout="wide"
    )
    if "_tg_graph_file" not in st.session_state or "_discarted" not in st.session_state:
        st.error("Load graph and check simplified graph first ")
    else:
        discarted_df,discarted_df_grouped = cache_compute_discarted_dfs(st.session_state._tg_graph_file, st.session_state['_min_track'])
        st.markdown('## Discarted units')
        st.bar_chart(discarted_df_grouped,y="Discarted units",x='segment')
        st.markdown('## Discarted spikes')
        st.bar_chart(discarted_df_grouped,y="Discarted spikes",x='segment')
        with st.expander("See full discarted units and spikes counts"):
            st.dataframe(discarted_df.sort_values(by=['segment','unit']))