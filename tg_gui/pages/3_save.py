import streamlit as st
import os
import pandas as pd

def create_default_filename(graph_file):
    filename = os.path.splitext(graph_file)[0]
    return f"{filename}_min_track_{st.session_state['_min_track']}.csv"

@st.cache_data
def groups_to_df(groups):
    df = []
    for gi,g in enumerate(groups):
        for c in g:
            df.append({'segment':c.segment,'cluster':c.unit,'tg_unit':gi})
    return pd.DataFrame(df)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Tracking_Graph: Save results",
        page_icon="💾",
        layout="wide"
    )

    st.markdown(
        """
        ## Export results
    """
    )
    if  '_groups' not in st.session_state:
        st.error("Please load results and run graph before exporting.")
    else:
        if "_export_file_name" not in st.session_state:
            st.session_state._export_file_name = create_default_filename(st.session_state._tg_graph_file)
        st.session_state.export_file_name = st.session_state._export_file_name
        
        col1, col2 = st.columns([1,0.2])
        with col1:
             output = st.text_input("## Output file path",key="export_file_name")
        with col2:
            change_default = st.button("Change to default")
            save_button =st.button("Save to file")
            

        if change_default:
            st.session_state._export_file_name = create_default_filename(st.session_state._tg_graph_file)
        df = groups_to_df(st.session_state._groups)

        if save_button:
            df.to_csv(st.session_state.export_file_name, index=False)
            st.success("DataFrame saved successfully!")

        with st.expander("See full assigment table"):
            st.dataframe(df.sort_values(by=['segment','cluster']))
    #C:\Users\fernando.chaure\Documents\GitHub\tracking_graph\EMU-004_subj-MCW-FH_002_task-gaps260_std_3_tesis.json