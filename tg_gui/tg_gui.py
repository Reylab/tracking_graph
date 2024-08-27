import streamlit as st
import os
from tracking_graph import  load_tg,read_to_gml,compute_model_quality,compute_connection_quality
from streamlit_echarts import st_echarts
import numpy as np
import pandas as pd
@st.cache_data
def load_graph(file_path):
    G = read_to_gml(st.session_state._tg_graph_file)
    return G

@st.cache_data(max_entries=2)
def _compute_model_quality(gpath,_G,type):
    if type=='connection':
        matrix, nodelist = compute_connection_quality(_G)
    else:
        matrix, nodelist = compute_model_quality(_G)
    nodelabels = []
    for node in nodelist:
        nodelabels.append(f"S:{node.segment}|C:{node.unit}")
    return matrix, nodelist,nodelabels

def on_text_change():
    graph_exists = os.path.exists(st.session_state.tg_graph_file)
    if graph_exists:
        st.success(f"Graph loaded: {st.session_state.tg_graph_file}")
        st.session_state._tg_graph_file = st.session_state.tg_graph_file
    else:
        st.session_state._tg_graph_file = ''
        st.error("File doesn't exist.")
    st.session_state['_selected'] = None
    if '_groups' in st.session_state:
        del st.session_state['_groups']
    if '_discarted' in st.session_state:
        del st.session_state['_discarted']

if __name__ == "__main__":
    st.set_page_config(
        page_title="Tracking_Graph: Modeling error",
        page_icon="⚠️",
        layout="wide"
    )

    st.markdown(
        """
        ## Measuring the error of tracking graph model
    """
    )
    if "_tg_graph_file" not in st.session_state:
        st.session_state._tg_graph_file = ""
        st.session_state.tg_graph_file = st.session_state._tg_graph_file
    elif st.session_state._tg_graph_file != "":
        st.session_state.tg_graph_file = st.session_state._tg_graph_file
    
    input = st.text_input("## Input graph file", key="tg_graph_file",placeholder='Full path is required', on_change=on_text_change)
   
    if st.session_state._tg_graph_file != "":
 
        G,sortings = load_tg(st.session_state._tg_graph_file)
        st.session_state._G = G
        st.session_state._sortings = sortings
        st.session_state._tg_sortings = None


        weights=[]
        for u, v, weight in G.edges(data="weight"):
            weights.append(weight)

        counts, bin_edges = np.histogram(weights, bins=20)

        total_counts = sum(counts)
        percentages = (counts / total_counts) * 100

        # Create a DataFrame with the histogram values
        histogram_df = pd.DataFrame({
            'Bin Start': bin_edges[:-1],
            'Bin End': bin_edges[1:],
            'Percentage': percentages
        })

        bar_chart_df = pd.DataFrame({
            'Edge Weight': ["{:.2f}".format((bin_edges[i+1]+bin_edges[i])/2) for i in range(len(bin_edges)-1)],
            'Percentage': percentages
        })
        st.markdown("### Weights histogram")
        st.bar_chart(bar_chart_df, x='Edge Weight', y="Percentage")
        error_type=st.radio("Select the type of quality you want to measure",('connection','edge'),key='error_type')
        
        st.markdown("### Heatmap of {} errors".format(error_type))
        matrix, nodelist,nodelabels=_compute_model_quality(st.session_state._tg_graph_file,st.session_state._G,error_type) 
        
        heatmap_data = []
        inodelabels = []
        jnodelabels = []
        for i,nodei in enumerate(nodelist):
            for j,nodej in enumerate(nodelist):
                if (nodej,nodei) in G.edges and (error_type!='connection' or i>j):
                    heatmap_data.append([nodelabels[i], nodelabels[j], matrix[i, j]])
                    if error_type=='connection':
                        inodelabels.append(i)
                        jnodelabels.append(j)
                else:
                    heatmap_data.append([nodelabels[i], nodelabels[j], None])
        
        if error_type=='connection':
            xnodelabels = [nodelabels[j] for j in sorted(set(inodelabels))]
            ynodelabels = [nodelabels[i] for i in sorted(set(jnodelabels))]
        else:
            xnodelabels = nodelabels
            ynodelabels = nodelabels
        # Create the heatmap options for echarts
        heatmap_options = {
        "toolbox": {
            "show": True,
            "feature": {
                "restore": {}
            },
            
        },
        "dataZoom": [
            {
                "type": 'slider',
                "xAxisIndex": 0,
                "filterMode": 'none'
            },
            {
                "type": 'inside',
                "xAxisIndex": 0,
                "filterMode": 'none' 
            },
            {
                "type": 'inside',
                "yAxisIndex": 0,
                "filterMode": 'none'
            },
            {
                "type": 'slider',
                "left":'left',
                "yAxisIndex": 0,
                "filterMode": 'none'
            }
        ],
            "tooltip": {
                "position": "top",
                "formatter": "{c0}",
            },

            "xAxis": {
                "type": "category",
                "data": xnodelabels,
                "splitArea": {
                    "show": True
                }
            },
            "yAxis": {
                "type": "category",
                "data": ynodelabels,
                "splitArea": {
                    "show": True
                }
            },
            "visualMap": {
                "type": "continuous",
                "min": 0,
                "max": 1,
                "inRange" : {"color": ['#FDE725','#F1605D', '#440154' ]  },
                "precision": 2,
                "calculable": True,
                "orient": "vertical",
                "right": "0%",
                "top": "10%",
                "itemHeight": "500",
            },
            "grid": {
                "top": "1%",
                "bottom": "13%",
                "left": "10%",
                "right": "10%"
            },
            "series": [{
                "name": "Values",
                "type": "heatmap",
                "data": heatmap_data,
                "label": {
                    "show": False
                }
            }]
        }

        # Display the heatmap in Streamlit

        st_echarts(options=heatmap_options, height="600px")
        #C:\Users\fernando.chaure\Documents\GitHub\tracking_graph\EMU-004_subj-MCW-FH_002_task-gaps260_std_3_tesis.json
    clean_cache = st.button("Clean cache", help="Useful after overwriting the input files.")
    if clean_cache:
        st.cache_data.clear()