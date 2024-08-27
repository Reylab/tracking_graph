from streamlit_echarts import st_echarts
import streamlit as st
from tracking_graph import tg_node,find_pos_genetic,number_to_mks_string,get_tg_groups
from tracking_graph import hex_leicolors
import networkx as nx

@st.cache_data(max_entries=3)
def cache_get_tg_groups(gpath, min_track,_G):
    groups,sG,discarted = get_tg_groups(_G, min_track)
    mean_waveforms = {}
    for gi,g in enumerate(groups):
        for c in g:
            cluster = st.session_state._sortings[c.segment][c.unit]
            if gi not in mean_waveforms:
                mean_waveforms[gi] = {}
            if c.segment not in mean_waveforms[gi]:
                mean_waveforms[gi][c.segment] = {}
                N_prod_wf = cluster['mean_waveform'] * cluster['N']
                mean_waveforms[gi][c.segment]['N'] = cluster['N']
                mean_waveforms[gi][c.segment]['mean_waveform'] = N_prod_wf
            else:
                N_prod_wf = cluster['mean_waveform'] * cluster['N']
                mean_waveforms[gi][c.segment]['N'] += cluster['N']
                mean_waveforms[gi][c.segment]['mean_waveform'] += N_prod_wf
    
    for u in mean_waveforms.keys():
        for s in mean_waveforms[u].keys():
            mean_waveforms[u][s]['mean_waveform'] /= mean_waveforms[u][s]['N']

    return groups,sG,discarted, mean_waveforms



@st.cache_data(max_entries=3)
def create_waveform_graph(gpath, min_track, selected,groups, _mean_waveforms):
    aux = _mean_waveforms[list(_mean_waveforms.keys())[0]]
    samples =  len(aux[list(aux.keys())[0]]['mean_waveform'])-1
      
    data_series = []
    max_segments = 0
    min_segments = 0
    data_legend = []
    for gi in selected:
        data = []
        for segment in _mean_waveforms[gi].keys():
            min_segments = min(min_segments,segment)
            max_segments = max(max_segments, segment)
            for i in range(samples+1):
                data.append((segment*samples+i, int(_mean_waveforms[gi][segment]['mean_waveform'][i])))
            data.append((None,None))
        data_legend.append("unit {}".format(gi))
        data_series.append({"type": "line",
                            "name": "unit {}".format(gi), "connectNulls": False, "emphasis": {"lineStyle": {"width": 4}},
        "data": data,'symbol': 'Circle' ,"showSymbol": False,"symbolSize": 1,'color':hex_leicolors(gi)})
        
        
           
    mark_areas = []
    for m in range(min_segments,max_segments+1):
        mark_areas.append([{"name": "{}".format(m), "xAxis": m*samples}, {"xAxis": (m+1)*samples}])


    waveform_graph = {
        "tooltip": {"trigger": "item",
            "axisPointer":{"type":"none",},
            "formatter": "{a}"
        },
        "legend": {
            "type": 'scroll', "left":20, "orient": 'vertical',
            "data": ['']+data_legend,  # Add legend entries for each series
            "selectedMode": "multiple"  # Allow multiple selections
        },
        "toolbox": {
            "show": True,
            "feature": {
                "restore": {},
                "zoom": {
                    "show": True,
                    "title": {
                        "zoom": "Zoom",
                        "back": "Reset Zoom"
                    }
                },
                "dataZoom": {
                    "show": True,
                    "title": {
                        "zoom": "Zoom",
                        "back": "Reset Zoom"
                    }
                },
                "pan": {
                    "show": True,
                    "title": "Pan"
                }
            }
        },
        "xAxis": {
            "type": "value",
            "max": (max_segments+1) * samples,  # Set the max limit for x-axis
            "axisLabel": {
                "show": False
            }
        },
        "dataZoom": [
            {
                "type": 'slider',
                "xAxisIndex": 0,
                "filterMode": 'none'  # Ensure edges are not filtered out
            },
            {
                "type": 'inside',
                "xAxisIndex": 0,
                "filterMode": 'none'  # Ensure edges are not filtered out
            },
            {
                "type": 'slider',
                "yAxisIndex": 0,
                "filterMode": 'none'  # Ensure edges are not filtered out
            }
        ],
        "yAxis": {
            "type": "value"
        },
        "series": [
            {
                "type": "line",
                "data": [],  # Segment 1
                "markArea": {
                    "data": mark_areas,
                    "itemStyle": {
                        "color": "rgba(0, 0, 0, 0.01)",
                        "borderColor": "rgba(0, 0, 0, 0.8)",
                        "borderWidth": 1
                    }
                }
            }]+data_series,
        "grid": {
            "left": "110",
            "right": "4%",
            "bottom": "15%",
            "containLabel": True
        }
    }
    return waveform_graph

@st.cache_data
def compute_egraph_elements(gpath, min_track, _G, _groups, _sG):
    node_to_simple_gf = {n:None for n in _G.nodes()}
    simple_gf = nx.DiGraph()
    classes_merged = {}
    simple_groups = []
    for groupnum,g in enumerate(groups):
        gelements = []
        for oldnode in g:
            nname = tg_node(unit=groupnum,segment=oldnode.segment)
            gelements.append(nname)
            if simple_gf.has_node(nname):
                classes_merged[nname].append(oldnode)
            else:
                simple_gf.add_node(nname)
                classes_merged[nname] = [oldnode]
            node_to_simple_gf[oldnode] = nname
        simple_groups.append(gelements)
    labels = {n: '{}'.format(number_to_mks_string(sum([st.session_state._sortings[xi.segment][xi.unit]['N'] for xi in x]))) for n, x in classes_merged.items()}
    for u, v in sG.edges():
        unew = node_to_simple_gf[u]
        vnew = node_to_simple_gf[v]
        if simple_gf.has_node(unew) and simple_gf.has_node(vnew):
            simple_gf.add_edge(unew, vnew)
    gpos = find_pos_genetic(simple_gf, simple_groups,generations=200)

    return gpos, simple_gf, classes_merged, labels

def return_egraph_elements(gpath, min_track, g_segment_gap,G, groups, sG):

    gpos, simple_gf, classes_merged, labels = compute_egraph_elements(gpath, min_track, G, groups, sG)

    maxtime = 0
    nodes = []
    for i, n in enumerate(simple_gf.nodes):
        if len(classes_merged[n])>1:
            shape='roundRect'
        else:
            shape='circle'
        maxtime = max(maxtime, n.segment)
        nodes.append({
            "id": str(n),
            "name":labels[n],
            "x": n.segment * g_segment_gap*20,#the 20 is to fix rounding erros for subpixel rendering
            "y":int(gpos[n.unit])*20, #the 20 is to fix rounding erros for subpixel rendering
            "value": """{}: size:{} merges:{}""".format(str(n),labels[n],len(classes_merged[n])),
            #"symbolSize": node_size / 100,
            "itemStyle": {"color": hex_leicolors(n.unit)},
            "symbol": shape
        })
        
    links = []
    for u, v, e in simple_gf.edges(data=True):

        if simple_gf.has_edge(v, u):
            symbol = ['circle', 'circle']
            symbolSize = [1, 1]
            color = 'blue'
            string = str(u) +"="+ str(v)
        else:
            symbol =  ['circle', 'arrow']
            symbolSize = [4, 10]
            color = 'red'
            string = str(u) +">"+ str(v)
        if (u.segment - v.segment)>1:
            curveness = -0.3
        elif (u.segment - v.segment)<-1:
            curveness = 0.3
        else:
            curveness = 0

        links.append({
            "source": str(u),
            "target": str(v),
            "symbol": symbol,
            "symbolSize": symbolSize,
            "lineStyle": {"curveness": curveness,
                          "color": color},
            "value": string,
            "emphasis" : {"lineStyle": {"width": 4}},
        })    
    options = {
        "tooltip": {"formatter":'{c}'},
        "toolbox": {
            "show": True,
            "feature": {
                "restore": {},
                "zoom": {
                    "show": True,
                    "title": {
                        "zoom": "Zoom",
                        "back": "Reset Zoom"
                    }
                },
                # "dataZoom": {
                #     "show": True,
                #     "title": {
                #         "zoom": "Zoom",
                #         "back": "Reset Zoom"
                #     }
                # },
                "pan": {
                    "show": True,
                    "title": "Pan"
                }
            }
        },
        # "xAxis": {
        #     "min" : 0,
        #     "max": maxtime
        # },
        # "yAxis": {
        #     "show": False,  # Hide the y-axis
        #     "max": max_y,
        #     "min" : 0
        # },
        # "dataZoom": [
        #     {
        #         "type": 'slider',
        #         "xAxisIndex": 0,
        #         "filterMode": 'none'  # Ensure edges are not filtered out
        #     },
        #     {
        #         "type": 'slider',
        #         "yAxisIndex": 0,
        #         "filterMode": 'none'  # Ensure edges are not filtered out
        #     }
        # ],
        "series": [
            {
                "type": 'graph',
                # "coordinateSystem": 'cartesian2d',
                "roam":True,
                "layout": 'none',
                "symbolSize": 30,
                "label": {
                    "show": True,
                    "fontWeight": 'bold',  # Make the font bold
                    "fontSize": 13
                },
                
                "edgeSymbol": ['circle', 'arrow'],
                "edgeSymbolSize": [4, 20],
                "edgeLabel": {
                    "fontSize": 20,
                    "formatter": '{@value}',
                },
                "emphasis": {"scale":True},
                "data": nodes,
                "links": links,
                "lineStyle": {
                    "opacity": 0.9,
                    "width": 2,
                    "curveness": 0
                }
            }
        ]
    }

    events = {
        "click": "function(params){if(params.dataType==='edge'){return{source:params.data.source,target:params.data.target}}else if(params.dataType==='node'){return{node:params.data.id}}}"
    }
    st.subheader("Simplified Graph")
    selected_element = st_echarts(options=options, events=events, height="600px",key='egraph')
    return selected_element

def change_min_track():
    st.session_state['_min_track'] = st.session_state['min_track']
    st.session_state['_selected'] = None
    if 'egraph' in  st.session_state:
        st.session_state['egraph']={}
        st.session_state['egraph']['node'] = ''

def change_g_segments_gap():
    st.session_state['_g_segments_gap'] = st.session_state['g_segments_gap']
    if 'egraph' in st.session_state:
        st.session_state['egraph']['node'] = ''
if __name__ == "__main__":
    st.set_page_config(
        page_title="Tracking_Graph: Simplify Graph",
        page_icon="↔️",
        layout="wide"
    )
    if "_min_track" not in st.session_state:
        st.session_state['min_track'] = 3
        st.session_state['_min_track'] = st.session_state['min_track']
    else:
        st.session_state['min_track'] = st.session_state['_min_track']

    if "_g_segments_gap" not in st.session_state:
        st.session_state['g_segments_gap'] = 1
        st.session_state['_g_segments_gap'] = st.session_state['g_segments_gap']
    else:
        st.session_state['g_segments_gap'] = st.session_state['_g_segments_gap']

    col1, col2, col3 = st.columns(3)
    with col1:
        number = st.number_input(
            "Minimun segments to keep", key='min_track', min_value=1,step=1,on_change=change_min_track)
    with col2:
        segments_graphic_gap = st.slider(
            "Graphic gap between segments", key='g_segments_gap', min_value=0.2,step=0.2, max_value=10.0,on_change=change_g_segments_gap)
    with col3:
        col31, col32 = st.columns(2)
        with col31:
            select_all_button = st.button("Select All",use_container_width =True)
            if select_all_button:
                st.session_state['_selected'] = None
                if 'egraph' in  st.session_state:
                    st.session_state['egraph']['node'] = ''
        with col32:
            clear_button = st.button("Clear selection",use_container_width =True)
            if clear_button:
                st.session_state['_selected'] = []
                if 'egraph' in  st.session_state:
                    st.session_state['egraph']['node'] = ''

    if "_tg_graph_file" not in st.session_state or st.session_state['_tg_graph_file'] == '':
        st.error("Load graph first")
    else:
        groups,sG,discarted,mean_waveforms = cache_get_tg_groups(st.session_state._tg_graph_file, st.session_state['_min_track'],st.session_state._G)
        if st.session_state['_selected'] is None:
            st.session_state['_selected'] = list(range(len(groups)))
        st.session_state['_groups'] = groups
        st.session_state['_discarted'] = discarted
        selected_element = return_egraph_elements(st.session_state._tg_graph_file, st.session_state['_min_track'],st.session_state['_g_segments_gap'], st.session_state._G, groups, sG)

        if selected_element:
            #st.write(f'Clicked element: {selected_element}')
            if list(st.session_state['egraph'].keys())[0] == 'node':
                if st.session_state['egraph']['node'] != '':
                    clicked_unit = tg_node.from_string(st.session_state['egraph']['node']).unit
                    if clicked_unit in st.session_state['_selected']:
                        st.session_state['_selected'].remove(clicked_unit)
                    else:
                        st.session_state['_selected'].append(clicked_unit)
            #st.write(st.session_state['_selected'])

    if st.session_state['_selected']:
        # Render the ECharts line chart in Streamlit
        st.markdown("### Mean Waveforms per segment")
        st_echarts(options=create_waveform_graph(st.session_state._tg_graph_file, st.session_state['_min_track'],st.session_state['_selected'],groups,mean_waveforms), height="400px", key='mean_waveforms_plot', width="100%")