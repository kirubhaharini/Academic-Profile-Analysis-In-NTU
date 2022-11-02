import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from ast import literal_eval
import numpy as np
import regex as re
import time
import math
from plotly.subplots import make_subplots
st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

##############
@st.cache(suppress_st_warning=True)
def load_data(filename):
    df = pd.read_csv(filename)
    return df

#############
@st.cache(suppress_st_warning=True)
def title_composition(df):
    titles = []
    for row in df['cleaned titles']:
        row = literal_eval(row)
        titles.append(row)
        
    titles = [item for sublist in titles for item in sublist]
    count = {}
    for elem in set(titles):
        count[elem] = 0
    for elem in titles:
        count[elem] += 1
    labels = list(count.keys())
    values = list(count.values())
    # st.write(count)
    return [labels,values]

##################
@st.cache(suppress_st_warning=True)
def research_areas(df,areas):

    ##### avg career length by sch
    schs = []
    for row in df['cleaned schools']:
        row = literal_eval(row)
        schs.append(row)
    schs = [item for sublist in schs for item in sublist]
    schools = set(schs)

    years_by_school = {}
    avg_career_length = {}

    for sch in schools:
        years_by_school[sch] = []
        for row in range(len(df)):
            researcher_schools = df.loc[row,'cleaned schools']
            if sch in researcher_schools: #add years to list
                yr = df.loc[row,'career length']
                if not pd.isna(yr):
                    years_by_school[sch].append(yr)
        avg_career_length[sch] = round(sum(years_by_school[sch])/len(years_by_school[sch]),1)
        years_by_school[sch] = [min(years_by_school[sch]),max(years_by_school[sch])]

    ##### graph
    fig = make_subplots(specs=[[{"secondary_y": True}]])#this a one cell subplot
    fig.update_layout(
                    template="plotly_white",legend=dict(orientation='v'))

    trace1 = go.Bar(x=areas['Research Areas'], y=areas['Average Publications'], opacity=0.5,name='Average Publications',marker_color ='#1f77b4')

    trace2p = go.Scatter(x=areas['Research Areas'], y=areas['Average Citations Growth Rate'],name='Average Citations Growth Rate',mode='lines+markers',line=dict(color='#e377c2', width=2))

    #The first trace is referenced to the default xaxis, yaxis (ie. xaxis='x1', yaxis='y1')
    fig.add_trace(trace1, secondary_y=False)

    #The second trace is referenced to xaxis='x1'(i.e. 'x1' is common for the two traces) 
    #and yaxis='y2' (the right side yaxis)

    fig.add_trace(trace2p, secondary_y=True)

    fig.update_yaxes(#left yaxis
                    title= 'Number of Publications',showgrid= False, secondary_y=False)
    fig.update_xaxes(
                    title= 'Top Research Areas')
    fig.update_yaxes(#right yaxis
                    showgrid= True, 
                    title= 'Citation Growth Rate (%)',
                    secondary_y=True)

    fig.update_layout(
    autosize=False,
    width=1000,
    height=500,
    margin=dict(
        l=10,
        r=200,
        b=200,
        t=40,
    ),
    )
    return [avg_career_length,fig]

####################
@st.cache(suppress_st_warning=True)
def plot_gauge_charts(mean_h_index_over_years,mean_citation_growth_rate):
    fig1 = go.Figure(go.Indicator(
        mode = "gauge+number",
        delta = {'reference': 100},
        value = round(mean_h_index_over_years,2),
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = { 'axis': { 'range': [0, 100] } }
    ))
    fig2 = go.Figure(go.Indicator(
        mode = "delta",
        value = mean_citation_growth_rate,
        delta = {'reference': 0}, #above 0 : +ve growth
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    fig2.update_layout(
    autosize=False,
    width=600,
    height=300,
    margin=dict(
        l=50,
        r=50,
        b=0,
        t=0,
    ),
    )

    return [fig1,fig2]

#####################
def get_co_authors_info(co):
    names = []
    unis = []
    urls = []
    if type(co) != str:
        co = ''
    else:
        co = literal_eval(co)
        vals = list(co.values())
        for i in vals:
            name= i[0]
            names.append(name)
            uni = i[1]
            unis.append(uni)
        urls = list(co.keys())
        
    return [names,unis,urls]

####################
@st.cache(suppress_st_warning=True)
def get_personal_info(r1):
    dr_url = r1['DR-NTU URLs'][0]
    email = r1['Email ID'][0]
    website = r1['Website URL'][0]
    orc = r1['ORC ID'][0]
    titles = r1['Titles'][0]
    schs = r1['Schools'][0]
    disp = ''
    if type(titles) == str:
        titles = literal_eval(titles)
        if type(schs) == str:
            try:
                schs = literal_eval(schs)
                for i in range(len(titles)):
                    disp += str('\n'+titles[i]+ ', '+schs[i] + '\n')
            except: 
                for i in range(len(titles)):
                    disp += str('\n'+titles[i]+ ', '+ '\n')
                    disp += schs
        else: # only title: 
            for i in range(len(titles)):
                disp += str('\n'+titles[i]+ ', '+'\n')
    elif type(schs) == str: #only schs
        try: 
            schs = literal_eval(schs)
            for i in range(len(schs)):
                disp += str('\n'+schs[i]+ ', '+ '\n')
        except: 
            disp += schs
    ggsch = r1['Google Scholar URL'][0]

    return [email,dr_url,website,ggsch,orc,disp]


###################
@st.cache(suppress_st_warning=True)
def plot_career_lengths(df):
    titles_career_lengths = {}
    titles = []
    for row in range(len(df)):
        title = df.loc[row,'cleaned titles']
        if title:
            title = list(literal_eval(title))
            for tit in title:
                titles.append(tit)
    titles = set(titles)
    for title in set(titles):
        titles_career_lengths[title] = []
    for row in range(len(df)):
        title = df.loc[row,'cleaned titles']
        career_length = df.loc[row,'career length']
        if title:
            title = list(literal_eval(title))
            for tit in title:
                if math.isnan(career_length) == True:
                    pass
                else: titles_career_lengths[tit].append(career_length)
            
    for k,v in titles_career_lengths.items():
        v = [x for x in v if str(x) != 'nan']
        titles_career_lengths[k] = [min(v),max(v)]

    titles = list(titles_career_lengths.keys())
    min_v = [x[0] for x in list(titles_career_lengths.values())]
    max_v = [x[1] for x in list(titles_career_lengths.values())]
    title_by_career_length = pd.DataFrame({'Titles':titles,'Min':min_v,'Max':max_v})


    ####### figure #######
    d = pd.melt(title_by_career_length,id_vars = 'Titles',value_vars=title_by_career_length[:],value_name='avg career length')
    d = d.sort_values(by='avg career length',ascending=False)
    min_df = d[d['variable']=='Min']
    max_df = d[d['variable']=='Max']

    #plot graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=min_df['Titles'], x=min_df['avg career length'], name="min year",
                        line_shape='linear'))
    fig.add_trace(go.Scatter(y=max_df['Titles'], x=max_df['avg career length'], name="max year",
                        text=["tweak line smoothness<br>with 'smoothing' in line object"],
                        hoverinfo='text+name',
                        line_shape='linear'))
    fig.update_traces(hoverinfo='name+x', mode='markers')
    fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))

    for dept in min_df['Titles']:
        min_val = min_df[min_df['Titles']==dept].reset_index(drop=True)['avg career length'][0]
        max_val = max_df[max_df['Titles']==dept].reset_index(drop=True)['avg career length'][0]
        fig.add_shape(type='line',
                        x0=min_val,
                        y0=dept,
                        x1=max_val,
                        y1=dept,
                        line=dict(color='lightblue',),
                        xref='x',
                        yref='y'
                    )
    fig.update_layout(xaxis_range=[0,41],xaxis_title="Number of Years",yaxis_title='Titles')

    fig.update_layout(
    autosize=False,
    width=1000,
    height=600,
    margin=dict(
        l=60,
        r=200,
        b=200,
        t=30,
    ),
    )
    return fig

####################
@st.cache(suppress_st_warning=True)
def citations_by_year(x1,y1,color):
    fig = px.line(x=x1, y=y1)
    fig.add_trace(
        go.Scatter(x=x1,
                y=y1,
                name="Citations",
                visible=True,
                line=dict(color=color, dash="solid")))
    fig.update_layout(showlegend=False,xaxis_title="Years",yaxis_title="Number of Citations")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)')
    return fig

####################
@st.cache(suppress_st_warning=True)
def make_length_equal(x1,y1):
    extra = len(y1) - len(x1)
    #add extra years to x1
    p = x1[0]
    for i in range(1,extra+1):
        x1.append(p-i)
    x1 = sorted(x1)
    return x1
#####################

df = load_data('FINAL DATA.csv')
scse = df[df['sch']=='School of Computer Science and Engineering']
areas = load_data('research areas.csv')
nus = load_data('nus profiles.csv')

######################################################################################################
def main_page():
    st.markdown("<h1 style='text-align: center; color: blue;'>SCSE (NTU) Statistics</h1>", unsafe_allow_html=True)
    st.markdown(" \n ")
    st.sidebar.markdown("# NTU SCSE Statistics")
    st.markdown("<h3 style='text-align: center; color: salmon;'>Distribution of Academic Profiles in SCSE</h3>", unsafe_allow_html=True)
    info = title_composition(scse)
    pie_chart = go.Figure(data=[go.Pie(labels=info[0], values=info[1])])
    figure, figure2 = st.columns((2, 7))
    with figure2:
        st.plotly_chart(pie_chart,height=2000,width=2000)

    avg_career_length, line_chart = research_areas(df,areas)
    st.markdown("<h3 style='text-align: center; color: salmon;'>Performance in Research Areas</h3>", unsafe_allow_html=True)
    a, b = st.columns((2, 7))
    with b:
        st.plotly_chart(line_chart)

    
    st.markdown("<h3 style='text-align: center; color: salmon;'>Career Lengths by Titles</h3>", unsafe_allow_html=True)
    careers = plot_career_lengths(df)
    c,d = st.columns((2, 7))
    with d:
        st.plotly_chart(careers)

################ compare schools ###############
    st.markdown("""------""")
    st.markdown(""" """)
    st.markdown("<h1 style='text-align: center; color: blue;'>Compare Faculties</h1>", unsafe_allow_html=True)
    schs = []
    for row in df['cleaned schools']:
        row = literal_eval(row)
        schs.append(row)
    schs = [item for sublist in schs for item in sublist]
    schools = set(schs)
    temp_schs_list = schools
    temp_schs_list =  [value for value in temp_schs_list if value != 'Others']
    temp_schs_list += ['NUS Computing']
    multiple_schools = st.checkbox('Compare with another school', value=False)
    # placeholder = st.empty()

    school1 = 'School of Computer Science and Engineering'
    if multiple_schools: # 2 schools
        school2 = st.selectbox("Select a school",temp_schs_list)
        sch1,sch2 = st.columns(2)
        
        
        with sch1:
            st.markdown("<h4 style='text-align: center; color: steelblue;'>School of Computer Science and Engineering</h4>", unsafe_allow_html=True)
            # school1 = st.selectbox("Select a school",schools)
            temp_schs_list =  [value for value in temp_schs_list if value != school1]
            sch_df = df[df['cleaned schools'].str.contains(school1)]
            # get avg career length progress bar:
            avg_yrs = float(avg_career_length[school1])
            progress_bar_sch1 = st.progress(0)
            status_text_sch1 = st.empty()
            for i in range(int(avg_yrs)):
                progress_bar_sch1.progress(i + 1)
                status_text_sch1.text(
                    'Average Career Length: ' + str(i+1) + ' years')
                time.sleep(0.1)
            status_text_sch1.text(
                    'Average Career Length: ' + str(avg_yrs) + ' years')
            mean_publications_over_years = sum(sch_df['Total Publications'].dropna())/len(sch_df['Total Publications'].dropna())
            mean_citation_growth_rate = sum(sch_df['citation growth rate'].dropna())/len(sch_df['citation growth rate'].dropna())
            top5 = sch_df.sort_values(by=['h_index/years','citation growth rate'],ascending=False)[:5].reset_index(drop=True)
            fig1,fig2 = plot_gauge_charts(mean_publications_over_years,mean_citation_growth_rate)
            # colm,colo,coln = st.columns([9,3,6])
            # with colm:
            st.markdown("<h5 style='text-align: center; color: crimson;'>Average Number of Publications per year</h5>", unsafe_allow_html=True)
            st.plotly_chart(fig1)
            # with coln:
            st.markdown("<h5 style='text-align: center; color: crimson;'>Mean Citation Growth Rate (% over past 3 years)</h5>", unsafe_allow_html=True)
            st.plotly_chart(fig2)

        with sch2:
            ph = st.empty()
            html_str = f""" <h4 style='text-align: center; color: steelblue;'>{school2}</h4> """
            ph.markdown(html_str, unsafe_allow_html=True)
            if school2 == 'NUS Computing':
                sch_df = nus
                avg_yrs = sum(nus['career length'].dropna())/len(nus['career length'].dropna())
            else: 
                sch_df = df[df['cleaned schools'].str.contains(school2)]
                # get avg career length progress bar:
                avg_yrs = float(avg_career_length[school2])
            progress_bar_sch1 = st.progress(0)
            status_text_sch1 = st.empty()
            for i in range(int(avg_yrs)):
                progress_bar_sch1.progress(i + 1)
                status_text_sch1.text(
                    'Average Career Length: ' + str(i+1) + ' years')
                time.sleep(0.1)
            status_text_sch1.text(
                    'Average Career Length: ' + str(round(avg_yrs,1)) + ' years')

            mean_publications_over_years = sum(sch_df['Total Publications'].dropna())/len(sch_df['Total Publications'].dropna())
            mean_citation_growth_rate = sum(sch_df['citation growth rate'].dropna())/len(sch_df['citation growth rate'].dropna())
            top5 = sch_df.sort_values(by=['h_index/years','citation growth rate'],ascending=False)[:5].reset_index(drop=True)
            fig1,fig2 = plot_gauge_charts(mean_publications_over_years,mean_citation_growth_rate)
            # colm,colo,coln = st.columns([9,3,6])
            # with colm:
            st.markdown("<h5 style='text-align: center; color: crimson;'>Average Number of Publications per year</h5>", unsafe_allow_html=True)
            st.plotly_chart(fig1)
            # with coln:
            st.markdown("<h5 style='text-align: center; color: crimson;'>Mean Citation Growth Rate (% over past 3 years)</h5>", unsafe_allow_html=True)
            st.plotly_chart(fig2)
            


    else: # just 1 school
        # school1 = st.selectbox("Select a school",schools)
        st.markdown("<h5 style='text-align: center; color: steelblue;'>School of Computer Science and Engineering</h5>", unsafe_allow_html=True)
        
        sch_df = scse #df[df['cleaned schools'].str.contains(school1)]
        # get avg career length progress bar:
        avg_yrs = float(avg_career_length[school1])
        progress_bar_sch1 = st.progress(0)
        status_text_sch1 = st.empty()
        for i in range(int(avg_yrs)):
            progress_bar_sch1.progress(i + 1)
            status_text_sch1.text(
                'Average Career Length: ' + str(i+1) + ' years')
            time.sleep(0.1)
        status_text_sch1.text(
                'Average Career Length: ' + str(avg_yrs) + ' years')
        mean_h_index_over_years = sum(sch_df['h_index/years'].dropna())/len(sch_df['h_index/years'].dropna())
        mean_citation_growth_rate = sum(sch_df['citation growth rate'].dropna())/len(sch_df['citation growth rate'].dropna())
        top5 = sch_df.sort_values(by=['h_index/years','citation growth rate'],ascending=False)[:5].reset_index(drop=True)
        fig1,fig2 = plot_gauge_charts(mean_h_index_over_years,mean_citation_growth_rate)
        colm,colo,coln = st.columns([9,3,6])
        with colm:
            st.markdown("<h5 style='text-align: center; color: crimson;'>Mean h-index (calibrated by years)</h5>", unsafe_allow_html=True)
            fig1 = go.Figure(go.Indicator(
            mode = "number",
            # delta = {'reference': 100},
            value = round(mean_h_index_over_years,2),
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = { 'axis': { 'range': [0, 100] } }
        ))
            fig1.update_layout(
                    autosize=False,
                    width=600,
                    height=300,
                    margin=dict(
                        l=50,
                        r=50,
                        b=50,
                        t=0,
                    ),
                    )
                
            st.plotly_chart(fig1)
        with coln:
            st.markdown("<h5 style='text-align: center; color: crimson;'>Mean Citation Growth Rate (% over past 3 years)</h5>", unsafe_allow_html=True)
            st.plotly_chart(fig2)

        # top 5 profs
        st.markdown("<h5 style='text-align: center; color: crimson;'>Best Performing Staff</h5>", unsafe_allow_html=True)
        names, prof1, prof2, prof3, prof4, prof5 = st.columns(6)
        profs = [prof1, prof2, prof3, prof4, prof5]
        with names: 
            st.markdown("***")
            st.markdown("<h6 style='text-align: center; color: steelblue;'>Name</h6>", unsafe_allow_html=True)
            st.markdown("<h6 style='text-align: center; color: steelblue;'>Calibrated h-index</h6>", unsafe_allow_html=True)
            st.markdown("<h6 style='text-align: center; color: steelblue;'>Citation Growth Rate</h6>", unsafe_allow_html=True)
            st.markdown("<h6 style='text-align: center; color: steelblue;'>Career Length</h6>", unsafe_allow_html=True)
            st.markdown("<h6 style='text-align: center; color: steelblue;'>Most popular venues</h6>", unsafe_allow_html=True)
            # st.markdown("***")
        for i in range(len(profs)):
            prof = profs[i]
            with prof:
                st.markdown("***")
                st.write(top5['Names'][i])
                # write
                if not math.isnan(top5['h_index/years'][i]):
                    h_index = round(top5['h_index/years'][i],2)
                st.write(h_index)
                if not math.isnan(top5['citation growth rate'][i]):
                    cites = top5['citation growth rate'][i]
                else: cites = '-'
                st.write(cites)
                if not math.isnan(top5['career length'][i]):
                    yrs = top5['career length'][i]
                else: yrs = '-'
                st.write(yrs)
                if type(top5['dblp venues'][i]) == str:
                    venues = literal_eval(top5['dblp venues'][i])
                    if len(venues)>0:
                        list_venues = venues[0]
                        for v in venues[1:]:
                            list_venues = list_venues + ', ' + v
                        venues = list_venues
                    else: venues = '-'
                else: venues = '-'
                st.write(venues)
                # st.markdown("***")
                

    
            
######################################################################################################




def page2():
    st.title("Researchers")
    st.sidebar.markdown("# Compare Researchers")
    scse_profs = scse['Names'].to_list()
    all_ntu_profs = df['Names'].to_list()
    options = ["Within SCSE", ' ', ' ', "Within NTU",' ', ' ',  "With NUS Computing"]
    tab1, tab2, tab3,tab4,tab5,tab6,tab7 = st.tabs(options)
    tabs_font_css = """
    <style>
    button[data-baseweb="tab"] {
    font-size: 18px;
    }
    </style>
    """

    st.write(tabs_font_css, unsafe_allow_html=True)

    with tab1: #within SCSE

        col1,col2,col3 = st.columns([3,2,4])
        
        with col1:
            st.text('Researcher 1')
            r_1 = st.selectbox(" ", scse_profs)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
        with col3:
            st.text('Researcher 2')
            temp_profs_list =  [value for value in scse_profs if value != r_1]
            r_2 = st.selectbox(" ", temp_profs_list)
            progress_bar2 = st.progress(0)
            status_text2 = st.empty()
        
        ################### personal details:
        r1 = scse[scse['Names'] == r_1].reset_index(drop=True)
        r2 = scse[scse['Names'] == r_2].reset_index(drop=True)
        col1,col2,col3 = st.columns([3,2,4])
        
        cols = ['Names','DR-NTU URLs', 'Email ID', 'Website URL', 'ORC ID', 'Titles', 'Schools','Google Scholar URL']
        with col1:

            email,dr_url,website,ggsch,orc,disp = get_personal_info(r1)
            st.write(':email: '+"[Email]("+(email)+")" +'\t\t'+':link: '+"[DR-NTU]("+(dr_url)+")")
            if type(website) == str:
                st.write(':link: '+"[Personal Website]("+(website)+")")
            if type(ggsch) == str:
                st.write(':male-student:'+"[Google Scholar]("+(ggsch)+")")
            if type(orc) == str:
                st.write(':link: '+"[ORC ID]("+(orc)+")")
            st.write(':school: '+"Schools: "+(disp))
        with col3:
            email,dr_url,website,ggsch,orc,disp = get_personal_info(r2)
            st.write(':email: '+"[Email]("+(email)+")" +'\t\t'+':link: '+"[DR-NTU]("+(dr_url)+")")
            if type(website) == str:
                st.write(':link: '+"[Personal Website]("+(website)+")")
            if type(ggsch) == str:
                st.write(':male-student:'+"[Google Scholar]("+(ggsch)+")")
            if type(orc) == str:
                st.write(':link: '+"[ORC ID]("+(orc)+")")
            st.write(':school: '+"Schools: "+(disp))

        ######## tot pubs ##########
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Total No. of Publications</h5>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns([3,2,4])
        with col1:
            rs1 = scse[scse['Names'] == r_1].reset_index(drop=True)
            pubs = rs1['Total Publications'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = int(pubs)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)
        
        with col3:
            rs2 = scse[scse['Names'] == r_2].reset_index(drop=True)
            pubs = rs2['Total Publications'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = int(pubs)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)

        ######## tot no of lead papers ##########
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Papers as Lead Author (for past 10 recent papers)</h5>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns([3,2,4])
        with col1:
            rs1 = scse[scse['Names'] == r_1].reset_index(drop=True)
            pubs = rs1['count of lead papers'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = int(pubs)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)
        
        with col3:
            rs2 = scse[scse['Names'] == r_2].reset_index(drop=True)
            pubs = rs2['count of lead papers'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = int(pubs)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)

        ################ career length ################
        #for researcher 1
        
        year1 = r1['years'][0]
        if type(year1) == str:
            x1 = literal_eval(year1)
            for i in range(x1[-1]-x1[0]):
                progress_bar.progress(i + 1)
                status_text.text(
                    'Career Length: ' + str(i+1) + ' years')
                time.sleep(0.1)
        else: status_text.text(
                    'Career Length: Not Available')
        #for researcher 2
        year2 = r2['years'][0]
        if type(year2) == str:
            x2 = literal_eval(year2)
            for i in range(x2[-1]-x2[0]):
                # Update progress bar.
                progress_bar2.progress(i + 1)
                status_text2.text(
                    'Career Length: ' + str(i+1) + ' years')
                time.sleep(0.1)
        else: status_text2.text(
                    'Career Length: Not Available')

        st.markdown('***')

        ############## h_index
        
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Average h-index (calibrated by years)</h5>", unsafe_allow_html=True)
        col10,col11,col12 = st.columns([3,2,4])

        with col10:
            pubs = rs1['h_index/years'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = round(pubs,2)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)
            
        with col12:
            pubs = rs2['h_index/years'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = round(pubs,2)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)

        ############## h index, citation rate, # publications
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Citation Growth Rate (% over past 3 years)</h5>", unsafe_allow_html=True)
        col10,col11,col12 = st.columns([3,2,4])

        with col10:
            mean_h_index_over_years = round(r1['h_index percentile'][0],2)
            mean_citation_growth_rate = r1['citation growth rate'][0]
            fig1,fig2 = plot_gauge_charts(mean_h_index_over_years,mean_citation_growth_rate)
            
            st.plotly_chart(fig2)
            
        with col12:
            mean_h_index_over_years = round(r2['h_index percentile'][0],2)
            mean_citation_growth_rate = r2['citation growth rate'][0]
            fig3,fig4 = plot_gauge_charts(mean_h_index_over_years,mean_citation_growth_rate)
            st.plotly_chart(fig4)
        

        #### citations per year
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Citations by Year</h5>", unsafe_allow_html=True)
        col16,col17,col18 = st.columns([3,2,4])
        year1 = r1['years'][0]
        x1 = x2 = None
        if type(year1) == str:
            x1 = literal_eval(year1)
            y1 = r1['citations by year'][0]
            y1 = literal_eval(y1)
            if len(x1) < len(y1):
                x1 = make_length_equal(x1,y1)
            elif len(y1) < len(x1):
                y1 = make_length_equal(y1,x1)
                

        year2 = r2['years'][0]
        if type(year2) == str:
            x2 = literal_eval(year1)
            y2 = r2['citations by year'][0]
            y2 = literal_eval(y2)
            if len(x2) < len(y2):
                x2 = make_length_equal(x2,y2)
            elif len(y2) < len(x2):
                y2 = make_length_equal(y2,x2)
        if not x1 or not x2:
            st.text('Data not available')
        if x1:
            with col16:
                color = 'salmon'
                fig = citations_by_year(x1,y1,color)
                st.plotly_chart(fig)
        if x2:
            with col18:
                color = 'crimson'
                fig = citations_by_year(x2,y2,color)
                st.plotly_chart(fig)

        ############### h index ############
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Average h-index Percentile in SCSE</h5>", unsafe_allow_html=True)
        col13,col14,col15 = st.columns([3,2,4])
        with col13:
            st.plotly_chart(fig1)
        with col15:
            st.plotly_chart(fig3)

        ########### Education #############
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Education</h5>", unsafe_allow_html=True)
        col4,col5,col6 = st.columns([3,2,4])

        with col4:
            ed1 = r1
            ed1 = ed1['education'][0]
            if type(ed1) == str:
                st.text(ed1.capitalize())
            else: st.text('-')
        with col6:
            ed2 = r2
            ed2 = ed2['education'][0]
            if type(ed2) == str:
                st.text(ed2.capitalize())
            else: st.text('-')

        ########### research expertise #############
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Research Expertise</h5>", unsafe_allow_html=True)
        col7,col8,col9 = st.columns([3,2,4])

        interest1 = []
        interest2 = []
        with col7:
            ed1 = r1
            ed1 = ed1['keywords'][0]
            if type(ed1) == str:
                ed1 = literal_eval(ed1)
                if len(ed1)>0:
                    for ed in ed1:
                        interest1.append(ed)
                        st.text(ed)
        with col9:
            ed2 = r2
            ed2 = ed2['keywords'][0]
            if type(ed2) == str:
                ed2 = literal_eval(ed2)
                if len(ed2)>0:
                    for ed in ed2:
                        interest2.append(ed)
                        st.text(ed)
        st.markdown('***')

        st.markdown("<h5 style='text-align: center; color: steelblue;'>Common Interests</h5>", unsafe_allow_html=True)
        intersection = list(set(interest1) & set(interest2))
        if len(intersection)>0:
            for key in intersection:
                html_str1 = f""" <h6 style='text-align: center; color: black;'>{key}</h6> """
                st.markdown(html_str1, unsafe_allow_html=True)
        else: 
            key = '-NA-'
            html_str1 = f""" <h6 style='text-align: center; color: black;'>{key}</h6> """
            st.markdown(html_str1, unsafe_allow_html=True)
        st.markdown('***')

        
            
        ############# co -authors ################
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Top 5 Co-Authors</h5>", unsafe_allow_html=True)
        exp = st.expander('Click to view top co-authors')
        co1 = r1['Co Authors'][0]
        names1,unis1,urls1 = get_co_authors_info(co1)
        co2 = r2['Co Authors'][0]
        names2,unis2,urls2 = get_co_authors_info(co2)
        with exp:
            col7,col8,col9 = st.columns([3,2,4])
            with col7:    
                if len(names1) > 0:
                    for i in range(len(names1)):
                        author = 'Co-author '+str(i+1)+':'
                        html_str1 = f""" <h6 style='text-align: center; color: black;'>{author}</h6> """
                        st.markdown(html_str1, unsafe_allow_html=True)
                        st.write(names1[i])
                        try:
                            st.write(unis1[i])
                        except:
                            pass
                        try:
                            st.write("[Google Scholar Link]("+(urls1[i])+")")
                        except: pass
                else: st.write('Not Available')
            with col9:
                if len(names2) > 0:
                    for i in range(len(names2)):
                        author = 'Co-author '+str(i+1)+':'
                        html_str1 = f""" <h6 style='text-align: center; color: black;'>{author}</h6> """
                        st.markdown(html_str1, unsafe_allow_html=True)
                        st.write(names2[i])
                        try:
                            st.write(unis2[i])
                        except: pass
                        try:
                            st.write("[Google Scholar Link]("+(urls2[i])+")")
                        except: pass
                else: st.write('Not Available')

        st.markdown('***')
        ############### common co-authors ###########
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Common Connections</h5>", unsafe_allow_html=True)
        co11 = r1['dblp coauthors'][0]
        r1_co = names1
        r1_co = [x.lower() for x in r1_co]
        co22 = r2['dblp coauthors'][0]
        r2_co = names2
        r2_co = [x.lower() for x in r2_co]
        if type(co11) == str:
            col1 = literal_eval(co11)
            col1 = [x.lower() for x in col1]
            r1_co += col1
        if type(co22) == str:
            co22 = literal_eval(co22)
            co22 = [x.lower() for x in co22]
            r2_co += co22
        ## intersection of names:
        commons = list(set(r1_co) & set(r2_co))
        if len(commons)>0:
            for i in range(len(commons)):
                author = commons[i].capitalize()
                html_str1 = f""" <h6 style='text-align: center; color: black;'>{author}</h6> """
                
        else: 
            author = '-NA-'
            html_str1 = f""" <h6 style='text-align: center; color: black;'>{author}</h6> """
        st.markdown(html_str1, unsafe_allow_html=True)  

        ########### venues #############
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Top 5 Venues</h5>", unsafe_allow_html=True)
        exp1 = st.expander('Click to view top co-authors')

        with exp1:
            col7,col8,col9 = st.columns([3,2,4])

            interest1 = []
            interest2 = []
            with col7:
                ed1 = r1
                ed1 = ed1['dblp venues'][0]
                if type(ed1) == str:
                    ed1 = literal_eval(ed1)
                    if len(ed1)>0:
                        for ed in ed1:
                            interest1.append(ed)
                            st.text(ed)
            with col9:
                ed2 = r2
                ed2 = ed2['dblp venues'][0]
                if type(ed2) == str:
                    ed2 = literal_eval(ed2)
                    if len(ed2)>0:
                        for ed in ed2:
                            interest2.append(ed)
                            st.text(ed)
            st.markdown('***')


    with tab4:

        col1,col2,col3 = st.columns([3,2,4])
        
        with col1:
            st.text('SCSE Researcher')
            r_1 = st.selectbox(" ", scse_profs,key='r1 within ntu')
            progress_bar = st.progress(0)
            status_text = st.empty()
            #display dept:
            r1 = scse[scse['Names'] == r_1].reset_index(drop=True)
            st.write(r1['sch'][0])
        with col3:
            st.text('Researcher from Other Faculty')
            temp_profs_list =  [value for value in all_ntu_profs if value not in scse_profs]
            r_2 = st.selectbox(" ", temp_profs_list,key='r2 within ntu')
            progress_bar2 = st.progress(0)
            status_text2 = st.empty()
             #display dept:
            r2 = df[df['Names'] == r_2].reset_index(drop=True)
            st.write(r2['sch'][0])

        
        ########### career length ###########
        #for researcher 1
        year1 = scse[scse['Names'] == r_1].reset_index(drop=True)
        year1 = year1['years'][0]
        if type(year1) == str:
            x1 = literal_eval(year1)
            for i in range(x1[-1]-x1[0]):
                progress_bar.progress(i + 1)
                status_text.text(
                    'Career Length: ' + str(i+1) + ' years')
                time.sleep(0.1)
        else: status_text.text(
                    'Career Length: Not Available')
        #for researcher 2
        year2 = df[df['Names'] == r_2].reset_index(drop=True)
        year2 = year2['years'][0]
        if type(year2) == str:
            x2 = literal_eval(year2)
            for i in range(x2[-1]-x2[0]):
                # Update progress bar.
                progress_bar2.progress(i + 1)
                status_text2.text(
                    'Career Length: ' + str(i+1) + ' years')
                time.sleep(0.1)
        else: status_text2.text(
                    'Career Length: Not Available')
        
        ################### personal details:
        rs1 = scse[scse['Names'] == r_1].reset_index(drop=True)
        rs2 = df[df['Names'] == r_2].reset_index(drop=True)
        col1,col2,col3 = st.columns([3,2,4])
        
        with col1:
            email,dr_url,website,ggsch,orc,disp = get_personal_info(rs1)
            st.write(':email: '+"[Email]("+(email)+")" +'\t\t'+':link: '+"[DR-NTU]("+(dr_url)+")")
            if type(website) == str:
                st.write(':link: '+"[Personal Website]("+(website)+")")
            if type(ggsch) == str:
                st.write(':male-student:'+"[Google Scholar]("+(ggsch)+")")
            if type(orc) == str:
                st.write(':link: '+"[ORC ID]("+(orc)+")")
            st.write(':school: '+"Schools: "+(disp))
        with col3:
            email,dr_url,website,ggsch,orc,disp = get_personal_info(rs2)
            st.write(':email: '+"[Email]("+(email)+")" +'\t\t'+':link: '+"[DR-NTU]("+(dr_url)+")")
            if type(website) == str:
                st.write(':link: '+"[Personal Website]("+(website)+")")
            if type(ggsch) == str:
                st.write(':male-student:'+"[Google Scholar]("+(ggsch)+")")
            if type(orc) == str:
                st.write(':link: '+"[ORC ID]("+(orc)+")")
            st.write(':school: '+"Schools: "+(disp))

        ######## tot no of lead papers ##########
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Papers as Lead Author (for past 10 recent papers)</h5>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns([3,2,4])
        with col1:
            pubs = rs1['count of lead papers'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = int(pubs)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)
        
        with col3:
            pubs = rs2['count of lead papers'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = int(pubs)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)


        ############## percentile
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Average h-index Percentile within Faculty (%)</h5>", unsafe_allow_html=True)
        col10,col11,col12 = st.columns([3,2,4])

        with col10:
            pubs = rs1['h_index percentile'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = round(pubs,2)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)
            
        with col12:
            pubs = rs2['h_index percentile'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = round(pubs,2)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)

        ############## h index, citation rate, # publications
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Citation Growth Rate (% over past 3 years)</h5>", unsafe_allow_html=True)
        col10,col11,col12 = st.columns([3,2,4])

        with col10:
            mean_h_index_over_years = round(rs1['h_index percentile'][0],2)
            mean_citation_growth_rate = rs1['citation growth rate'][0]
            fig1,fig2 = plot_gauge_charts(mean_h_index_over_years,mean_citation_growth_rate)
            
            st.plotly_chart(fig2)
            
        with col12:
            mean_h_index_over_years = round(rs2['h_index percentile'][0],2)
            mean_citation_growth_rate = rs2['citation growth rate'][0]
            fig3,fig4 = plot_gauge_charts(mean_h_index_over_years,mean_citation_growth_rate)
            st.plotly_chart(fig4)
        
        
        ########### Education #############
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Education</h5>", unsafe_allow_html=True)
        col4,col5,col6 = st.columns([3,2,4])

        with col4:
            ed1 = rs1
            ed1 = ed1['education'][0]
            if type(ed1) == str:
                st.text(ed1.capitalize())
            else: st.text('-')
        with col6:
            ed2 = rs2
            ed2 = ed2['education'][0]
            if type(ed2) == str:
                st.text(ed2.capitalize())
            else: st.text('-')

        ########### research expertise #############
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Research Expertise</h5>", unsafe_allow_html=True)
        col7,col8,col9 = st.columns([3,2,4])

        interest1 = []
        interest2 = []
        with col7:
            ed1 = rs1
            ed1 = ed1['keywords'][0]
            if type(ed1) == str:
                ed1 = literal_eval(ed1)
                if len(ed1)>0:
                    for ed in ed1:
                        interest1.append(ed)
                        st.text(ed)
        with col9:
            ed2 = rs2
            ed2 = ed2['keywords'][0]
            if type(ed2) == str:
                ed2 = literal_eval(ed2)
                if len(ed2)>0:
                    for ed in ed2:
                        interest2.append(ed)
                        st.text(ed)
        st.markdown('***')

        st.markdown("<h5 style='text-align: center; color: steelblue;'>Common Interests</h5>", unsafe_allow_html=True)
        intersection = list(set(interest1) & set(interest2))
        if len(intersection)>0:
            for key in intersection:
                html_str1 = f""" <h6 style='text-align: center; color: black;'>{key}</h6> """
                st.markdown(html_str1, unsafe_allow_html=True)
        else: 
            key = '-NA-'
            html_str1 = f""" <h6 style='text-align: center; color: black;'>{key}</h6> """
            st.markdown(html_str1, unsafe_allow_html=True)
        st.markdown('***')

        
            
        ############# co -authors ################
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Top 5 Co-Authors</h5>", unsafe_allow_html=True)
        exp = st.expander('Click to view top co-authors')
        co1 = rs1['Co Authors'][0]
        names1,unis1,urls1 = get_co_authors_info(co1)
        co2 = rs2['Co Authors'][0]
        names2,unis2,urls2 = get_co_authors_info(co2)
        with exp:
            col7,col8,col9 = st.columns([3,2,4])
            with col7:    
                if len(names1) > 0:
                    for i in range(len(names1)):
                        author = 'Co-author '+str(i+1)+':'
                        html_str1 = f""" <h6 style='text-align: center; color: black;'>{author}</h6> """
                        st.markdown(html_str1, unsafe_allow_html=True)
                        st.write(names1[i])
                        try:
                            st.write(unis1[i])
                        except:
                            pass
                        try:
                            st.write("[Google Scholar Link]("+(urls1[i])+")")
                        except: pass
                else: st.write('Not Available')
            with col9:
                if len(names2) > 0:
                    for i in range(len(names2)):
                        author = 'Co-author '+str(i+1)+':'
                        html_str1 = f""" <h6 style='text-align: center; color: black;'>{author}</h6> """
                        st.markdown(html_str1, unsafe_allow_html=True)
                        st.write(names2[i])
                        try:
                            st.write(unis2[i])
                        except: pass
                        try:
                            st.write("[Google Scholar Link]("+(urls2[i])+")")
                        except: pass
                else: st.write('Not Available')

        st.markdown('***')
        ############### common co-authors ###########
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Common Connections</h5>", unsafe_allow_html=True)
        co11 = rs1['dblp coauthors'][0]
        r1_co = names1
        co22 = rs2['dblp coauthors'][0]
        r2_co = names2
        if type(co11) == str:
            r1_co += literal_eval(co11)
        if type(co22) == str:
            r2_co += literal_eval(co22)
        ## intersection of names:
        commons = list(set(r1_co) & set(r2_co))
        if len(commons)>0:
            for i in range(len(commons)):
                author = commons[i].capitalize()
                html_str1 = f""" <h6 style='text-align: center; color: black;'>{author}</h6> """
                
        else: 
            author = '-NA-'
            html_str1 = f""" <h6 style='text-align: center; color: black;'>{author}</h6> """
        st.markdown(html_str1, unsafe_allow_html=True)  

    with tab7:
        col1,col2,col3 = st.columns([3,2,4])
        nus_profs = nus['Name'].to_list()
        with col1:
            st.text('SCSE NTU Researcher')
            r_1 = st.selectbox(" ", scse_profs,key='r1 ntu scse')
            progress_bar = st.progress(0)
            status_text = st.empty()
            #display dept:
            r1 = scse[scse['Names'] == r_1].reset_index(drop=True)
        with col3:
            st.text('NUS Computing Researcher')
            r_2 = st.selectbox(" ", nus_profs,key='r2 nus')
            progress_bar2 = st.progress(0)
            status_text2 = st.empty()
             #display dept:
            r2 = nus[nus['Name'] == r_2].reset_index(drop=True)
        

        ########### career length ###########
        #for researcher 1
        year1 = scse[scse['Names'] == r_1].reset_index(drop=True)
        year1 = year1['years'][0]
        if type(year1) == str:
            x1 = literal_eval(year1)
            for i in range(x1[-1]-x1[0]):
                progress_bar.progress(i + 1)
                status_text.text(
                    'Career Length: ' + str(i+1) + ' years')
                time.sleep(0.1)
        else: status_text.text(
                    'Career Length: Not Available')
        #for researcher 2
        year2 = nus[nus['Name'] == r_2].reset_index(drop=True)
        year2 = year2['years'][0]
        if type(year2) == str:
            x2 = literal_eval(year2)
            for i in range(x2[-1]-x2[0]):
                # Update progress bar.
                progress_bar2.progress(i + 1)
                status_text2.text(
                    'Career Length: ' + str(i+1) + ' years')
                time.sleep(0.1)
        else: status_text2.text(
                    'Career Length: Not Available')
        
        ################### personal details:
        rs1 = scse[scse['Names'] == r_1].reset_index(drop=True)
        rs2 = nus[nus['Name'] == r_2].reset_index(drop=True)
        col1,col2,col3 = st.columns([3,2,4])
        r1 = rs1
        r2 = rs2
        with col1:
            email,dr_url,website,ggsch,orc,disp = get_personal_info(rs1)
            st.write(':email: '+"[Email]("+(email)+")" +'\t\t'+':link: '+"[DR-NTU]("+(dr_url)+")")
            if type(website) == str:
                st.write(':link: '+"[Personal Website]("+(website)+")")
            if type(ggsch) == str:
                st.write(':male-student:'+"[Google Scholar]("+(ggsch)+")")
            if type(orc) == str:
                st.write(':link: '+"[ORC ID]("+(orc)+")")
            st.write(':school: '+"Designation: "+(disp))
        with col3:
            titles = r2['Designation'][0]
            disp = ''
            if type(titles) == str:
                disp = titles
            ggsch = r2['Google Scholar URL'][0]
            if type(ggsch) == str:
                st.write(':male-student:'+"[Google Scholar]("+(ggsch)+")")
            st.write(':school: '+"Designation: "+(disp))

        ######## tot no of lead papers ##########
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Papers as Lead Author (for past 10 recent papers)</h5>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns([3,2,4])
        with col1:
            pubs = rs1['count of lead papers'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = int(pubs)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)
        
        with col3:
            pubs = rs2['count of lead papers'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = int(pubs)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)

        ##############
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Average Number of Publications per year</h5>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns([3,2,4])
        with col1:
            mean_publications_over_years = r1['Total Publications'][0]
            mean_citation_growth_rate = r1['citation growth rate'][0]
            fig1,fig2 = plot_gauge_charts(mean_publications_over_years,mean_citation_growth_rate)
            pubs = r1['Total Publications'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = int(pubs)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)
            # st.plotly_chart(fig1)
        with col3:
            mean_publications_over_years = r2['Total Publications'][0]
            mean_citation_growth_rate = r2['citation growth rate'][0]
            fig3,fig4 = plot_gauge_charts(mean_publications_over_years,mean_citation_growth_rate)
            # with coln:
            # st.plotly_chart(fig3)
            pubs = r2['Total Publications'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = int(pubs)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)
        st.markdown('***')
        #############

        st.markdown("<h5 style='text-align: center; color: steelblue;'>Mean Citation Growth Rate (% over past 3 years)</h5>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns([3,2,4])
        with col1:
            st.plotly_chart(fig2)
        with col3:
            st.plotly_chart(fig4)

        ############## percentile
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Average h-index (calibrated by years)</h5>", unsafe_allow_html=True)
        col10,col11,col12 = st.columns([3,2,4])

        with col10:
            pubs = rs1['h_index/years'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = round(pubs,2)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)
            
        with col12:
            pubs = rs2['h_index/years'][0]
            if math.isnan(pubs):
                pubs = 'Not Available'
            else: pubs = round(pubs,2)
            html_str1 = f""" <h4 style='text-align: center; color: crimson;'>{pubs}</h4> """
            st.markdown(html_str1, unsafe_allow_html=True)

    
        ###############
        #### citations per year
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Citations by Year</h5>", unsafe_allow_html=True)
        col16,col17,col18 = st.columns([3,2,4])
        year1 = r1['years'][0]
        x1 = x2 = None
        if type(year1) == str:
            x1 = literal_eval(year1)
            y1 = r1['citations by year'][0]
            y1 = literal_eval(y1)
            if len(x1) < len(y1):
                x1 = make_length_equal(x1,y1)
            elif len(y1) < len(x1):
                y1 = make_length_equal(y1,x1)
                

        year2 = r2['years'][0]
        if type(year2) == str:
            x2 = literal_eval(year1)
            y2 = r2['citations by year'][0]
            y2 = literal_eval(y2)
            if len(x2) < len(y2):
                x2 = make_length_equal(x2,y2)
            elif len(y2) < len(x2):
                y2 = make_length_equal(y2,x2)
        if not x1 or not x2:
            st.text('Data not available')
        if x1:
            with col16:
                color = 'salmon'
                fig = citations_by_year(x1,y1,color)
                st.plotly_chart(fig)
        if x2:
            with col18:
                color = 'crimson'
                fig = citations_by_year(x2,y2,color)
                st.plotly_chart(fig)

        ############### h index ############
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Average h-index Percentile within Faculty (%)</h5>", unsafe_allow_html=True)
        col13,col14,col15 = st.columns([3,2,4])
        with col13:
            st.plotly_chart(fig1)
        with col15:
            st.plotly_chart(fig3)

        ########### Education #############
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Education</h5>", unsafe_allow_html=True)
        col4,col5,col6 = st.columns([3,2,4])

        with col4:
            ed1 = r1
            ed1 = ed1['education'][0]
            if type(ed1) == str:
                st.text(ed1.capitalize())
            else: st.text('-')
        with col6:
            ed2 = r2
            ed2 = ed2['Education'][0]
            if type(ed2) == str:
                st.text(ed2.capitalize())
            else: st.text('-')

        ########### research expertise #############
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Research Expertise</h5>", unsafe_allow_html=True)
        col7,col8,col9 = st.columns([3,2,4])

        interest1 = []
        interest2 = []
        with col7:
            ed1 = r1
            ed1 = ed1['keywords'][0]
            if type(ed1) == str:
                ed1 = literal_eval(ed1)
                if len(ed1)>0:
                    for ed in ed1:
                        interest1.append(ed)
                        st.text(ed)
        with col9:
            ed2 = r2
            ed2 = ed2['Research Interests'][0]
            if type(ed2) == str:
                ed2 = ed2.split(',')
                if len(ed2)>0:
                    for ed in ed2:
                        interest2.append(ed)
                        st.text(ed)
        st.markdown('***')

        st.markdown("<h5 style='text-align: center; color: steelblue;'>Common Interests</h5>", unsafe_allow_html=True)
        intersection = list(set(interest1) & set(interest2))
        if len(intersection)>0:
            for key in intersection:
                html_str1 = f""" <h6 style='text-align: center; color: black;'>{key}</h6> """
                st.markdown(html_str1, unsafe_allow_html=True)
        else: 
            key = '-NA-'
            html_str1 = f""" <h6 style='text-align: center; color: black;'>{key}</h6> """
            st.markdown(html_str1, unsafe_allow_html=True)
        st.markdown('***')

        
            
        ############# co -authors ################
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Top 5 Co-Authors</h5>", unsafe_allow_html=True)
        exp = st.expander('Click to view top co-authors')
        co1 = r1['Co Authors'][0]
        names1,unis1,urls1 = get_co_authors_info(co1)
        co2 = r2['Co Authors'][0]
        names2,unis2,urls2 = get_co_authors_info(co2)
        with exp:
            col7,col8,col9 = st.columns([3,2,4])
            with col7:    
                if len(names1) > 0:
                    for i in range(len(names1)):
                        author = 'Co-author '+str(i+1)+':'
                        html_str1 = f""" <h6 style='text-align: center; color: black;'>{author}</h6> """
                        st.markdown(html_str1, unsafe_allow_html=True)
                        st.write(names1[i])
                        try:
                            st.write(unis1[i])
                        except:
                            pass
                        try:
                            st.write("[Google Scholar Link]("+(urls1[i])+")")
                        except: pass
                else: st.write('Not Available')
            with col9:
                if len(names2) > 0:
                    for i in range(len(names2)):
                        author = 'Co-author '+str(i+1)+':'
                        html_str1 = f""" <h6 style='text-align: center; color: black;'>{author}</h6> """
                        st.markdown(html_str1, unsafe_allow_html=True)
                        st.write(names2[i])
                        try:
                            st.write(unis2[i])
                        except: pass
                        try:
                            st.write("[Google Scholar Link]("+(urls2[i])+")")
                        except: pass
                else: st.write('Not Available')

        st.markdown('***')
        ############### common co-authors ###########
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Common Connections</h5>", unsafe_allow_html=True)
        co11 = r1['dblp coauthors'][0]
        r1_co = names1
        r1_co = [x.lower() for x in r1_co]
        co22 = r2['dblp coauthors'][0]
        r2_co = names2
        r2_co = [x.lower() for x in r2_co]
        if type(co11) == str:
            col1 = literal_eval(co11)
            col1 = [x.lower() for x in col1]
            r1_co += col1
        if type(co22) == str:
            co22 = literal_eval(co22)
            co22 = [x.lower() for x in co22]
            r2_co += co22
        ## intersection of names:
        commons = list(set(r1_co) & set(r2_co))
        if len(commons)>0:
            for i in range(len(commons)):
                author = commons[i]
                html_str1 = f""" <h6 style='text-align: center; color: black;'>{author}</h6> """
                
        else: 
            author = '-NA-'
            html_str1 = f""" <h6 style='text-align: center; color: black;'>{author}</h6> """
        st.markdown(html_str1, unsafe_allow_html=True)  

        ########### venues #############
        st.markdown('***')
        st.markdown("<h5 style='text-align: center; color: steelblue;'>Top 5 Venues</h5>", unsafe_allow_html=True)
        exp1 = st.expander('Click to view top venues')
        
        with exp1:
            col7,col8,col9 = st.columns([3,2,4])
            interest1 = []
            interest2 = []
            with col7:
                ed1 = r1
                ed1 = ed1['dblp venues'][0]
                if type(ed1) == str:
                    ed1 = literal_eval(ed1)
                    if len(ed1)>0:
                        for ed in ed1:
                            interest1.append(ed)
                            st.text(ed)
            with col9:
                ed2 = r2
                ed2 = ed2['dblp venues'][0]
                if type(ed2) == str:
                    ed2 = literal_eval(ed2)
                    if len(ed2)>0:
                        for ed in ed2:
                            interest2.append(ed)
                            st.text(ed)
        st.markdown('***')
######################################################################################################


page_names_to_funcs = {
    "SCSE Statistics": main_page,
    "Compare Researchers": page2,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
