import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))



st.set_page_config(page_title="🏏Ipl Analysis Dashboard",layout="wide")

#load data

file_path = os.path.join(os.path.dirname(__file__), "ipl_dashboard_final.csv")
matches = pd.read_csv(file_path)
matches['date']=pd.to_datetime(matches['date'])
matches['season']=matches['season'].astype(str)

# create sidebar filters 
st.sidebar.header('🔍 Filters')
st.sidebar.markdown("Select filters to explore IPL data")
season_list=['All']+sorted(matches['season'].unique().tolist())
selected_season=st.sidebar.selectbox('Select Season',season_list)
team_list=['All']+sorted(pd.concat([matches['team1'],matches['team2']]).unique().tolist())  
selected_team=st.sidebar.selectbox('Select Team',team_list)
city_list=['All']+sorted(matches['city'].unique().tolist())
selected_city=st.sidebar.selectbox('Select City',city_list)


#filter apply
filtered=matches.copy()
if selected_season!='All':
    filtered=filtered[filtered['season']==selected_season]
if selected_team!='All':
    filtered=filtered[(filtered['team1']==selected_team)|(filtered['team2']==selected_team)]
if selected_city!='All':
    filtered=filtered[filtered['city']==selected_city]
    
# title
st.markdown(
    "<h1 style='color:royalblue;'>🏏 IPL Analysis Dashboard</h1>",
    unsafe_allow_html=True
    )
st.markdown("---")

### key metrics  ###
st.subheader('📊 Key Performance Indicators')
col1 , col2 , col3 , col4 , col5 = st.columns(5)
with col1:
    st.metric("🏏 Total Matches",filtered.shape[0])
with col2:
    st.metric("👥 Total Teams",filtered['team1'].nunique())
with col3:
    st.metric("📍Total City",filtered['city'].nunique())
with col4:
    st.metric("📅 Total Seasons",filtered['season'].nunique())
with col5:
    st.metric("🎯Toss Win Rate",f"{filtered['toss_won_match'].mean()*100:.1f}%")
st.markdown('----')

tab1, tab2,tab3 = st.tabs(["🔍 Filtered Analysis","📊 Overall Analysis","🔮 Match Winner Prediction"])
with tab1:
    st.subheader("🎲 Toss Analysis")
    col1,col2=st.columns(2)

    with col1:
        fig,ax=plt.subplots()
        filtered['toss_decision'].value_counts().plot(kind='pie',ax=ax,autopct='%1.1f%%')
        ax.set_title("Batting vs Fielding Decisions After Toss Victory",color='grey')
        st.pyplot(fig)
    with col2:
        fig,ax=plt.subplots()
        filtered.groupby('toss_decision')['toss_won_match'].mean().plot(kind='bar',color=['grey','green'],ax=ax)
        ax.set_ylabel('Win Rate')
        ax.set_title("Which Decision Leads to More Wins?",color='grey')
       
        st.pyplot(fig)

    st.markdown("---")

    st.subheader("🏆 Team Analysis")
    col1,col2=st.columns(2)

    with col1:
        fig,ax=plt.subplots()
        filtered['winner'].value_counts().plot(kind='bar', ax=ax)
        plt.xticks(rotation=90)
        ax.set_title("Total Matches Won by Each Team", color='grey')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        filtered.groupby("toss_winner")['toss_won_match'].mean().sort_values(ascending=False).plot(kind='bar', color='grey', ax=ax)
        plt.xticks(rotation=90)
        ax.set_title("Teams Win Rate After Winning Toss",color='grey')
        st.pyplot(fig)
        
    st.markdown("---")

    st.subheader("📈 Match Analysis")
    col1,col2=st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        filtered['result_margin'].plot(kind='hist', color='purple', bins=20,edgecolor='black', ax=ax)
        ax.set_title("Analysis of Match Winning Margins",color='grey')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        filtered.groupby('toss_won_match')['result_margin'].mean().plot(kind='bar', color='green', ax=ax)
        plt.xticks(rotation=90)
        ax.set_xticklabels(['Lost After Winning Toss','Win After Winning Toss'])
        ax.set_title("Impact of Toss on Winning Margin",color='grey')
        st.pyplot(fig)
    st.markdown('---')
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(filtered['target_runs'], filtered['result_margin'], color='red')
    ax.set_xlabel('Target Runs')
    ax.set_ylabel('Result Margin') 
    ax.set_title("Target Runs vs Match Winning Margin",color='grey')
    st.pyplot(fig)

    st.markdown("---")  
    st.subheader("🌤️ Weather Analysis")
    st.caption("⚠️ Note: Weather data represents match city's weather conditions. Actual stadium conditions may slightly vary.")
    
    col1,col2=st.columns(2)

    with col1:
        st.metric("🌡 Avg Temperature",
              round(filtered['temperature'].mean(),1))

    with col2:
        st.metric("🌧 Avg Rainfall",
              round(filtered['rainfall'].mean(),2))
    
    col1,col2=st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.boxplot(y=filtered['temperature'], ax=ax, color='purple')
        ax.set_title("Temperature Distribution",color='grey')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(y=filtered['rainfall'], ax=ax, color='blue')
        ax.set_title("Rainfall Distribution",color='grey')
        st.pyplot(fig)

    st.markdown("---")  
    st.subheader("⭐ Player Of The Match Analysis")
    fig,ax=plt.subplots()
    top_players=filtered['player_of_match'].value_counts().head(10).sort_values(ascending=False)
    sns.barplot( x=top_players.values,y=top_players.index,palette='viridis',ax=ax)
    for i, v in enumerate(top_players.values):
        ax.text(v + 0.2, i, str(v))
    ax.set_title('Top 10 Players by Player Of The Match Award',color='grey',fontsize=9)
    ax.set_ylabel('Players')
    ax.set_xlabel('No. Of Awards')
    st.pyplot(fig)    

    st.markdown("***")
    st.subheader("🏟️Venue Analysis")
    fig,ax=plt.subplots(figsize=(24,12))
    venuess=filtered.groupby('venue')['result_margin'].mean().sort_values()
    sns.barplot(x=venuess.values,y=venuess.index,palette='flare',ax=ax)
    ax.set_title("Which Venues Produce One-Sided Matches?",fontsize=16)
    st.pyplot(fig)

    st.markdown("---")
    with st.expander("🏟️ View Venue Heatmap (Click to expand)"):
        fig,ax=plt.subplots(figsize=(10,20))
        venue_heatmap=filtered.pivot_table( index="venue",columns='toss_decision',values="toss_won_match",aggfunc="mean")
        sns.heatmap(venue_heatmap,annot=True,cmap="plasma",ax=ax)
        plt.tight_layout()
        ax.set_title(" Best Toss Strategy by Venue",fontsize=12,color='grey')
        st.pyplot(fig)

    st.markdown('---')
    st.subheader("⚔️  Team Head to Head Analysis")
    Team_list=sorted(pd.concat([matches['team1'],matches['team2']]).unique().tolist())
    col1,col2=st.columns(2)
    with col1:
        team1=st.selectbox(" Select Team1",Team_list,key="team1")
    with col2:
        team2=st.selectbox("Select Team2",Team_list,key="team2")
    if team1==team2:
        st.warning("⚠️ Please select two different teams.")
    else:
        h2h=matches[
            ((matches['team1']==team1) &(matches['team2']==team2))|
            ((matches['team2']==team1) & (matches['team1']==team2))
        ]
        total=h2h.shape[0]
        team1_wins=h2h[h2h['winner']==team1].shape[0]
        team2_wins=h2h[h2h['winner']==team2].shape[0]
        col1,col2,col3=st.columns(3)
        with col1:
            st.metric("Total Matches",total)
        with col2:
            st.metric(f"{team1} Wins",team1_wins)
        with col3:
            st.metric(f"{team2} Wins",team2_wins)

        fig,ax=plt.subplots()
        wedges,texts,autotexts= ax.pie([team1_wins, team2_wins],autopct="%1.1f%%",startangle=90)   
        ax.legend(wedges,[team1, team2],title="Teams",fontsize=5,title_fontsize=7)
        st.pyplot(fig)

with tab2:
       
    st.subheader("📈 Season-wise Insights")
    col1,col2=st.columns(2)
    with col1:
        fig,ax=plt.subplots()
        matches.groupby('season')['team1'].count().plot(kind='bar',color='green',ax=ax)
        plt.xticks(rotation=90)
        ax.set_title("Season-wise Matches Played",color='grey')
        st.pyplot(fig) 

    with col2:
        fig,ax=plt.subplots()
        matches.groupby('season')['toss_won_match'].mean().plot(kind='bar',color='crimson',ax=ax)
        plt.xticks(rotation=90)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x*100:.0f}%'))
        ax.set_title("Season-wise Toss Win %",color='grey')
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🎲 Toss Analysis")
    col1,col2=st.columns(2)

    with col1:
        fig,ax=plt.subplots(figsize=(5,4))
        matches.groupby('season')['toss_won_match'].mean().plot(kind='bar',color='crimson',ax=ax)
        plt.xticks(rotation=90)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x*100:.0f}%'))
        ax.set_title("Season-wise Toss Win %",color='grey')
        st.pyplot(fig) 

    with col2:
        venue_toss = matches.groupby('venue')['toss_won_match'].mean()
        venue_count = matches.groupby('venue')['toss_won_match'].count()
        popular_venues = venue_count[venue_count > 10]
        venue_toss = venue_toss[popular_venues.index].sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5,4))
        venue_toss.plot(kind='line', marker='*',color='crimson', ax=ax)
        plt.xticks(rotation=90)
        ax.set_title("Top 10 Venues Where Toss Matters Most",color='grey')
        st.pyplot(fig)
        
    st.markdown("---")

        
    st.subheader("🏆 Team Analysis")
      
    heatmap_data = matches.pivot_table(index='toss_winner', columns='toss_decision', values='toss_won_match', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(14,8))
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', ax=ax)
    ax.set_title("Team Win Rate: Bat vs Field (Heatmap)",color='grey',fontsize=18)
    st.pyplot(fig)
    st.markdown("---")
    st.subheader("📈 Match Analysis")
    
    fig, ax = plt.subplots()
    season_matches = matches.groupby('season')['team1'].count()
    season_matches.plot(kind='line', color='green', ax=ax)
    for i, y in enumerate(season_matches.values):
        ax.text(i, y, str(y), fontsize=7)
    plt.xticks(rotation=90)
    ax.set_title("Number of Matches Played per Season",color='grey',fontsize=9)
    st.pyplot(fig)

    
    st.markdown('---')
    
    st.subheader("🌤️ Weather Analysis")
   
    fig, ax = plt.subplots(figsize=(14,5))
    matches.groupby('city')['temperature'].mean().plot(kind='bar', color='green', ax=ax)
    plt.xticks(rotation=90)
    ax.set_title("City-wise Average Temperature",color='grey',fontsize=16)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🏟️Venue Analysis")
    fig,ax=plt.subplots(figsize=(12,6))
    venues=matches['venue'].value_counts().head(10)
    sns.barplot(x=venues.values,y=venues.index,palette='rocket',ax=ax)
    for i,s in enumerate(venues.values):
        ax.text(s+0.2,i,str(s))
    ax.set_title('Top 10 Venues By Matches Played',color='grey',fontsize=16)
    ax.set_xlabel('No. Of Matches')
    ax.set_ylabel('Venue')
    st.pyplot(fig)

            
with tab3:
    st.markdown("---")
    st.subheader("🔮 Match Winner Prediction")
    st.info("⚠️ Prediction based on historical patterns — 52% accuracy. Cricket is unpredictable!")

    col1, col2 = st.columns(2)
    with col1:
        p_team1 = st.selectbox("Select Team 1", Team_list, key="p1")
    with col2:
        p_team2 = st.selectbox("Select Team 2", Team_list, key="p2")

    filtered_matches=matches[
      ((matches['team1']==p_team1) & (matches['team2']==p_team2)) 
    |
      ((matches['team1']==p_team2) & (matches['team2']==p_team1))
    ]


    col1, col2 = st.columns(2)
    with col1:
        p_toss_winner = st.selectbox("Toss Winner", [p_team1, p_team2], key="ptoss")
    with col2:
        p_toss_decision = st.selectbox("Toss Decision", ['bat', 'field'], key="pdec")

  

    col1, col2 = st.columns(2)
    with col1:
        p_city = st.selectbox("City", sorted(filtered_matches['city'].unique()), key="pcity")
        filtered_city=filtered_matches[filtered_matches['city']==p_city]
    with col2:
        p_venue = st.selectbox("Venue", sorted(filtered_city['venue'].unique()), key="pvenue")

    col1, col2 = st.columns(2)
    with col1:
        p_season = st.number_input("Season",min_value=2008, max_value=2035, value=2028, step=1)
    with col2:
        p_temp = st.number_input("Expected Temperature (°C)", min_value=10.0, max_value=50.0, value=32.0)

    p_rain = st.slider("Expected Rainfall (mm)", 0.0, 30.0, 0.0)

    if st.button("🏆 Predict Winner"):
        try:
            payload = {
                "team1": p_team1,
                "team2": p_team2,
                "venue": p_venue,
                "temperature": float(p_temp),
                "rainfall": float(p_rain),
                "toss_winner": p_toss_winner,
                "toss_decision": p_toss_decision,
                "season": int(p_season),
                "city": p_city
            }

            response = requests.post(
                "https://ipl-match-prediction-0i3k.onrender.com/predict",
                json=payload
            )
            winner = response.json()["predicted_winner"]

            st.success(f"🏆 Predicted Winner: **{winner}**")
            st.balloons()

        except Exception as e:
            st.error(f"Error: {e} — Please check your inputs!")
   




    

st.markdown("---")
st.subheader("📌 Key Insights")
st.write("""
- 🏏 Toss winner wins only **50.8%** matches — largely a myth!
- ⚡ Field decision has higher success rate **(53.9%)** vs Bat **(45.4%)**
- 🌍 2009 IPL (South Africa) had coldest matches — avg **16-24°C**
- 🏆 Mumbai Indians have most overall wins **(144)**
- 🌧️ **21 matches** were rain-affected (DLS method used)
- ⭐ **AB de Villiers** is among the players with the most Player of the Match awards.
- 🌡️ Average IPL match temperature: **34.7°C**
- 🏟️ **Wankhede Stadium** hosted most IPL matches
- 🤝 Head to Head feature added — explore any 2 teams rivalry!
- 🌤️ **Weather API integrated** — real historical weather data for all 1090 matches
""")


