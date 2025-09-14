import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide", page_title="Marketing Intelligence Dashboard", initial_sidebar_state="expanded")


@st.cache_data
def load_data():
    fb = pd.read_csv("Facebook.csv")
    gg = pd.read_csv("Google.csv")
    tt = pd.read_csv("TikTok.csv")
    biz = pd.read_csv("business.csv")

    fb['channel'] = 'Facebook'
    gg['channel'] = 'Google'
    tt['channel'] = 'TikTok'

    for df in (fb, gg, tt, biz):
        df.columns = [c.strip() for c in df.columns]

    # Parse dates
    fb['date'] = pd.to_datetime(fb['date'])
    gg['date'] = pd.to_datetime(gg['date'])
    tt['date'] = pd.to_datetime(tt['date'])
    biz['date'] = pd.to_datetime(biz['date'])

    marketing = pd.concat([fb, gg, tt], ignore_index=True, sort=False)

    # Fill missing numeric values with 0 where appropriate
    numeric_cols = ['impression','clicks','spend','attributed revenue']
    for col in numeric_cols:
        if col in marketing.columns:
            marketing[col] = pd.to_numeric(marketing[col], errors='coerce').fillna(0)

    # Business numeric cleanup
    biz_cols = ['# of orders','# of new orders','new customers','total revenue','gross profit','COGS']
    for col in biz_cols:
        if col in biz.columns:
            biz[col] = pd.to_numeric(biz[col], errors='coerce').fillna(0)

    # Merge marketing aggregated at date-channel level with business by date when needed
    return marketing, biz

def derive_metrics(marketing_df, business_df):
    df = marketing_df.copy()
    # Derive metrics at row-level
    df['CTR'] = np.where(df['impression']>0, df['clicks']/df['impression'], np.nan)
    df['CPC'] = np.where(df['clicks']>0, df['spend']/df['clicks'], np.nan)
    df['CPM'] = np.where(df['impression']>0, df['spend']/df['impression']*1000, np.nan)
    df['ROAS'] = np.where(df['spend']>0, df['attributed revenue']/df['spend'], np.nan)

    # Aggregate marketing by date and channel for dashboard
    daily_channel = df.groupby(['date','channel']).agg({
        'impression':'sum',
        'clicks':'sum',
        'spend':'sum',
        'attributed revenue':'sum'
    }).reset_index()

    daily_channel['CTR'] = np.where(daily_channel['impression']>0, daily_channel['clicks']/daily_channel['impression'], np.nan)
    daily_channel['CPC'] = np.where(daily_channel['clicks']>0, daily_channel['spend']/daily_channel['clicks'], np.nan)
    daily_channel['CPM'] = np.where(daily_channel['impression']>0, daily_channel['spend']/daily_channel['impression']*1000, np.nan)
    daily_channel['ROAS'] = np.where(daily_channel['spend']>0, daily_channel['attributed revenue']/daily_channel['spend'], np.nan)

    # Aggregate marketing across channels by date
    daily_total = daily_channel.groupby('date').agg({
        'impression':'sum',
        'clicks':'sum',
        'spend':'sum',
        'attributed revenue':'sum'
    }).reset_index()

    daily_total['CTR'] = np.where(daily_total['impression']>0, daily_total['clicks']/daily_total['impression'], np.nan)
    daily_total['CPC'] = np.where(daily_total['clicks']>0, daily_total['spend']/daily_total['clicks'], np.nan)
    daily_total['CPM'] = np.where(daily_total['impression']>0, daily_total['spend']/daily_total['impression']*1000, np.nan)
    daily_total['ROAS'] = np.where(daily_total['spend']>0, daily_total['attributed revenue']/daily_total['spend'], np.nan)

    # Merge with business metrics by date
    merged_daily = pd.merge(daily_total, business_df, on='date', how='left')

    # Marketing contribution: attributed revenue / total revenue
    merged_daily['marketing_pct_of_revenue'] = np.where(merged_daily['total revenue']>0, merged_daily['attributed revenue']/merged_daily['total revenue'], np.nan)
    # CAC: spend / new customers (handle division by zero)
    merged_daily['CAC'] = np.where(merged_daily['new customers']>0, merged_daily['spend']/merged_daily['new customers'], np.nan)
    # Profit margin
    merged_daily['profit_margin'] = np.where(merged_daily['total revenue']>0, merged_daily['gross profit']/merged_daily['total revenue'], np.nan)

    return df, daily_channel, daily_total, merged_daily

# --- Load data ---
marketing_raw, business = load_data()
marketing_raw.columns = [c.lower() for c in marketing_raw.columns]
business.columns = [c.lower() for c in business.columns]

marketing, daily_channel, daily_total, merged_daily = derive_metrics(marketing_raw, business)

# Sidebar filters
st.sidebar.header("Filters & Controls")
min_date = marketing['date'].min()
max_date = marketing['date'].max()
date_range = st.sidebar.date_input(f"Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Ensure a valid range is always set
if not date_range or len(date_range) != 2:
    date_range = [min_date, max_date]
    start_date, end_date = min_date, max_date
else:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

channels = st.sidebar.multiselect("Select channels", options=['Facebook','Google','TikTok'], default=['Facebook','Google','TikTok'])
tactics = st.sidebar.multiselect("Select tactics (optional)", options=sorted(marketing['tactic'].dropna().unique()), default=None)
states = st.sidebar.multiselect("Select states (optional)", options=sorted(marketing['state'].dropna().unique()), default=None)
campaigns = st.sidebar.multiselect("Select campaigns (optional)", options=sorted(marketing['campaign'].dropna().unique()), default=None)

# Apply filters to marketing and merged_daily
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mask = (marketing['date']>=start_date) & (marketing['date']<=end_date) & (marketing['channel'].isin(channels))
if tactics:
    mask = mask & (marketing['tactic'].isin(tactics))
if states:
    mask = mask & (marketing['state'].isin(states))
if campaigns:
    mask = mask & (marketing['campaign'].isin(campaigns))

filtered_marketing = marketing[mask].copy()

# Recompute aggregates after filtering
agg_channel = filtered_marketing.groupby(['channel']).agg({
    'impression':'sum','clicks':'sum','spend':'sum','attributed revenue':'sum'
}).reset_index()

agg_channel['CTR'] = np.where(agg_channel['impression']>0, agg_channel['clicks']/agg_channel['impression'], np.nan)
agg_channel['CPC'] = np.where(agg_channel['clicks']>0, agg_channel['spend']/agg_channel['clicks'], np.nan)
agg_channel['CPM'] = np.where(agg_channel['impression']>0, agg_channel['spend']/agg_channel['impression']*1000, np.nan)
agg_channel['ROAS'] = np.where(agg_channel['spend']>0, agg_channel['attributed revenue']/agg_channel['spend'], np.nan)

# Merge filtered daily totals with business for date range
daily_total_filtered = daily_total[(daily_total['date']>=start_date)&(daily_total['date']<=end_date)]
merged_daily_filtered = merged_daily[(merged_daily['date']>=start_date)&(merged_daily['date']<=end_date)].copy()

# Top-level KPIs (computed from filtered view)
total_revenue = merged_daily_filtered['total revenue'].sum(skipna=True)
total_spend = daily_total_filtered['spend'].sum(skipna=True)
total_attributed_revenue = daily_total_filtered['attributed revenue'].sum(skipna=True)
total_orders = merged_daily_filtered['# of orders'].sum(skipna=True)
total_new_customers = merged_daily_filtered['new customers'].sum(skipna=True)
overall_roas = total_attributed_revenue/total_spend if total_spend>0 else np.nan
overall_cac = total_spend/total_new_customers if total_new_customers>0 else np.nan
marketing_pct = total_attributed_revenue/total_revenue if total_revenue>0 else np.nan
avg_profit_margin = merged_daily_filtered['profit_margin'].mean(skipna=True)

# --- Layout ---
st.title("📊 Business Intelligence Dashboard")

# KPI cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Revenue", f"${total_revenue:,.0f}")
kpi2.metric("Total Spend", f"${total_spend:,.0f}", delta=None)
kpi3.metric("Attributed Revenue", f"${total_attributed_revenue:,.0f}")
kpi4.metric("ROAS (Attributed / Spend)", f"{overall_roas:.2f}" if not np.isnan(overall_roas) else "N/A")
kpi5.metric("CAC (Spend / New Customers)", f"${overall_cac:.2f}" if not np.isnan(overall_cac) else "N/A")

st.markdown("---")

# Row: Trend charts
col1, col2 = st.columns([2,1], gap="medium")
with col1:
    st.subheader("Revenue vs Marketing Spend")
    fig = go.Figure()
    # Total revenue line
    fig.add_trace(go.Scatter(x=merged_daily_filtered['date'], y=merged_daily_filtered['total revenue'], mode='lines+markers', name='Total Revenue', hovertemplate='%{y:$,.0f}<extra></extra>'))
    # Spend line (from daily_total_filtered)
    fig.add_trace(go.Scatter(x=daily_total_filtered['date'], y=daily_total_filtered['spend'], mode='lines+markers', name='Total Marketing Spend', hovertemplate='%{y:$,.0f}<extra></extra>'))
    # Attributed revenue line
    fig.add_trace(go.Scatter(x=daily_total_filtered['date'], y=daily_total_filtered['attributed revenue'], mode='lines+markers', name='Attributed Revenue', hovertemplate='%{y:$,.0f}<extra></extra>'))
    fig.update_layout(legend=dict(orientation='h', y=-0.15), height=420, margin=dict(l=10,r=10,t=0,b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Marketing Contribution & Profitability")
    # Extract month for coloring
    merged_daily_filtered['month'] = merged_daily_filtered['date'].dt.strftime('%b %Y')
    fig2 = go.Figure()
    for month, group in merged_daily_filtered.groupby('month'):
        fig2.add_trace(go.Bar(
            x=group['date'],
            y=group['marketing_pct_of_revenue'],
            name=month
        ))
    fig2.update_layout(
        yaxis_tickformat=".0%",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Avg Profit Margin:** " + (f"{avg_profit_margin:.1%}" if not np.isnan(avg_profit_margin) else "N/A"))

st.markdown("---")

# Channel performance section
st.subheader("Channel Performance Summary")
colA, colB = st.columns([1.2,1], gap="medium")
with colA:
    st.dataframe(agg_channel[['channel','impression','clicks','spend','attributed revenue','CTR','CPC','ROAS']].sort_values(by='spend', ascending=False).assign(
        impression=lambda d: d['impression'].map('{:,.0f}'.format),
        clicks=lambda d: d['clicks'].map('{:,.0f}'.format),
        spend=lambda d: d['spend'].map('${:,.0f}'.format),
        attributed_revenue=lambda d: d['attributed revenue'].map('${:,.0f}'.format),
        CTR=lambda d: d['CTR'].map('{:.2%}'.format),
        CPC=lambda d: d['CPC'].map('${:,.2f}'.format),
        ROAS=lambda d: d['ROAS'].map('{:.2f}'.format)
    ))

with colB:
    st.markdown("### Spend vs Attributed Revenue by Channel")
    fig3 = px.bar(agg_channel.melt(id_vars='channel', value_vars=['spend','attributed revenue']), x='channel', y='value', color='variable', barmode='group', height=350)
    fig3.update_yaxes(tickprefix="$")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# Campaign performance (interactive)
st.subheader("Campaign & Tactic Deep Dive")
# Top campaigns by ROAS
campaign_aggs = filtered_marketing.groupby(['campaign','channel']).agg({'impression':'sum','clicks':'sum','spend':'sum','attributed revenue':'sum'}).reset_index()
campaign_aggs['ROAS'] = np.where(campaign_aggs['spend']>0, campaign_aggs['attributed revenue']/campaign_aggs['spend'], np.nan)
campaign_aggs['CAC'] = np.nan  # if we wanted to tie to new customers per campaign we would need attribution mapping

top_n = st.slider("Show top N campaigns by spend", min_value=5, max_value=50, value=10)
top_campaigns = campaign_aggs.sort_values('spend', ascending=False).head(top_n)

col21, col22 = st.columns([2,1])
with col21:
    st.markdown("#### Top campaigns by Spend")
    fig4 = px.bar(top_campaigns.sort_values('spend', ascending=True), x='spend', y='campaign', color='channel', orientation='h', height=420)
    fig4.update_xaxes(tickprefix="$")
    st.plotly_chart(fig4, use_container_width=True)
with col22:
    st.markdown("#### Top campaigns by ROAS")
    top_roas = campaign_aggs.sort_values('ROAS', ascending=False).head(top_n)
    fig5 = px.scatter(top_roas, x='spend', y='ROAS', size='attributed revenue', color='channel', hover_data=['campaign'], height=420)
    fig5.update_xaxes(tickprefix="$")
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# Funnel (Impression -> Clicks -> Orders)
st.subheader("Marketing Funnel (Impressions → Clicks → Orders)")
funnel_impr = filtered_marketing['impression'].sum()
funnel_clicks = filtered_marketing['clicks'].sum()
funnel_orders = merged_daily_filtered['# of orders'].sum()

funnel_df = pd.DataFrame({
    "stage": ["Impressions","Clicks","Orders"],
    "value": [funnel_impr, funnel_clicks, funnel_orders]
})

fig_funnel = px.funnel(funnel_df, x='value', y='stage', height=300)
st.plotly_chart(fig_funnel, use_container_width=True)

st.markdown("---")

# Time series per channel (small multiples)

if channels:
    st.subheader("Channel Trends")
    channels_to_show = channels
    tabs = st.tabs(channels_to_show)
    for i, ch in enumerate(channels_to_show):
        with tabs[i]:
            ch_df = daily_channel[(daily_channel['channel']==ch) & (daily_channel['date']>=start_date) & (daily_channel['date']<=end_date)].copy()
            if ch_df.empty:
                st.write("No data for", ch)
            else:
                fig_ch = go.Figure()
                fig_ch.add_trace(go.Bar(x=ch_df['date'], y=ch_df['spend'], name='Spend', yaxis='y1'))
                fig_ch.add_trace(go.Scatter(x=ch_df['date'], y=ch_df['attributed revenue'], name='Attributed Revenue', yaxis='y2', mode='lines+markers'))
                fig_ch.update_layout(yaxis=dict(title='Spend'), yaxis2=dict(title='Attributed Revenue', overlaying='y', side='right'), height=350)
                fig_ch.update_layout(legend=dict(orientation='h', y=-0.15))

                st.plotly_chart(fig_ch, use_container_width=True)
                st.dataframe(
                    ch_df[['date','impression','clicks','spend','attributed revenue','CTR','CPC','ROAS']]
                    .assign(
                        date=lambda d: d['date'].dt.strftime('%Y-%m-%d'),
                        impression=lambda d: d['impression'].map('{:,.0f}'.format),
                        clicks=lambda d: d['clicks'].map('{:,.0f}'.format),
                        spend=lambda d: d['spend'].map('${:,.0f}'.format),
                        attributed_revenue=lambda d: d['attributed revenue'].map('${:,.0f}'.format),
                        CTR=lambda d: d['CTR'].map('{:.2%}'.format),
                        CPC=lambda d: d['CPC'].map('${:,.2f}'.format),
                        ROAS=lambda d: d['ROAS'].map('{:.2f}'.format)
                    )
                    .rename(columns={"attributed revenue": "Attributed Revenue"})  # fix column name
                    .loc[:, ['date','impression','clicks','spend','Attributed Revenue','CTR','CPC','ROAS']]
                    .set_index('date'),
                    height=300
    )


    st.markdown("---")

# State level performance (if state exists)
if 'state' in marketing.columns:
    st.subheader("State-level Performance")
    state_aggs = filtered_marketing.groupby('state').agg({'impression':'sum','clicks':'sum','spend':'sum','attributed revenue':'sum'}).reset_index()
    if not state_aggs.empty:
        state_aggs['ROAS'] = np.where(state_aggs['spend']>0, state_aggs['attributed revenue']/state_aggs['spend'], np.nan)
        fig_state = px.bar(state_aggs.sort_values('spend', ascending=False).head(15), x='state', y='spend', color='ROAS', hover_data=['attributed revenue'], height=420)
        fig_state.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_state, use_container_width=True)
        st.dataframe(state_aggs.sort_values('spend', ascending=False).head(50))
    else:
        st.write("No state-level data available for selected filters.")

st.markdown("---")

# Correlation: Spend vs Attributed Revenue scatter with regression
st.subheader("Spend vs Attributed Revenue (Correlation & Diminishing Returns)")
scatter_df = filtered_marketing.groupby(['date']).agg({'spend':'sum','attributed revenue':'sum'}).reset_index()
if scatter_df.empty:
    st.write("No data for selected filters.")
else:
    fig_corr = px.scatter(scatter_df, x='spend', y='attributed revenue', trendline='ols', height=420)
    fig_corr.update_xaxes(tickprefix="$")
    fig_corr.update_yaxes(tickprefix="$")
    st.plotly_chart(fig_corr, use_container_width=True)
    corr_val = scatter_df['spend'].corr(scatter_df['attributed revenue'])
    st.write(f"Pearson correlation (spend vs attributed revenue): {corr_val:.2f}")

st.markdown("---")

# Export: allow user to download filtered marketing data and aggregates
st.subheader("Export filtered data")
@st.cache_data
def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

if not filtered_marketing.empty:
    st.download_button("Download filtered marketing rows (CSV)", data=to_csv(filtered_marketing), file_name="filtered_marketing.csv", mime="text/csv")
if not merged_daily_filtered.empty:
    st.download_button("Download merged daily summary (CSV)", data=to_csv(merged_daily_filtered), file_name="merged_daily_summary.csv", mime="text/csv")
