import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from core.engines import AssetEngine, MLEngine, SentimentEngine, DataFetcher
from core.ui_components import inject_custom_css, custom_metric, safe_text
from fpdf import FPDF

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Elite Wealth SaaS Portal", page_icon="💎")
inject_custom_css()

# --- INITIALIZATION ---
TICKERS_US = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "GLD", "BTC"]
TICKERS_EU = ["ASML", "LVMH", "SAP", "OR.PA", "MC.PA", "DAX.DE", "CAC.PA"]
ALL_TICKERS = TICKERS_US + TICKERS_EU

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def init_data():
    # Primary: yfinance, Fallback: Simulation
    data = AssetEngine.get_real_data(ALL_TICKERS)
    if data is not None and not data.empty:
        st.session_state.mkt_data = data
        st.session_state.returns = data.pct_change().dropna()
    else:
        st.session_state.mkt_data = AssetEngine.generate_market_data(ALL_TICKERS)
        st.session_state.returns = st.session_state.mkt_data.pct_change().dropna()
    
    # Track Benchmark (Safe selection)
    if not st.session_state.mkt_data.empty:
        if "SPY" in st.session_state.mkt_data.columns:
            st.session_state.benchmark = st.session_state.mkt_data["SPY"]
        elif len(st.session_state.mkt_data.columns) > 0:
            st.session_state.benchmark = st.session_state.mkt_data.iloc[:, 0]
        else:
            st.session_state.benchmark = pd.Series()
    else:
        st.session_state.benchmark = pd.Series()

if 'mkt_data' not in st.session_state:
    init_data()

if 'clients' not in st.session_state:
    # Default weights (Moderate)
    def_weights = np.array([0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05])
    st.session_state.clients = [
        {"id": 1, "Name": "Jean Dupont", "Profile": "Modéré", "KYC": "Complet", "Score": 65, "Weights": def_weights, "Value": 1450200},
        {"id": 2, "Name": "Marie Lefebvre", "Profile": "Prudent", "KYC": "Incomplet", "Score": 30, "Weights": def_weights, "Value": 850000},
    ]

if 'trades' not in st.session_state:
    st.session_state.trades = []

# Global Engines
ml = MLEngine()
sent = SentimentEngine()
currency = "USD"

# --- AUTHENTICATION MOCKUP ---
if not st.session_state.authenticated:
    st.markdown("<h1 style='text-align: center; color: #38bdf8;'>💎 Elite Wealth SaaS Portal</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.form("login_form"):
            st.subheader("Accès Personnalisé")
            name = st.text_input("Veuillez entrer votre prénom pour vous connecter")
            if st.form_submit_button("Se Connecter"):
                if name.strip().lower() == "mackenson":
                    st.session_state.authenticated = True
                    st.session_state.user_name = "Mackenson"
                    st.success("Bonjour Mackenson ! Accès autorisé.")
                    st.rerun()
                else:
                    st.error("Prénom non reconnu. Accès refusé.")
    st.stop()

st.sidebar.markdown(f"**Admin : {st.session_state.user_name}**")
st.sidebar.markdown(f"**AUM Total:** {sum(c['Value'] for c in st.session_state.clients):,.0f} {currency}")

menu = st.sidebar.radio("Navigation", [
    "👥 Clients & CRM", 
    "📊 Dashboard", 
    "📈 Portefeuille", 
    "🛡️ Risk Intelligence", 
    "🧠 AI Predictions", 
    "🚀 Simulation", 
    "⚡ Trading OMS",
    "📜 Rapport PDF"
])
st.sidebar.divider()
client_idx = st.sidebar.selectbox("Fiche Client Active", range(len(st.session_state.clients)), 
                                  format_func=lambda x: st.session_state.clients[x]['Name'])
curr_client = st.session_state.clients[client_idx]

# Recalculate metrics for current client
metrics = AssetEngine.calculate_risk_metrics(st.session_state.returns, curr_client['Weights'])
weights = curr_client['Weights']

# --- ROUTER ---
if menu == "📊 Dashboard":
    st.markdown(f"<h2 class='section-header'>Aperçu Consolidé : {curr_client['Name']}</h2>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: custom_metric("Valeur Portefeuille", f"{curr_client['Value']:,} {currency}", "+1.2%", "Dernière session")
    with c2: custom_metric("Sharpe Ratio", f"{metrics['Sharpe']}", "High", "Based on active profile")
    with c3:
        if not st.session_state.returns.empty:
            anomalies = ml.detect_anomalies(st.session_state.returns)
            status = "NORMAL" if anomalies[-1] == 1 else "ANOMALY"
        else:
            status = "N/A"
        custom_metric("AI Market Status", status, "Stable" if status=="NORMAL" else "Wait", "Isolation Forest Scan")
    with c4: 
        s_score = sent.get_score(DataFetcher.fetch_mock_sentiment("Global"))
        custom_metric("Sentiment AI", f"{s_score*100:+.0f}", "Live", "Social & News Fusion")
    
    st.markdown("<br>", unsafe_allow_html=True)
    c_left, c_right = st.columns([2, 1])
    with c_left:
        fig_perf = go.Figure()
        port_cum = st.session_state.returns.dot(weights).cumsum()
        bench_cum = st.session_state.benchmark.pct_change().cumsum()
        
        fig_perf.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Elite Portfolio", line=dict(color='#38bdf8', width=3)))
        fig_perf.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name="Benchmark (SPY)", line=dict(color='#94a3b8', dash='dash')))
        
        fig_perf.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title="Performance vs Benchmark")
        st.plotly_chart(fig_perf, use_container_width=True)
    with c_right:
        fig_alloc = px.pie(values=weights, names=ALL_TICKERS, hole=0.6, title="Allocation par Actif")
        fig_alloc.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig_alloc, use_container_width=True)

elif menu == "📈 Portefeuille":
    st.markdown("<h2 class='section-header'>Portefeuille Détaillé</h2>", unsafe_allow_html=True)
    df_p = pd.DataFrame({
        "Actif": ALL_TICKERS, 
        "Poids (%)": weights * 100, 
        f"Valeur ({currency})": [1450200 * w for w in weights]
    })
    st.dataframe(
        df_p, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            f"Valeur ({currency})": st.column_config.NumberColumn(format=f"%.2f {currency}")
        }
    )
    if st.button("⚖️ Rééquilibrer le Portefeuille"):
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.info("Rééquilibrage en cours...")
            fig_reb = px.bar(x=ALL_TICKERS, y=weights, title="Allocation Avant", color_discrete_sequence=['#94a3b8'])
            st.plotly_chart(fig_reb, use_container_width=True)
        with col_r2:
            st.success("Portefeuille réaligné.")
            # Mock rebalance: shift slightly towards target
            new_w = weights * 1.02
            new_w = new_w / new_w.sum()
            fig_reb2 = px.bar(x=ALL_TICKERS, y=new_w, title="Allocation Après", color_discrete_sequence=['#38bdf8'])
            st.plotly_chart(fig_reb2, use_container_width=True)
            st.session_state.clients[client_idx]['Weights'] = new_w

elif menu == "🛡️ Risk Intelligence":
    st.markdown("<h2 class='section-header'>Risk Intelligence & Analytics</h2>", unsafe_allow_html=True)
    rk1, rk2, rk3 = st.columns(3)
    with rk1: st.metric("Value-at-Risk (95%)", f"-{metrics['VaR_95']*1450200:,.0f} {currency}")
    with rk2: st.metric("Volatilité Ann.", f"{metrics['Volatility']*100:.1f}%")
    with rk3: st.metric("Max Drawdown", f"{metrics['MaxDD']*100:.1f}%")

    st.markdown("### Profil Risque vs Rendement")
    if not st.session_state.returns.empty:
        asset_metrics = AssetEngine.get_asset_metrics(st.session_state.returns)
        fig_rv = px.scatter(asset_metrics, x="Volatility", y="Return", text="Asset", 
                            title="Analyse Comparative : Risque vs Rendement",
                            labels={"Volatility": "Risque (Volatilité Ann.)", "Return": "Rendement Attendu (Ann.)"})
        # Add Portfolio point
        fig_rv.add_trace(go.Scatter(x=[metrics['Volatility']], y=[metrics['AnnualRet']], 
                                    mode='markers+text', name='Current Portfolio',
                                    text=['MON PORTEFEUILLE'], textposition="top center",
                                    marker=dict(color='#38bdf8', size=15, symbol='star')))
        fig_rv.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_rv, use_container_width=True)

elif menu == "🧠 AI Predictions":
    st.markdown("<h2 class='section-header'>Predictive Analytics</h2>", unsafe_allow_html=True)
    selected_asset = st.selectbox("Sélectionner l'actif", ALL_TICKERS)
    col_a1, col_a2 = st.columns([1, 2])
    with col_a1:
        s_text = DataFetcher.fetch_mock_sentiment(selected_asset)
        s_score = sent.get_score(s_text)
        st.info(f"**Sentiment Récent:** {s_text}")
        st.metric("Sentiment Score", f"{s_score:.2f}", delta="Bullish" if s_score > 0 else "Bearish")
        
        preds = ml.predict_returns(st.session_state.mkt_data, selected_asset)
        if len(preds) > 0:
            trend = "Hausse" if preds[-1] > preds[0] else "Baisse"
            move = (preds[-1]/preds[0]-1)*100
            st.metric("Tendance IA (7j)", trend, delta=f"{move:+.2f}%")
        else:
            st.metric("Tendance IA (7j)", "N/A")
    with col_a2:
        future_dates = [st.session_state.mkt_data.index[-1] + timedelta(days=i) for i in range(1, 8)]
        fig_ml = go.Figure()
        fig_ml.add_trace(go.Scatter(x=st.session_state.mkt_data.index[-20:], y=st.session_state.mkt_data[selected_asset][-20:], name="Historique"))
        fig_ml.add_trace(go.Scatter(x=future_dates, y=preds, name="Projection ML", line=dict(dash='dash', color='#38bdf8')))
        fig_ml.update_layout(template="plotly_dark", title=f"Projection {selected_asset}")
        st.plotly_chart(fig_ml, use_container_width=True)

elif menu == "🚀 Simulation":
    st.markdown("<h2 class='section-header'>Simulation Stratégique</h2>", unsafe_allow_html=True)
    sim_days = st.slider("Horizon (jours)", 30, 504, 252)
    if st.button("Lancer Simulations Monte Carlo"):
        mu, sigma = st.session_state.returns.mean().dot(weights), st.session_state.returns.dot(weights).std()
        sim_data = np.zeros((sim_days, 50))
        for i in range(50):
            rets = np.random.normal(mu, sigma, sim_days)
            sim_data[:, i] = 1450200 * np.exp(np.cumsum(rets))
        fig_mc = go.Figure()
        for i in range(10): fig_mc.add_trace(go.Scatter(y=sim_data[:, i], opacity=0.3, showlegend=False, line=dict(color='gray')))
        fig_mc.add_trace(go.Scatter(y=sim_data.mean(axis=1), name="Moyenne Portefeuille", line=dict(color='#38bdf8', width=3)))
        
        # Benchmark simulation (simplified)
        if not st.session_state.benchmark.empty:
            b_mu = st.session_state.benchmark.pct_change().mean()
            b_sigma = st.session_state.benchmark.pct_change().std()
            b_sim = 1450200 * np.exp(np.cumsum(np.random.normal(b_mu, b_sigma, sim_days)))
            fig_mc.add_trace(go.Scatter(y=b_sim, name="Projection Marché (SPY)", line=dict(color='#94a3b8', dash='dash')))
            
        fig_mc.update_layout(template="plotly_dark", title="Simulation Monte Carlo : Portefeuille vs Marché")
        st.plotly_chart(fig_mc, use_container_width=True)

elif menu == "⚡ Trading OMS":
    st.markdown("<h2 class='section-header'>SaaS Order Management System</h2>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🚀 Exécution Directe", "⚖️ Modélisation de Portefeuille"])
    
    with tab1:
        with st.form("trade_form"):
            t_ticker = st.selectbox("Actif", ALL_TICKERS)
            t_side = st.radio("Côté", ["BUY", "SELL"])
            t_qty = st.number_input("Quantité", min_value=1)
            if st.form_submit_button("Exécuter"):
                st.session_state.trades.append({"Date": datetime.now().strftime("%H:%M:%S"), "Asset": t_ticker, "Side": t_side, "Qty": t_qty, "Status": "Executed"})
                st.success("Ordre exécuté !")
        st.dataframe(pd.DataFrame(st.session_state.trades), use_container_width=True)

    with tab2:
        st.subheader("Optimisation selon le Profil")
        profile_choice = st.selectbox("Choisir Profil Cible", ["Prudent", "Modéré", "Agressif"])
        
        # Preset Compositions
        presets = {
            "Prudent": {"GLD": 0.3, "SAP": 0.2, "SPY": 0.2, "Others": 0.3},
            "Modéré": {"SPY": 0.4, "AAPL": 0.1, "ASML": 0.1, "Others": 0.4},
            "Agressif": {"NVDA": 0.3, "BTC": 0.2, "QQQ": 0.3, "Others": 0.2}
        }
        
        target = presets[profile_choice]
        st.write(f"**Composition Suggérée ({profile_choice}):**")
        st.json(target)
        
        # Comparison Chart (Simplified)
        fig_comp = go.Figure(data=[
            go.Bar(name='Actuel', x=list(target.keys()), y=[0.1]*len(target)), # Placeholder actuals
            go.Bar(name='Cible', x=list(target.keys()), y=list(target.values()))
        ])
        fig_comp.update_layout(barmode='group', template="plotly_dark", title="Analyse d'Écart de Poids")
        st.plotly_chart(fig_comp, use_container_width=True)
        
        if st.button("Appliquer la Composition Cible"):
            st.warning("Génération automatique d'ordres de rééquilibrage...")
            st.success("Ordres envoyés à l'OMS.")

elif menu == "👥 Clients & CRM":
    st.markdown("<h2 class='section-header'>Client Intelligence & Risk Profiling</h2>", unsafe_allow_html=True)
    
    # Client Selection
    st.subheader(f"Dossier Actif : {curr_client['Name']}")
    
    col_c1, col_c2 = st.columns([1, 1.5])
    
    with col_c1:
        st.markdown("### Résumé du Profil")
        st.write(f"- **Profil actuel :** {curr_client['Profile']}")
        st.write(f"- **Score de Risque :** {curr_client['Score']}/100")
        st.write(f"- **Statut KYC :** {curr_client['KYC']}")
        st.info("Utilisez l'onglet 'Rapport PDF' pour générer le document officiel.")

    with col_c2:
        st.markdown("### 📝 Formulaire de Recommandation & Profilage")
        with st.form("risk_profiling_form"):
            st.write("Veuillez remplir ce formulaire pour une analyse complète.")
            full_name = st.text_input("Nom Complet du Client", value=curr_client['Name'])
            
            q1 = st.slider("1. Horizon d'investissement souhaité (ans)", 1, 30, 15)
            q2 = st.selectbox("2. Réaction face à une baisse de 15% du marché", 
                              ["Vendre tout", "Attendre patiemment", "Acheter davantage"])
            q3 = st.radio("3. Objectif principal", ["Préservation du capital", "Croissance équilibrée", "Maximisation des gains (Speculation)"])
            q4 = st.slider("4. Pourcentage d'épargne alloué aux actifs risqués", 0, 100, 50)
            q5 = st.selectbox("5. Expérience en investissement", ["Débutant", "Intermédiaire", "Expert (Trading actif)"])
            
            if st.form_submit_button("Lancer l'Analyse"):
                # Comprehensive Scoring Logic
                score = (q1 * 1.5) 
                score += (30 if q2=="Acheter davantage" else 15 if q2=="Attendre patiemment" else 0)
                score += (30 if q3=="Maximisation des gains (Speculation)" else 15 if q3=="Croissance équilibrée" else 5)
                score += (q4 / 3)
                score += (20 if q5=="Expert (Trading actif)" else 10 if q5=="Intermédiaire" else 0)
                score = min(score, 100)
                
                # Update Session State
                new_profile = "Agressif" if score > 75 else "Modéré" if score > 45 else "Prudent"
                st.session_state.clients[client_idx]['Name'] = full_name
                st.session_state.clients[client_idx]['Score'] = int(score)
                st.session_state.clients[client_idx]['Profile'] = new_profile
                
                st.markdown("---")
                st.subheader(f"Résultat pour {full_name} : Profil {new_profile}")
                
                # Advanced Recommendations
                if new_profile == "Agressif":
                    recs = {"NVDA": "Growth", "BTC": "Asset Diversification", "SPY": "Core"}
                    advice = "Stratégie offensive axée sur la surperformance technologique."
                elif new_profile == "Modéré":
                    recs = {"SPY": "Core", "MSFT": "Stability", "GLD": "Hedge"}
                    advice = "Allocation équilibrée entre croissance US et actifs refuges."
                else:
                    recs = {"GLD": "Store of Value", "SAP": "Value Europe", "Cash/Bills": "Safety"}
                    advice = "Approche conservatrice visant à minimiser la volatilité."
                
                # Advanced Recommendations & Assignment
                if new_profile == "Agressif":
                    recs = {"NVDA": 0.3, "BTC": 0.2, "QQQ": 0.3, "SPY": 0.2}
                    advice = "Stratégie offensive axée sur la surperformance technologique."
                elif new_profile == "Modéré":
                    recs = {"SPY": 0.5, "MSFT": 0.15, "AAPL": 0.15, "GLD": 0.2}
                    advice = "Allocation équilibrée entre croissance US et actifs refuges."
                else:
                    recs = {"GLD": 0.4, "SAP": 0.2, "CAC.PA": 0.2, "Cash/Bills": 0.2}
                    advice = "Approche conservatrice visant à minimiser la volatilité."
                
                # Mapper weights (simplified)
                w_dict = {t: 0.0 for t in ALL_TICKERS}
                for t, v in recs.items():
                    if t in w_dict: w_dict[t] = v
                
                # Normalize
                final_w = np.array(list(w_dict.values()))
                final_w = final_w / final_w.sum()
                
                st.session_state.clients[client_idx]['Weights'] = final_w
                
                st.success(f"**Recommandation Stratégique :** {advice}")
                st.write("✅ **Portefeuille généré et appliqué au client.**")
                
                cols = st.columns(len(recs))
                for i, (k, v) in enumerate(recs.items()):
                    cols[i].metric(k, f"{v*100:.0f}%")
                
elif menu == "📜 Rapport PDF":
    st.markdown("<h2 class='section-header'>Génération de Rapport Officiel</h2>", unsafe_allow_html=True)
    st.subheader(f"Client : {curr_client['Name']}")
    st.write("Ce module compile les données de performance, d'allocation et de risque pour générer un rapport PDF professionnel.")
    
    if st.button("� Générer le Rapport PDF"):
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Header
            pdf.set_font("Arial", 'B', 20)
            pdf.set_text_color(56, 189, 248)
            pdf.cell(0, 15, "Elite Wealth Management - Private Portfolio", ln=True, align='C')
            pdf.ln(10)
            
            # Client Info
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, f"Rapport de Gestion : {curr_client['Name']}", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
            pdf.ln(10)
            
            # Summary Table
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "1. Synthese de l'Investissement", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 8, f"- Profil de Risque : {curr_client['Profile']}", ln=True)
            pdf.cell(0, 8, f"- Valeur Totale : {curr_client['Value']:,} {currency}", ln=True)
            pdf.cell(0, 8, f"- Performance Annuelle : {metrics['AnnualRet']*100:.1f}%", ln=True)
            pdf.cell(0, 8, f"- Ratio de Sharpe : {metrics['Sharpe']}", ln=True)
            pdf.ln(10)
            
            # Risk Stats
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "2. Analyse de Risque Intelligence", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 8, f"- Value-at-Risk (95%) : -{metrics['VaR_95']*curr_client['Value']:,.2f} {currency}", ln=True)
            pdf.cell(0, 8, f"- Volatilite : {metrics['Volatility']*100:.1f}%", ln=True)
            pdf.cell(0, 8, f"- Max Drawdown : {metrics['MaxDD']*100:.1f}%", ln=True)
            
            pdf_output = pdf.output(dest='S').encode('latin-1')
            st.download_button("📥 Télécharger le PDF", data=pdf_output, file_name=f"Elite_Report_{curr_client['Name']}.pdf", mime="application/pdf")
            st.success("Rapport généré avec succès !")
        except Exception as e:
            st.error(f"Erreur lors de la génération : {e}")
