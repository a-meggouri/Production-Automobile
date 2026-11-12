"""Suivi de la production automobile : Tanger et Kénitra.

Lancer avec : streamlit run app.py
"""
import hmac
import textwrap
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

st.set_page_config(page_title="Suivi de production automobile", page_icon="🚗", layout="wide")

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
UTILISATEUR = "Amine"
MOT_DE_PASSE = "0000"
DATA = Path(__file__).parent / "dataset.csv"

C = dict(fond="#EDEFF3", encre="#14182B", bleu="#3F37C9", rouge="#E4372C",
         jaune="#F5B301", vert="#1F9D6B", acier="#6B7287", trait="#D5D9E3")
PAL_CAT = ["#3F37C9", "#7A73E0", "#B3AFF0", "#F5B301"]
COUL_EQUIPE = {"Matin": "#9AA1B5", "Après-midi": C["bleu"], "Nuit": C["encre"]}
CM_DEFAUT = LinearSegmentedColormap.from_list("defaut", ["#F2F3F8", C["rouge"]])
CM_TRS = LinearSegmentedColormap.from_list("trs", [C["rouge"], "#F2F3F8", C["bleu"]])

ORDRE_LIGNES = ["Emboutissage", "Ferrage", "Peinture", "Montage final"]
ORDRE_EQUIPES = ["Matin", "Après-midi", "Nuit"]
JOURS = {"Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi", "Thursday": "Jeudi",
         "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche"}
ORDRE_JOURS = list(JOURS.values())
MOIS_LONG = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
             "septembre", "octobre", "novembre", "décembre"]
MOIS_COURT = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août",
              "sept.", "oct.", "nov.", "déc."]
CAUSE_MAINT = "Maintenance préventive"

# ----------------------------------------------------------------------------
# Style de l'interface
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Figtree:wght@400;500;600&display=swap');
:root{--fond:#EDEFF3;--encre:#14182B;--bleu:#3F37C9;--rouge:#E4372C;--jaune:#F5B301;
      --vert:#1F9D6B;--acier:#6B7287;--trait:#D5D9E3;}
.stApp{background:var(--fond);font-family:'Figtree',system-ui,sans-serif;color:var(--encre);}
.stApp p,.stApp label,.stApp li{font-family:'Figtree',system-ui,sans-serif;}
#MainMenu,footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1280px;padding-top:2.2rem;padding-bottom:4rem;}

/* Couleurs forcées : l'interface reste lisible même si le thème Streamlit est sombre */
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp label,
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p{color:var(--encre);}
button[data-baseweb="tab"]{color:var(--encre);}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--bleu);}
section[data-testid="stSidebar"] *{color:var(--encre);}
section[data-testid="stSidebar"] [data-baseweb="tag"] *{color:#fff;}
.stApp [data-baseweb="select"] > div,.stApp [data-baseweb="base-input"],
.stApp [data-baseweb="input"] > div{background:#fff;border-color:var(--trait);}
.stApp input{color:var(--encre);-webkit-text-fill-color:var(--encre);}
.stApp [data-testid^="stBaseButton"]{background:var(--encre);border:none;border-radius:10px;}
.stApp [data-testid^="stBaseButton"] *{color:#fff;}
.stApp [data-testid="stAlert"],.stApp [data-testid="stExpander"]{background:#fff;border:1px solid var(--trait);border-radius:12px;}
.stApp [data-testid="stAlert"] *,.stApp [data-testid="stExpander"] *{color:var(--encre);}
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--trait);}
.side-titre{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.35rem;margin:0 0 .6rem;}
button[data-baseweb="tab"]{font-weight:600;font-size:1rem;padding:.6rem 1rem;}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--bleu);}
div[data-baseweb="tab-highlight"]{background-color:var(--bleu);}

.hero-t{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;line-height:1;
        font-size:clamp(2.4rem,5.2vw,4.2rem);letter-spacing:-.035em;margin:0 0 .7rem;}
.stApp .hero-s{font-size:1.05rem;color:var(--acier);max-width:60ch;margin:0 0 2rem;}

.chaine{position:relative;padding-top:6px;}
.rail{position:absolute;top:10px;left:0;right:0;height:3px;background:var(--trait);}
.car{position:absolute;top:-9px;left:0;width:30px;height:12px;background:var(--bleu);
     border-radius:8px 8px 2px 2px;z-index:3;animation:roule 2.8s cubic-bezier(.5,0,.2,1) 1 both;}
.car::before,.car::after{content:"";position:absolute;bottom:-3px;width:6px;height:6px;
     border-radius:50%;background:var(--encre);}
.car::before{left:4px;}.car::after{right:4px;}
@keyframes roule{from{left:0}to{left:calc(100% - 30px)}}
@media (prefers-reduced-motion:reduce){.car{animation:none;left:calc(100% - 30px);}}
.postes{display:grid;grid-template-columns:repeat(4,1fr);}
.poste{position:relative;padding:30px 14px 0 0;display:flex;flex-direction:column;gap:2px;}
.poste::before{content:"";position:absolute;left:0;top:5px;width:11px;height:11px;border-radius:50%;
     background:var(--fond);border:2.5px solid var(--encre);z-index:2;}
.poste.alerte::before{background:var(--rouge);border-color:var(--rouge);}
.pn{font-weight:600;font-size:.95rem;color:var(--encre);}
.pv{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:2.1rem;
    letter-spacing:-.02em;line-height:1.1;color:var(--encre);}
.pl{color:var(--acier);font-size:.85rem;}
.chip{display:inline-block;align-self:flex-start;margin-top:.55rem;padding:.2rem .7rem;
      border-radius:999px;border:1.5px solid var(--bleu);color:var(--bleu);font-weight:600;font-size:.85rem;}
.alerte .chip{background:var(--rouge);border-color:var(--rouge);color:#fff;}
.pt{margin-top:.35rem;font-size:.85rem;color:var(--acier);}
@media (max-width:800px){.postes{grid-template-columns:repeat(2,1fr);row-gap:1.5rem;}.rail,.car{display:none;}}

.bandeau{display:grid;grid-template-columns:repeat(6,1fr);background:#fff;border-radius:16px;
         margin:2.2rem 0 1.2rem;overflow:hidden;}
.k{padding:1.1rem 1.3rem;border-left:1px solid var(--trait);}
.k:first-child{border-left:none;}
.kv{display:block;font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.85rem;
    letter-spacing:-.02em;line-height:1.2;color:var(--encre);}
.kv.r{color:var(--rouge);}.kv.b{color:var(--bleu);}
.ku{font-size:.95rem;color:var(--acier);font-weight:500;margin-left:.2rem;}
.kl{display:block;font-size:.88rem;color:var(--acier);}
@media (max-width:900px){.bandeau{grid-template-columns:repeat(2,1fr);}
     .k{border-left:none;border-top:1px solid var(--trait);}}

.bt{font-weight:700;font-size:1.05rem;margin:1.3rem 0 .1rem;color:var(--encre);}
.bl{color:var(--acier);font-size:.88rem;margin-bottom:.5rem;}
.constat{background:#fff;border-radius:14px;padding:1rem 1.3rem;margin-bottom:.8rem;
         border-left:6px solid var(--vert);}
.constat.non{border-left-color:var(--jaune);}
.cv{font-size:.85rem;font-weight:600;color:var(--vert);}
.constat.non .cv{color:#9A6B00;}
.ct{font-weight:700;font-size:1.05rem;margin:.1rem 0 .3rem;color:var(--encre);}
.ctx{color:#3B4157;font-size:.95rem;}

.marque{background:var(--encre);color:#F4F5F8;border-radius:22px;padding:2.6rem 2.4rem;min-height:520px;}
.marque .hero-t{font-size:clamp(2.2rem,4vw,3.3rem);color:#fff;}
.marque .hero-s{color:#B9BED0;}
.marque .rail{background:rgba(255,255,255,.22);}
.marque .poste::before{background:var(--encre);border-color:#fff;}
.marque .pn{font-size:.85rem;color:#fff;}
.marque .chaine{margin-top:3.5rem;}
.marque .postes{grid-template-columns:repeat(2,1fr);row-gap:.6rem;}
.login-t{color:var(--encre);font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:2.2rem;
         letter-spacing:-.03em;margin-top:2rem;}
.login-s{color:var(--acier);margin-bottom:1.2rem;}

.synthese{background:var(--encre);border-radius:20px;padding:1.8rem 2rem;margin-bottom:1.4rem;}
.synthese .sv{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:2.6rem;
              letter-spacing:-.03em;line-height:1.1;color:#fff;}
.synthese .st{color:#C9CEDF;font-size:1rem;margin-top:.4rem;max-width:70ch;}
.titre-s{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.5rem;
         color:var(--encre);margin:1.6rem 0 .8rem;}
.recos{display:grid;grid-template-columns:repeat(2,1fr);gap:.9rem;}
.reco{background:#fff;border-radius:14px;padding:1.1rem 1.3rem;border-top:4px solid var(--bleu);}
.reco .rt{font-weight:700;font-size:1.02rem;color:var(--encre);margin-bottom:.35rem;}
.reco .rx{font-size:.93rem;color:#3B4157;}
@media (max-width:800px){.recos{grid-template-columns:1fr;}}
</style>
"""

# ----------------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------------
def fr(x, dec=0):
    """Nombre au format français : 1 234,5."""
    if pd.isna(x):
        return "–"
    return f"{x:,.{dec}f}".replace(",", "\u202f").replace(".", ",")


def date_fr(ts):
    return f"{ts.day} {MOIS_LONG[ts.month - 1]} {ts.year}"


def theme_mpl():
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
        "axes.edgecolor": C["trait"], "axes.labelcolor": C["acier"], "text.color": C["encre"],
        "xtick.color": C["acier"], "ytick.color": C["acier"], "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["trait"], "grid.linewidth": .6, "axes.axisbelow": True,
    })


def axe_dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f"1 {MOIS_COURT[mdates.num2date(x).month - 1]}"))
    ax.grid(axis="x", visible=False)


def montrer(fig):
    if fig is None:
        st.info("Pas assez de données pour ce graphique avec les filtres actuels.")
        return
    st.pyplot(fig)
    plt.close(fig)


def bloc(titre, lecture):
    st.markdown(f'<div class="bt">{titre}</div><div class="bl">{lecture}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement des données…")
def charger():
    brut = pd.read_csv(DATA, encoding="utf-8-sig")
    qualite = dict(
        brut=len(brut),
        doublons=int(brut.duplicated().sum()),
        cycle_vide=int(brut["temps_cycle_s"].isna().sum()),
        cycle_vide_prod_nulle=int((brut["temps_cycle_s"].isna() & (brut["vehicules_produits"] == 0)).sum()),
        incoherences=int((brut["vehicules_conformes"]
                          != brut["vehicules_produits"] - brut["vehicules_defectueux"]).sum()),
    )
    df = brut.drop_duplicates().copy()
    df["date_heure"] = pd.to_datetime(df["date_heure"])
    df["date"] = pd.to_datetime(df["date"])
    df["jour"] = pd.Categorical(df["jour_semaine"].map(JOURS), categories=ORDRE_JOURS, ordered=True)
    df["cause_arret"] = df["cause_arret"].replace({"Maintenance prévent": CAUSE_MAINT})
    df["maintenance"] = df["cause_arret"].eq(CAUSE_MAINT)
    df["semaine"] = df["date"].dt.to_period("W").dt.start_time
    return df, qualite


def agreger(d, par):
    """Agrégats pondérés : le taux de défauts est total défauts / total produits."""
    g = d.groupby(par, observed=True).agg(
        produits=("vehicules_produits", "sum"), defauts=("vehicules_defectueux", "sum"),
        trs=("TRS_pct", "mean"), arret_min=("temps_arret_min", "sum"),
        kwh=("energie_kwh", "sum"), n=("vehicules_produits", "size"))
    g["taux_defaut"] = g["defauts"] / g["produits"].replace(0, np.nan) * 100
    g["kwh_veh"] = g["kwh"] / g["produits"].replace(0, np.nan)
    return g.reset_index()


def kpis(d):
    p = d["vehicules_produits"].sum()
    return dict(
        prod=p,
        taux=d["vehicules_defectueux"].sum() / p * 100 if p else np.nan,
        trs=d["TRS_pct"].mean(),
        arret=d["temps_arret_min"].sum() / 60,
        kwh=d["energie_kwh"].sum() / p if p else np.nan,
        cadence=p / d["cadence_cible_h"].sum() * 100,
    )


def serie_hebdo(d):
    s = agreger(d, "semaine")
    s = s[s["produits"] > 0]
    if len(s):
        s = s[s["n"] >= 0.5 * s["n"].max()]  # écarte les semaines incomplètes
    return s.reset_index(drop=True)


def pente(s):
    if len(s) < 3:
        return None
    return float(np.polyfit(np.arange(len(s)), s["taux_defaut"].values, 1)[0])


# ----------------------------------------------------------------------------
# Graphiques
# ----------------------------------------------------------------------------
def fig_production(d):
    j = d.groupby("date")["vehicules_produits"].sum()
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.fill_between(j.index, j.values, color=C["bleu"], alpha=.10, lw=0)
    ax.plot(j.index, j.values, color=C["bleu"], lw=.6, alpha=.5)
    ax.plot(j.index, j.rolling(7, min_periods=1).mean(), color=C["bleu"], lw=2.4)
    ax.set_ylabel("Véhicules par jour")
    axe_dates(ax)
    return fig


def fig_defauts(d):
    s = serie_hebdo(d)
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.plot(s["semaine"], s["taux_defaut"], color=C["bleu"], lw=2.2, marker="o", ms=4)
    if len(s) >= 3:
        x = np.arange(len(s))
        ax.plot(s["semaine"], np.polyval(np.polyfit(x, s["taux_defaut"].values, 1), x),
                ls="--", color=C["rouge"], lw=1.7)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Véhicules défectueux (%)")
    axe_dates(ax)
    return fig


def carte(p, cmap, dec=2, figsize=(5, 3), annot=True, cbar=False):
    if p.dropna(how="all").empty:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    lab = p.map(lambda v: fr(v, dec)) if annot else False
    sns.heatmap(p, ax=ax, cmap=cmap, annot=lab, fmt="", linewidths=1.2, linecolor=C["fond"],
                cbar=cbar, cbar_kws={"shrink": .8}, annot_kws={"fontsize": 9})
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)
    ax.tick_params(axis="y", rotation=0)
    ax.grid(False)
    return fig


def pivot_ordre(d, idx, col, ordre_idx, ordre_col, val):
    p = agreger(d, [idx, col]).pivot(index=idx, columns=col, values=val)
    return p.reindex(index=[o for o in ordre_idx if o in p.index],
                     columns=[o for o in ordre_col if o in p.columns])


def fig_trs_heure(d):
    p = d.pivot_table(index="jour", columns="heure", values="TRS_pct", aggfunc="mean", observed=True)
    fig = carte(p, CM_TRS, annot=False, figsize=(8.6, 3.3), cbar=True)
    return fig


def fig_donut(d):
    a = d.groupby("modele")["vehicules_produits"].sum().sort_values(ascending=False)
    if a.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    ax.pie(a.values, colors=PAL_CAT[:len(a)], startangle=90, counterclock=False,
           wedgeprops=dict(width=.36, edgecolor=C["fond"], linewidth=2))
    ax.legend([f"{m}  {fr(v / a.sum() * 100)} %" for m, v in a.items()], loc="center left",
              bbox_to_anchor=(1, .5), frameon=False, fontsize=9)
    ax.text(0, 0, f"{fr(a.sum() / 1000)} k\nvéhicules", ha="center", va="center", fontsize=10)
    return fig


def barres(d, par, col, unite, ordre=None, pire="max", dec=2, figsize=(4.4, 3.0)):
    a = agreger(d, par).dropna(subset=[col])
    if a.empty:
        return None
    if ordre:
        a = a.set_index(par).reindex([o for o in ordre if o in set(a[par])]).reset_index()
    else:
        a = a.sort_values(col)
    cible = a[col].max() if pire == "max" else a[col].min()
    couleurs = [C["rouge"] if v == cible else C["bleu"] for v in a[col]]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(a[par], a[col], color=couleurs, height=.62)
    for y, v in enumerate(a[col]):
        ax.text(v, y, f" {fr(v, dec)}{unite}", va="center", fontsize=9)
    ax.set_xlim(0, a[col].max() * 1.3)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    if ordre:
        ax.invert_yaxis()
    return fig


def fig_pareto(d):
    c = d[d["cause_arret"] != "Aucun"].groupby("cause_arret")["temps_arret_min"].sum()
    c = c[c > 0].sort_values(ascending=False)
    if c.empty:
        return None
    cum = c.cumsum() / c.sum() * 100
    x = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.bar(x, c.values / 60, color=[C["jaune"] if k == CAUSE_MAINT else C["bleu"] for k in c.index], width=.66)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(k, 12) for k in c.index], fontsize=9)
    ax.set_ylabel("Heures d'arrêt")
    ax2 = ax.twinx()
    ax2.plot(x, cum.values, color=C["rouge"], marker="o", ms=4, lw=1.8)
    ax2.axhline(80, ls=":", color=C["acier"], lw=1)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumul (%)")
    ax2.grid(False)
    return fig


def fig_arrets_equipe(d):
    p = d[d["cause_arret"] != "Aucun"].pivot_table(index="cause_arret", columns="equipe",
                                                  values="temps_arret_min", aggfunc="sum", observed=True)
    if p.empty:
        return None
    p = p.reindex(columns=[e for e in ORDRE_EQUIPES if e in p.columns]).fillna(0) / 60
    p = p.loc[p.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    p.plot(kind="barh", stacked=True, ax=ax, color=[COUL_EQUIPE[e] for e in p.columns], width=.7)
    ax.set_ylabel("")
    ax.set_xlabel("Heures d'arrêt")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, title="")
    return fig


def fig_cycle(d):
    if d["temps_cycle_s"].notna().sum() == 0:
        return None
    ordre = [l for l in ORDRE_LIGNES if l in set(d["ligne"])]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    sns.boxplot(data=d, x="ligne", y="temps_cycle_s", order=ordre, color="#B3AFF0", fliersize=1.5,
                linewidth=1, medianprops=dict(color=C["encre"], linewidth=2), ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Temps de cycle (s)")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", visible=False)
    return fig


# ----------------------------------------------------------------------------
# Constats et recommandations (calculés sur les données filtrées)
# ----------------------------------------------------------------------------
def constats(d):
    res = []
    eq = agreger(d, "equipe").set_index("equipe")
    ln = agreger(d, "ligne").set_index("ligne")
    h = d[~d["maintenance"]]
    trs_eq = h.groupby("equipe")["TRS_pct"].mean()

    if "Nuit" in eq.index and len(eq) > 1 and eq.drop("Nuit")["produits"].sum() > 0:
        autres = eq.drop("Nuit")
        t_autres = autres["defauts"].sum() / autres["produits"].sum() * 100
        txt = (f"Défauts : {fr(eq.loc['Nuit', 'taux_defaut'], 2)} % la nuit contre "
               f"{fr(t_autres, 2)} % pour les autres équipes.")
        if "Nuit" in trs_eq.index and len(trs_eq) > 1:
            trs_autres = trs_eq.drop("Nuit").mean()
            txt += f" TRS hors maintenance : {fr(trs_eq['Nuit'], 1)} % contre {fr(trs_autres, 1)} %."
            if trs_eq["Nuit"] >= trs_autres:
                txt += " L'écart de la nuit porte sur la qualité, pas sur le TRS."
        res.append(("L'équipe de nuit est moins performante",
                    eq["taux_defaut"].idxmax() == "Nuit", txt))

    if "Peinture" in ln.index and len(ln) > 1 and ln.drop("Peinture")["produits"].sum() > 0:
        autres = ln.drop("Peinture")
        t_autres = autres["defauts"].sum() / autres["produits"].sum() * 100
        res.append(("La ligne Peinture a le taux de défauts le plus élevé",
                    ln["taux_defaut"].idxmax() == "Peinture",
                    f"Peinture : {fr(ln.loc['Peinture', 'taux_defaut'], 2)} % de défauts contre "
                    f"{fr(t_autres, 2)} % pour l'ensemble des autres lignes."))

    s = serie_hebdo(d)
    p = pente(s)
    if p is not None:
        k = max(1, min(4, len(s) // 2))
        res.append(("Le taux de défauts baisse avec le temps", p < 0,
                    f"Tendance linéaire : {fr(p * 52 / 12, 2)} point par mois. Moyenne des {k} premières "
                    f"semaines : {fr(s['taux_defaut'].head(k).mean(), 2)} %, des {k} dernières : "
                    f"{fr(s['taux_defaut'].tail(k).mean(), 2)} %."))

    dm = d[(d["jour"] == "Dimanche") & (d["equipe"] == "Matin")]
    if len(dm):
        part = dm["maintenance"].mean() * 100
        t_tout = d[d["equipe"] == "Matin"]["TRS_pct"].mean()
        t_hors = h[h["equipe"] == "Matin"]["TRS_pct"].mean()
        res.append(("Le dimanche matin est réservé à la maintenance préventive", part >= 99,
                    f"{fr(part)} % des créneaux du dimanche matin sont en maintenance, pour "
                    f"{fr(dm['vehicules_produits'].sum())} véhicules produits. Ces créneaux font baisser le "
                    f"TRS apparent de l'équipe du matin à {fr(t_tout, 1)} %, contre {fr(t_hors, 1)} % "
                    f"hors maintenance."))
    return res


def recommandations(d):
    ln = agreger(d, "ligne")
    eq = agreger(d, "equipe")
    rec = []
    if len(ln) > 1:
        pire = ln.loc[ln["taux_defaut"].idxmax()]
        rec.append((f"Traiter la ligne {pire['ligne']} en priorité",
                    f"{fr(pire['taux_defaut'], 2)} % de défauts. Un contrôle qualité en sortie de poste et "
                    f"un suivi des paramètres de process limiteraient les reprises."))
    if len(eq) > 1:
        pire = eq.loc[eq["taux_defaut"].idxmax()]
        rec.append((f"Renforcer l'équipe {pire['equipe']}",
                    f"{fr(pire['taux_defaut'], 2)} % de défauts. Encadrement, formation et passation de "
                    f"consignes entre équipes sont à examiner."))
    c = d[(~d["maintenance"]) & (d["cause_arret"] != "Aucun")].groupby("cause_arret")["temps_arret_min"].sum()
    c = c.sort_values(ascending=False)
    if len(c) and c.sum() > 0:
        rec.append(("Réduire les arrêts non planifiés",
                    f"« {c.index[0]} » pèse {fr(c.iloc[0] / c.sum() * 100)} % des minutes d'arrêt hors "
                    f"maintenance. Maintenance conditionnelle et stock de pièces critiques ciblent cette cause."))
    rec.append(("Comparer les équipes hors maintenance planifiée",
                "Le dimanche matin fait baisser le TRS de l'équipe du matin. Un TRS calculé sur le temps "
                "réellement planifié rend la comparaison équitable."))
    if len(ln) > 1:
        e = ln.loc[ln["kwh_veh"].idxmax()]
        rec.append((f"Auditer l'énergie de la ligne {e['ligne']}",
                    f"{fr(e['kwh_veh'], 1)} kWh par véhicule, la plus forte consommation unitaire."))
    rec.append(("Garder le cap sur la qualité",
                "Fixer un seuil d'alerte hebdomadaire sur le taux de défauts pour confirmer que la baisse "
                "observée se poursuit."))
    return rec


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_connexion():
    gauche, droite = st.columns([1.15, 1], gap="large")
    with gauche:
        st.markdown(
            '<div class="marque"><div class="hero-t">Suivi de la production automobile</div>'
            '<p class="hero-s">Tanger et Kénitra, quatre postes de fabrication, trois équipes. '
            'Qualité, rendement, arrêts et énergie dans un seul tableau de bord.</p>'
            '<div class="chaine"><div class="rail"><i class="car"></i></div><div class="postes">'
            '<div class="poste"><span class="pn">Emboutissage</span></div>'
            '<div class="poste"><span class="pn">Ferrage</span></div>'
            '<div class="poste"><span class="pn">Peinture</span></div>'
            '<div class="poste"><span class="pn">Montage final</span></div>'
            '</div></div></div>', unsafe_allow_html=True)
    with droite:
        st.markdown('<div class="login-t">Connexion</div>'
                    '<div class="login-s">Saisissez vos identifiants pour ouvrir le tableau de bord.</div>',
                    unsafe_allow_html=True)
        with st.form("connexion"):
            u = st.text_input("Nom d'utilisateur")
            p = st.text_input("Mot de passe", type="password")
            ok = st.form_submit_button("Se connecter")
        if ok:
            bon_u = hmac.compare_digest(u.strip().casefold().encode(), UTILISATEUR.casefold().encode())
            bon_p = hmac.compare_digest(p.encode(), MOT_DE_PASSE.encode())
            if bon_u and bon_p:
                st.session_state["auth"] = True
                st.rerun()
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")


def filtres(df):
    sb = st.sidebar
    sb.markdown('<p class="side-titre">Filtres</p>', unsafe_allow_html=True)
    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    periode = sb.date_input("Période", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    debut, fin = periode if isinstance(periode, tuple) and len(periode) == 2 else (dmin, dmax)

    def choix(label, options):
        return sb.multiselect(label, options, default=options)

    sites = choix("Site", sorted(df["site"].unique()))
    lignes = choix("Ligne", [l for l in ORDRE_LIGNES if l in set(df["ligne"])])
    equipes = choix("Équipe", ORDRE_EQUIPES)
    modeles = choix("Modèle", sorted(df["modele"].unique()))
    exclure = sb.toggle("Exclure la maintenance planifiée",
                        help="Retire les créneaux de maintenance préventive (dimanche matin) de tous les "
                             "indicateurs et graphiques.")
    sb.divider()
    sb.caption(f"Connecté : {UTILISATEUR}")
    if sb.button("Se déconnecter"):
        st.session_state["auth"] = False
        st.rerun()

    m = (df["date"] >= pd.Timestamp(debut)) & (df["date"] <= pd.Timestamp(fin))
    m &= df["site"].isin(sites) & df["ligne"].isin(lignes) & df["equipe"].isin(equipes) & df["modele"].isin(modeles)
    return df[m], exclure


def en_tete(d_all, d):
    st.markdown(
        f'<div class="hero-t">Production automobile</div>'
        f'<p class="hero-s">Tanger et Kénitra, du {date_fr(d_all["date"].min())} au '
        f'{date_fr(d_all["date"].max())}. {fr(len(d))} créneaux horaires analysés.</p>',
        unsafe_allow_html=True)

    a = agreger(d, "ligne").set_index("ligne")
    pire = a["taux_defaut"].idxmax() if len(a) > 1 else None
    html = '<div class="chaine"><div class="rail"><i class="car"></i></div><div class="postes">'
    for l in ORDRE_LIGNES:
        if l not in a.index:
            continue
        r = a.loc[l]
        cls = "poste alerte" if l == pire else "poste"
        html += (f'<div class="{cls}"><span class="pn">{l}</span><span class="pv">{fr(r["produits"])}</span>'
                 f'<span class="pl">véhicules produits</span>'
                 f'<span class="chip">{fr(r["taux_defaut"], 2)} % de défauts</span>'
                 f'<span class="pt">TRS {fr(r["trs"], 1)} %</span></div>')
    st.markdown(html + '</div></div>', unsafe_allow_html=True)

    k = kpis(d)
    items = [("Véhicules produits", fr(k["prod"]), "", ""),
             ("Taux de défauts", fr(k["taux"], 2), " %", "r"),
             ("TRS moyen", fr(k["trs"], 1), " %", "b"),
             ("Temps d'arrêt", fr(k["arret"]), " h", ""),
             ("Énergie par véhicule", fr(k["kwh"], 1), " kWh", ""),
             ("Cadence cible atteinte", fr(k["cadence"]), " %", "")]
    cases = "".join(f'<div class="k"><span class="kv {c}">{v}<span class="ku">{u}</span></span>'
                    f'<span class="kl">{lab}</span></div>' for lab, v, u, c in items)
    st.markdown(f'<div class="bandeau">{cases}</div>', unsafe_allow_html=True)


def tableau_de_bord():
    df, qualite = charger()
    d_all, exclure = filtres(df)
    d = d_all[~d_all["maintenance"]] if exclure else d_all
    if d.empty:
        st.warning("Aucune donnée pour cette sélection. Élargissez la période ou ajoutez des valeurs "
                   "dans les filtres.")
        st.stop()

    theme_mpl()
    en_tete(d_all, d)
    t1, t2, t3, t4, t5, t6 = st.tabs(["Vue d'ensemble", "Comparaisons", "Arrêts",
                                      "Qualité et énergie", "Conclusion", "Données"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            bloc("Production quotidienne", "Trait épais : moyenne glissante sur 7 jours.")
            montrer(fig_production(d))
        with c2:
            bloc("Taux de défauts par semaine", "Tirets rouges : tendance linéaire.")
            montrer(fig_defauts(d))
        c3, c4 = st.columns([1.8, 1])
        with c3:
            bloc("TRS moyen selon le jour et l'heure", "Les cases rouges signalent un rendement très bas : "
                 "repérez le dimanche matin.")
            montrer(fig_trs_heure(d))
        with c4:
            bloc("Répartition par modèle", "Part des véhicules produits.")
            montrer(fig_donut(d))

    with t2:
        c1, c2, c3 = st.columns(3)
        with c1:
            bloc("Défauts par ligne", "Le rouge marque la valeur la plus élevée.")
            montrer(barres(d, "ligne", "taux_defaut", " %"))
        with c2:
            bloc("Défauts par équipe", "Taux pondéré par la production.")
            montrer(barres(d, "equipe", "taux_defaut", " %", ordre=ORDRE_EQUIPES))
        with c3:
            bloc("Défauts par site", "Comparaison entre les deux usines.")
            montrer(barres(d, "site", "taux_defaut", " %"))
        c4, c5 = st.columns(2)
        with c4:
            bloc("Défauts selon la ligne et l'équipe", "Taux de défauts en %.")
            montrer(carte(pivot_ordre(d, "ligne", "equipe", ORDRE_LIGNES, ORDRE_EQUIPES, "taux_defaut"),
                          CM_DEFAUT, figsize=(6, 3.3)))
        with c5:
            bloc("TRS selon la ligne et l'équipe", "TRS moyen en %.")
            montrer(carte(pivot_ordre(d, "ligne", "equipe", ORDRE_LIGNES, ORDRE_EQUIPES, "trs"),
                          CM_TRS, dec=1, figsize=(6, 3.3)))
        c6, c7 = st.columns(2)
        with c6:
            bloc("TRS par site", "Moyenne simple des créneaux.")
            montrer(barres(d, "site", "trs", " %", pire="min", dec=1))
        with c7:
            bloc("TRS par équipe", "Activez le filtre de maintenance pour comparer à périmètre égal.")
            montrer(barres(d, "equipe", "trs", " %", ordre=ORDRE_EQUIPES, pire="min", dec=1))

    with t3:
        total = d["temps_arret_min"].sum()
        if total > 0:
            maint = d[d["maintenance"]]["temps_arret_min"].sum() / total * 100
            st.info(f"La maintenance préventive représente {fr(maint)} % des minutes d'arrêt. "
                    f"Elle est planifiée : elle apparaît en jaune dans le Pareto.")
        c1, c2 = st.columns([1.5, 1])
        with c1:
            bloc("Pareto des causes d'arrêt", "Barres : heures perdues. Courbe : cumul en %, repère à 80 %.")
            montrer(fig_pareto(d))
        with c2:
            bloc("Causes d'arrêt par équipe", "Heures d'arrêt, classées par cause.")
            montrer(fig_arrets_equipe(d))

    with t4:
        c1, c2, c3 = st.columns(3)
        with c1:
            bloc("Énergie par véhicule", "kWh consommés par véhicule produit.")
            montrer(barres(d, "ligne", "kwh_veh", " kWh", dec=1))
        with c2:
            bloc("Défauts selon le modèle et la ligne", "Taux de défauts en %.")
            montrer(carte(pivot_ordre(d, "modele", "ligne", sorted(d["modele"].unique()), ORDRE_LIGNES,
                                      "taux_defaut"), CM_DEFAUT, figsize=(5.2, 3.2)))
        with c3:
            bloc("Temps de cycle par ligne", "Médiane en trait noir. Vide si la production est nulle.")
            montrer(fig_cycle(d))

    with t5:
        cs = constats(d_all)
        n_ok = sum(1 for _, ok, _ in cs if ok)
        if cs:
            st.markdown(
                f'<div class="synthese"><div class="sv">{n_ok} constats sur {len(cs)} confirmés</div>'
                f'<div class="st">La ligne Peinture, l\'équipe de nuit et les arrêts non planifiés concentrent '
                f'l\'essentiel des pertes. La qualité s\'améliore au fil des semaines.</div></div>',
                unsafe_allow_html=True)
        st.markdown('<div class="titre-s">Ce que disent les données</div>', unsafe_allow_html=True)
        for titre, ok, txt in cs:
            st.markdown(f'<div class="constat {"" if ok else "non"}"><div class="cv">'
                        f'{"Confirmé" if ok else "Non confirmé"}</div><div class="ct">{titre}</div>'
                        f'<div class="ctx">{txt}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="titre-s">Recommandations</div>', unsafe_allow_html=True)
        cartes = "".join(f'<div class="reco"><div class="rt">{t}</div><div class="rx">{x}</div></div>'
                         for t, x in recommandations(d))
        st.markdown(f'<div class="recos">{cartes}</div>', unsafe_allow_html=True)
        with st.expander("Méthode et limites"):
            st.markdown(
                "- Le taux de défauts est calculé en divisant le total des véhicules défectueux par le total "
                "des véhicules produits, et non en moyennant les pourcentages de chaque créneau.\n"
                "- Le TRS est la moyenne simple des créneaux, sans pondération par la durée de production.\n"
                "- Les constats portent sur les filtres actifs et sur environ cinq mois de données.\n"
                "- Une différence entre équipes ou lignes décrit une association, pas une cause prouvée.")

    with t6:
        st.markdown("### Qualité des données")
        st.markdown(
            f"- {fr(qualite['brut'])} lignes lues, {fr(qualite['doublons'])} doublon(s), "
            f"{fr(qualite['incoherences'])} incohérence(s) entre produits, défectueux et conformes.\n"
            f"- {fr(qualite['cycle_vide'])} valeurs vides dans le temps de cycle, dont "
            f"{fr(qualite['cycle_vide_prod_nulle'])} correspondent à une production nulle : elles sont "
            f"conservées vides plutôt que remplacées par zéro.")
        st.download_button("Télécharger la sélection (CSV)",
                           d_all.to_csv(index=False).encode("utf-8-sig"),
                           file_name="production_selection.csv", mime="text/csv")
        st.dataframe(d_all.drop(columns=["semaine", "maintenance", "jour_semaine"]))


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    if st.session_state.get("auth"):
        tableau_de_bord()
    else:
        page_connexion()


if __name__ == "__main__":
    main()
