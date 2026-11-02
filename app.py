"""
Dashboard de suivi de la production automobile (Tanger et Kénitra)
Étudiant : MEGGOURI Amine  |  Données : dataset.csv

Lancer :  streamlit run app.py      (identifiant : Amine / mot de passe : 0000)
"""
from __future__ import annotations

import hmac
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Production automobile",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================================================================
# 1. CONSTANTES : identifiants, ordres, palette
# ====================================================================
CHEMIN_DONNEES = Path(__file__).parent / "dataset.csv"
IDENTIFIANT = "amine"      # comparé sans tenir compte de la casse
MOT_DE_PASSE = "0000"

ORDRE_LIGNES = ["Emboutissage", "Ferrage", "Peinture", "Montage final"]
ORDRE_EQUIPES = ["Matin", "Après-midi", "Nuit"]
JOURS_FR = {
    "Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
    "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi",
    "Sunday": "Dimanche",
}
ORDRE_JOURS = list(JOURS_FR.values())
CAUSE_PLANIFIEE = "Maintenance préventive"

MARINE, SARCELLE, VERMILLON = "#12355B", "#1F7A8C", "#E4572E"
AMBRE, GRIS, GRIS_CLAIR = "#F2A541", "#7A8B99", "#C9D2DA"

COUL_EQUIPE = {"Matin": AMBRE, "Après-midi": SARCELLE, "Nuit": MARINE}
COUL_LIGNE = {"Emboutissage": SARCELLE, "Ferrage": AMBRE,
              "Peinture": VERMILLON, "Montage final": MARINE}
COUL_SITE = {"Tanger": SARCELLE, "Kénitra": MARINE}
COUL_MODELE = [SARCELLE, AMBRE, MARINE, VERMILLON]

# Métriques disponibles pour les comparaisons : libellé -> (clé, décimales)
METRIQUES = {
    "Production (véhicules)": ("production", 0),
    "Taux de défauts (%)": ("taux_defaut", 2),
    "TRS moyen (%)": ("trs", 1),
    "Arrêts subis (heures)": ("arret_subi_h", 0),
    "Énergie par véhicule (kWh)": ("energie_veh", 2),
    "Atteinte de la cadence (%)": ("atteinte", 1),
}

# ====================================================================
# 2. STYLE (CSS + Matplotlib/Seaborn)
# ====================================================================
CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.bandeau {background: #12355B; color: #fff; padding: 22px 28px;
          border-radius: 10px; margin-bottom: 18px;}
.bandeau .b-titre {font-size: 1.7rem; font-weight: 700; line-height: 1.2;}
.bandeau .b-sous {font-size: .95rem; opacity: .85; margin-top: 4px;}
.kpi {background: #fff; border: 1px solid #E1E7ED; border-left: 6px solid var(--c);
      border-radius: 8px; padding: 14px 16px; height: 100%;}
.kpi .t {font-size: .82rem; color: #556577;}
.kpi .v {font-size: 1.85rem; font-weight: 700; color: #12355B; line-height: 1.25;}
.kpi .s {font-size: .78rem; color: #556577;}
.constat {background: #fff; border: 1px solid #E1E7ED; border-radius: 8px;
          padding: 14px 18px; margin-bottom: 12px;}
.constat .verdict-ok {color: #1D7A46; font-weight: 700;}
.constat .verdict-ko {color: #B3261E; font-weight: 700;}
.connexion-titre {font-size: 1.6rem; font-weight: 700; color: #12355B; margin-bottom: 2px;}
.connexion-sous {color: #556577; margin-bottom: 14px;}
</style>
"""


def style_graphiques() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRIS_CLAIR, "axes.labelcolor": "#33475B",
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": "#E9EEF2",
        "xtick.color": "#33475B", "ytick.color": "#33475B",
        "font.size": 9.5, "legend.frameon": False,
    })


def afficher(fig) -> None:
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def fr(x, nd: int = 0) -> str:
    """Format français : 1 234,5"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


# ====================================================================
# 3. CHARGEMENT ET NETTOYAGE
# ====================================================================
@st.cache_data(show_spinner="Chargement et nettoyage des données…")
def charger_donnees(chemin: str):
    """Lit le CSV, le nettoie et renvoie (df propre, rapport de nettoyage)."""
    brut = pd.read_csv(chemin, encoding="utf-8-sig")
    df = brut.copy()
    rapport = []

    def noter(etape, constat, action):
        rapport.append({"Étape": etape, "Constat": constat, "Action": action})

    noter("Lecture", f"{len(brut):,} lignes × {brut.shape[1]} colonnes".replace(",", " "),
          "Lecture avec pandas (encodage utf-8-sig)")

    # -- espaces parasites dans le texte
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip().astype("object")

    # -- doublons
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    noter("Doublons", f"{n_dup} ligne(s) dupliquée(s)", "Suppression" if n_dup else "Aucune action")

    # -- dates
    df["date_heure"] = pd.to_datetime(df["date_heure"])
    df["date"] = df["date_heure"].dt.normalize()
    noter("Dates", f"Période du {df.date.min():%d/%m/%Y} au {df.date.max():%d/%m/%Y}",
          "Conversion de date_heure et date en datetime")

    # -- libellé tronqué dans la source
    n_lib = int(df["cause_arret"].str.startswith("Maintenance").sum())
    df["cause_arret"] = df["cause_arret"].where(
        ~df["cause_arret"].str.startswith("Maintenance"), CAUSE_PLANIFIEE)
    noter("Libellés", f"« Maintenance prévent » tronqué dans {n_lib} lignes",
          f"Renommé en « {CAUSE_PLANIFIEE} »")

    # -- jours en français + ordres des catégories
    df["jour_semaine"] = pd.Categorical(df["jour_semaine"].map(JOURS_FR),
                                        categories=ORDRE_JOURS, ordered=True)
    df["equipe"] = pd.Categorical(df["equipe"], categories=ORDRE_EQUIPES, ordered=True)
    df["ligne"] = pd.Categorical(df["ligne"], categories=ORDRE_LIGNES, ordered=True)
    noter("Catégories", "Jours en anglais, ordre alphabétique des lignes",
          "Jours traduits ; lignes et équipes ordonnées (flux de production)")

    # -- valeurs manquantes
    n_nan = int(df["temps_cycle_s"].isna().sum())
    n_nan_prod0 = int((df["temps_cycle_s"].isna() & (df["vehicules_produits"] == 0)).sum())
    autres_nan = int(df.drop(columns="temps_cycle_s").isna().sum().sum())
    noter("Valeurs manquantes",
          f"temps_cycle_s vide dans {n_nan} lignes, dont {n_nan_prod0} avec production nulle ; "
          f"{autres_nan} autre(s) valeur(s) manquante(s)",
          "Conservées (NaN) : un temps de cycle n'existe pas sans production. "
          "Pas d'imputation, les moyennes ignorent les NaN")

    # -- cohérence des colonnes
    ecart_conf = int((df.vehicules_produits - df.vehicules_defectueux != df.vehicules_conformes).sum())
    taux_calc = (df.vehicules_defectueux / df.vehicules_produits.replace(0, np.nan) * 100).fillna(0)
    ecart_taux = int((~np.isclose(taux_calc, df.taux_defaut_pct, atol=0.02)).sum())
    def_sup = int((df.vehicules_defectueux > df.vehicules_produits).sum())
    prod_sup = int((df.vehicules_produits > df.cadence_cible_h).sum())
    negatifs = int((df.select_dtypes("number") < 0).sum().sum())
    noter("Cohérence",
          f"conformes ≠ produits − défectueux : {ecart_conf} ; taux de défaut recalculé différent : "
          f"{ecart_taux} ; défectueux > produits : {def_sup} ; produits > cadence cible : {prod_sup} ; "
          f"valeurs négatives : {negatifs}",
          "Aucune correction nécessaire" if not (ecart_conf or ecart_taux or def_sup or negatifs)
          else "À examiner")

    # -- valeurs extrêmes
    ext = df[df.taux_defaut_pct > 20]
    noter("Valeurs extrêmes",
          f"{len(ext)} lignes avec un taux de défaut > 20 % (maximum {df.taux_defaut_pct.max():.0f} %), "
          f"surtout sur de faibles volumes (médiane {ext.vehicules_produits.median():.1f} véhicules par heure "
          f"contre {df.vehicules_produits.median():.0f} pour l'ensemble)",
          "Conservées (mesures réelles). Les KPI utilisent un ratio de sommes "
          "(total défectueux / total produit) et non la moyenne des taux horaires")

    # -- jours incomplets (ex. dernier jour du fichier)
    df["jour_complet"] = df.groupby("date")["heure"].transform("nunique") == 24
    incomplets = df.loc[~df.jour_complet, "date"].dt.strftime("%d/%m/%Y").unique().tolist()
    noter("Jours incomplets",
          f"{len(incomplets)} jour(s) avec moins de 24 heures de données : {', '.join(incomplets) or 'aucun'}",
          "Conservés dans les totaux, mais exclus des courbes de production (ils feraient chuter la tendance)")

    # -- variables dérivées
    df["arret_planifie"] = df["cause_arret"] == CAUSE_PLANIFIEE
    df["semaine"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    df["mois"] = df["date"].dt.to_period("M").dt.to_timestamp()
    noter("Variables créées", "arret_planifie, jour_complet, semaine, mois",
          "Permettent de séparer les arrêts planifiés des arrêts subis et d'agréger dans le temps")

    return df, pd.DataFrame(rapport), brut.isna().sum()


# ====================================================================
# 4. KPI ET OUTILS STATISTIQUES
# ====================================================================
def kpis(d: pd.DataFrame, exclure_planifie: bool = True) -> dict:
    """KPI sur un sous-ensemble. TRS et cadence : hors maintenance planifiée si demandé."""
    if d.empty:
        return {k: np.nan for k, _ in METRIQUES.values()} | {"arret_planifie_h": np.nan, "arret_moy_min": np.nan}
    prod = d.vehicules_produits.sum()
    perf = d[~d.arret_planifie] if exclure_planifie else d
    subi = d[~d.arret_planifie]
    return {
        "production": prod,
        "taux_defaut": d.vehicules_defectueux.sum() / prod * 100 if prod else np.nan,
        "trs": perf.TRS_pct.mean() if len(perf) else np.nan,
        "arret_subi_h": subi.temps_arret_min.sum() / 60,
        "arret_planifie_h": d[d.arret_planifie].temps_arret_min.sum() / 60,
        "arret_moy_min": subi.temps_arret_min.mean() if len(subi) else np.nan,
        "energie_veh": d.energie_kwh.sum() / prod if prod else np.nan,
        "atteinte": perf.vehicules_produits.sum() / perf.cadence_cible_h.sum() * 100
        if len(perf) else np.nan,
    }


def comparer(d: pd.DataFrame, cle: str, excl: bool) -> pd.DataFrame:
    return pd.DataFrame({k: kpis(g, excl) for k, g in d.groupby(cle, observed=True)}).T


def croiser(d: pd.DataFrame, ligne: str, colonne: str, cle: str, excl: bool) -> pd.DataFrame:
    res = {}
    for (a, b), g in d.groupby([ligne, colonne], observed=True):
        res[(a, b)] = kpis(g, excl)[cle]
    s = pd.Series(res)
    s.index = pd.MultiIndex.from_tuples(s.index)
    return s.unstack()


def test_deux_proportions(x1, n1, x2, n2):
    """Test z de comparaison de deux proportions (approximation normale)."""
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (x1 / n1 - x2 / n2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def txt_p(p: float) -> str:
    return "p &lt; 0,001" if p < 0.001 else f"p = {p:.3f}".replace(".", ",")


def tendance_defauts(d: pd.DataFrame):
    """Taux de défauts hebdomadaire (semaines complètes) + droite de régression."""
    sem = d.groupby("semaine", observed=True).agg(
        produits=("vehicules_produits", "sum"),
        defauts=("vehicules_defectueux", "sum"),
        jours=("date", "nunique"),
    )
    sem = sem[sem.jours >= 7]
    if len(sem) < 3:
        return None
    sem["taux"] = sem.defauts / sem.produits * 100
    x = np.arange(len(sem))
    pente, ordonnee = np.polyfit(x, sem.taux.values, 1)
    r = float(np.corrcoef(x, sem.taux.values)[0, 1])
    return sem, pente, ordonnee, r


def carte(titre: str, valeur: str, sous_titre: str, couleur: str) -> str:
    return (f'<div class="kpi" style="--c:{couleur}"><div class="t">{titre}</div>'
            f'<div class="v">{valeur}</div><div class="s">{sous_titre}</div></div>')


# ====================================================================
# 5. GRAPHIQUES
# ====================================================================
def barres(ax, t: pd.DataFrame, cle: str, nd: int, couleurs: dict) -> None:
    labels = [str(i) for i in t.index]
    vals = t[cle].astype(float).values
    barres_ = ax.bar(labels, vals, color=[couleurs.get(l, GRIS) for l in labels], width=0.62)
    ax.bar_label(barres_, labels=[fr(v, nd) for v in vals], padding=3, fontsize=9)
    ax.margins(y=0.15)
    ax.grid(axis="x", visible=False)
    if max(len(l) for l in labels) > 9:
        ax.tick_params(axis="x", rotation=18)


def graphique_pareto(minutes: pd.Series, titre: str):
    s = minutes[minutes > 0].sort_values(ascending=False)
    if s.empty:
        return None, []
    cumul = s.cumsum() / s.sum() * 100
    precedent = [0.0] + list(cumul.values[:-1])
    vitales = [c for c, p in zip(s.index, precedent) if p < 80]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    pos = np.arange(len(s))
    ax.bar(pos, s.values / 60, color=[VERMILLON if c in vitales else GRIS_CLAIR for c in s.index], width=0.65)
    ax.set_ylabel("Heures d'arrêt")
    ax.set_xticks(pos)
    ax.set_xticklabels(s.index, rotation=25, ha="right")
    ax.grid(axis="x", visible=False)
    ax2 = ax.twinx()
    ax2.plot(pos, cumul.values, color=MARINE, marker="o", lw=2)
    ax2.axhline(80, color=MARINE, ls="--", lw=1)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumul (%)")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    for x, v in zip(pos, cumul.values):
        ax2.annotate(f"{v:.0f} %", (x, v), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=8.5, color=MARINE)
    ax.set_title(titre)
    return fig, vitales


def cartes_kpi(d: pd.DataFrame, excl: bool) -> None:
    k = kpis(d, excl)
    base = "hors maintenance planifiée" if excl else "maintenance planifiée incluse"
    cols = st.columns(6)
    contenu = [
        ("Production totale", fr(k["production"]), "véhicules produits", SARCELLE),
        ("Taux de défauts", f'{fr(k["taux_defaut"], 2)} %', "défectueux / produits", VERMILLON),
        ("TRS moyen", f'{fr(k["trs"], 1)} %', base, MARINE),
        ("Arrêts subis", f'{fr(k["arret_subi_h"])} h',
         f'+ {fr(k["arret_planifie_h"])} h de maintenance planifiée', AMBRE),
        ("Énergie par véhicule", f'{fr(k["energie_veh"], 2)} kWh', "énergie totale / véhicules", GRIS),
        ("Atteinte de la cadence", f'{fr(k["atteinte"], 1)} %', "production / cadence cible", SARCELLE),
    ]
    for col, (t, v, s, c) in zip(cols, contenu):
        col.markdown(carte(t, v, s, c), unsafe_allow_html=True)


# ====================================================================
# 6. PAGES
# ====================================================================
def page_vue_ensemble(d, excl):
    cartes_kpi(d, excl)
    st.write("")

    g1, g2 = st.columns([3, 2])
    with g1:
        gran = st.radio("Granularité de la tendance", ["Jour", "Semaine", "Mois"], horizontal=True)
        freq = {"Jour": "D", "Semaine": "W-SUN", "Mois": "MS"}[gran]
        complets = d[d.jour_complet]
        serie = (complets if len(complets) else d).set_index("date").vehicules_produits.resample(freq).sum()
        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        ax.plot(serie.index, serie.values, color=GRIS_CLAIR if gran == "Jour" else SARCELLE,
                lw=1.2 if gran == "Jour" else 2.2, marker="" if gran == "Jour" else "o",
                label="Production")
        if gran == "Jour" and len(serie) >= 7:
            ax.plot(serie.index, serie.rolling(7, center=True, min_periods=4).mean(),
                    color=SARCELLE, lw=2.4, label="Moyenne mobile 7 jours")
            ax.legend(loc="lower right")
        ax.set_title(f"Production par {gran.lower()}")
        ax.set_ylabel("Véhicules")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        afficher(fig)
        if (~d.jour_complet).any():
            st.caption("Les jours incomplets (données partielles en fin de fichier) sont exclus de cette courbe. "
                       "Le dernier mois ou la dernière semaine peut rester partiel.")
    with g2:
        st.write("")
        st.write("")
        par_modele = d.groupby("modele").vehicules_produits.sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(4.6, 3.9))
        ax.pie(par_modele.values, labels=par_modele.index, colors=COUL_MODELE[: len(par_modele)],
               autopct=lambda p: f"{p:.0f} %", pctdistance=0.78, startangle=90, counterclock=False,
               wedgeprops=dict(width=0.42, edgecolor="white"), textprops=dict(fontsize=9))
        ax.set_title("Répartition par modèle")
        afficher(fig)

    st.subheader("Rythme de production sur la semaine")
    piv = (d.groupby(["jour_semaine", "heure"], observed=True).vehicules_produits.mean()
           .unstack().reindex(ORDRE_JOURS).dropna(how="all"))
    fig, ax = plt.subplots(figsize=(12, 3.4))
    sns.heatmap(piv, cmap="YlGnBu", ax=ax, linewidths=0.4, linecolor="white",
                cbar_kws={"label": "Véhicules par ligne et par heure"})
    ax.set_xlabel("Heure de la journée")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    ax.set_title("Production moyenne selon le jour et l'heure")
    afficher(fig)
    st.caption("Le bloc vide du dimanche de 6 h à 13 h correspond à la maintenance préventive planifiée.")


def page_performance(d, excl):
    st.caption("Comparez les sites, les lignes et les équipes sur l'indicateur de votre choix.")
    libelle = st.selectbox("Indicateur", list(METRIQUES))
    cle, nd = METRIQUES[libelle]

    colonnes = st.columns(3)
    for col, (cle_dim, titre, coul) in zip(colonnes, [
            ("site", "Par site", COUL_SITE), ("ligne", "Par ligne", COUL_LIGNE),
            ("equipe", "Par équipe", COUL_EQUIPE)]):
        t = comparer(d, cle_dim, excl)
        with col:
            fig, ax = plt.subplots(figsize=(4.6, 3.6))
            barres(ax, t, cle, nd, coul)
            ax.set_title(titre)
            afficher(fig)

    g1, g2 = st.columns(2)
    with g1:
        m = croiser(d, "equipe", "ligne", cle, excl)
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        sns.heatmap(m, annot=True, fmt=f".{nd}f", cmap="YlOrRd" if cle in ("taux_defaut", "arret_subi_h", "energie_veh") else "YlGnBu",
                    linewidths=0.5, linecolor="white", ax=ax, cbar=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"{libelle} : équipe × ligne")
        afficher(fig)
    with g2:
        base = d[~d.arret_planifie] if excl else d
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        sns.boxplot(data=base, x="equipe", y="TRS_pct", hue="equipe", palette=COUL_EQUIPE,
                    legend=False, ax=ax, fliersize=1.5, linewidth=1)
        ax.set_xlabel("")
        ax.set_ylabel("TRS horaire (%)")
        ax.set_title("Dispersion du TRS par équipe")
        afficher(fig)

    st.subheader("Tableau comparatif")
    onglets = st.tabs(["Sites", "Lignes", "Équipes"])
    noms = {"production": "Production", "taux_defaut": "Taux défauts (%)", "trs": "TRS (%)",
            "arret_subi_h": "Arrêts subis (h)", "arret_planifie_h": "Maint. planifiée (h)",
            "energie_veh": "kWh / véhicule", "atteinte": "Cadence atteinte (%)"}
    for onglet, cle_dim in zip(onglets, ["site", "ligne", "equipe"]):
        with onglet:
            t = comparer(d, cle_dim, excl)[list(noms)].astype(float).round(2)
            st.dataframe(t.rename(columns=noms))


def page_qualite(d, excl):
    res = tendance_defauts(d)
    g1, g2 = st.columns([3, 2])
    with g1:
        if res is None:
            st.info("Sélectionnez au moins 3 semaines complètes pour afficher la tendance.")
        else:
            sem, pente, ordonnee, r = res
            fig, ax = plt.subplots(figsize=(8.5, 3.8))
            ax.plot(sem.index, sem.taux, marker="o", color=VERMILLON, lw=2, label="Taux hebdomadaire")
            x = np.arange(len(sem))
            ax.plot(sem.index, ordonnee + pente * x, ls="--", color=MARINE, lw=2, label="Tendance linéaire")
            ax.set_ylabel("Taux de défauts (%)")
            ax.set_title("Évolution hebdomadaire du taux de défauts")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            ax.legend()
            afficher(fig)
            sens = "baisse" if pente < 0 else "hausse"
            st.caption(f"Tendance en {sens} de {fr(abs(pente) * 4.345, 2)} point de % par mois "
                       f"(r = {fr(r, 2)}, R² = {fr(r * r, 2)}, {len(sem)} semaines complètes).")
    with g2:
        t = comparer(d, "ligne", excl)
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        barres(ax, t, "taux_defaut", 2, COUL_LIGNE)
        ax.set_title("Taux de défauts par ligne (%)")
        afficher(fig)

    g3, g4 = st.columns(2)
    with g3:
        m = croiser(d, "equipe", "ligne", "taux_defaut", excl)
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        sns.heatmap(m, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, linecolor="white",
                    ax=ax, cbar_kws={"label": "%"})
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("Taux de défauts : équipe × ligne")
        afficher(fig)
    with g4:
        t = comparer(d, "jour_semaine", excl)
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        ax.plot(t.index.astype(str), t.taux_defaut.astype(float), marker="o", color=VERMILLON, lw=2)
        ax.set_ylabel("Taux de défauts (%)")
        ax.set_title("Taux de défauts selon le jour")
        ax.tick_params(axis="x", rotation=25)
        afficher(fig)


def page_arrets(d, excl):
    inclure = st.checkbox("Inclure la maintenance préventive planifiée dans le Pareto", value=False)
    base = d if inclure else d[~d.arret_planifie]
    minutes = base[base.cause_arret != "Aucun"].groupby("cause_arret").temps_arret_min.sum()
    titre = "Pareto des causes d'arrêt" + ("" if inclure else " (arrêts subis)")
    fig, vitales = graphique_pareto(minutes, titre)
    if fig is None:
        st.info("Aucun arrêt dans la sélection.")
        return
    afficher(fig)
    st.caption("En rouge : les causes qui expliquent environ 80 % du temps d'arrêt. Causes prioritaires : "
               + ", ".join(vitales) + ".")

    g1, g2 = st.columns(2)
    with g1:
        parts = pd.crosstab(d.equipe, d.cause_arret, normalize="index") * 100
        ordre = ["Aucun"] + [c for c in parts.columns if c != "Aucun"]
        parts = parts[ordre]
        palette = sns.color_palette("Set2", len(ordre))
        couleurs = [GRIS_CLAIR] + list(palette[1:])
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        parts.plot(kind="barh", stacked=True, ax=ax, color=couleurs, width=0.7)
        ax.set_xlabel("Part des heures-ligne (%)")
        ax.set_ylabel("")
        ax.invert_yaxis()
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
        ax.set_title("Situation des heures de production par équipe")
        afficher(fig)
    with g2:
        subi = d[~d.arret_planifie]
        par_heure = subi.groupby("heure").agg(arret=("temps_arret_min", "mean"))
        equipe_h = subi.groupby("heure").equipe.agg(lambda s: s.mode().iat[0])
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.bar(par_heure.index, par_heure.arret, color=[COUL_EQUIPE[str(e)] for e in equipe_h], width=0.8)
        ax.set_xlabel("Heure de la journée")
        ax.set_ylabel("Minutes d'arrêt par ligne-heure")
        ax.set_xticks(range(0, 24, 2))
        ax.set_title("Arrêts subis selon l'heure")
        ax.grid(axis="x", visible=False)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COUL_EQUIPE.values()]
        ax.legend(handles, COUL_EQUIPE.keys(), loc="upper right", ncol=3, fontsize=8)
        afficher(fig)

    st.subheader("Causes d'arrêt par ligne (heures)")
    m = (base[base.cause_arret != "Aucun"].pivot_table(index="cause_arret", columns="ligne",
         values="temps_arret_min", aggfunc="sum", observed=True) / 60)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    sns.heatmap(m, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": "heures"})
    ax.set_xlabel("")
    ax.set_ylabel("")
    afficher(fig)


def page_energie(d, excl):
    k = kpis(d, excl)
    arret_total = d[d.vehicules_produits == 0]
    c1, c2, c3 = st.columns(3)
    c1.markdown(carte("Énergie par véhicule", f'{fr(k["energie_veh"], 2)} kWh', "toutes lignes confondues", SARCELLE),
                unsafe_allow_html=True)
    c2.markdown(carte("Énergie totale", f'{fr(d.energie_kwh.sum() / 1000, 0)} MWh', "sur la période filtrée", MARINE),
                unsafe_allow_html=True)
    c3.markdown(carte("Énergie consommée sans production", f'{fr(arret_total.energie_kwh.sum() / 1000, 1)} MWh',
                      f"{len(arret_total)} heures-ligne à production nulle", AMBRE), unsafe_allow_html=True)
    st.write("")

    g1, g2, g3 = st.columns(3)
    for col, (cle_dim, titre, coul) in zip([g1, g2, g3], [
            ("ligne", "kWh par véhicule, par ligne", COUL_LIGNE),
            ("site", "kWh par véhicule, par site", COUL_SITE),
            ("equipe", "kWh par véhicule, par équipe", COUL_EQUIPE)]):
        with col:
            fig, ax = plt.subplots(figsize=(4.6, 3.6))
            barres(ax, comparer(d, cle_dim, excl), "energie_veh", 2, coul)
            ax.set_title(titre)
            afficher(fig)

    mens = d.groupby(["mois", "ligne"], observed=True).agg(e=("energie_kwh", "sum"), p=("vehicules_produits", "sum"))
    mens["kwh_veh"] = mens.e / mens.p
    mens = mens.reset_index()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    sns.lineplot(data=mens, x="mois", y="kwh_veh", hue="ligne", palette=COUL_LIGNE, marker="o", ax=ax)
    ax.set_ylabel("kWh par véhicule")
    ax.set_xlabel("")
    ax.set_title("Évolution mensuelle de l'intensité énergétique")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.legend(title="")
    afficher(fig)


def page_donnees(d, df, rapport, manquantes):
    st.subheader("Rapport de nettoyage")
    st.dataframe(rapport, hide_index=True)
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Valeurs manquantes (fichier brut)")
        m = manquantes[manquantes > 0]
        if m.empty:
            st.success("Aucune valeur manquante.")
        else:
            fig, ax = plt.subplots(figsize=(5.5, 2.4))
            ax.barh(m.index, m.values, color=AMBRE)
            ax.bar_label(ax.containers[0], padding=3)
            ax.set_title("Nombre de lignes vides par colonne")
            afficher(fig)
    with g2:
        st.subheader("Types de données (après nettoyage)")
        st.dataframe(pd.DataFrame({"Type": df.dtypes.astype(str), "Valeurs uniques": df.nunique()}))
    st.subheader("Aperçu des données filtrées")
    st.dataframe(d.drop(columns=["semaine", "mois", "jour_complet"]).head(500))
    st.subheader("Statistiques descriptives")
    st.dataframe(d.select_dtypes("number").describe().T.round(2))
    st.download_button("Télécharger les données filtrées (CSV)",
                       d.to_csv(index=False, sep=";").encode("utf-8-sig"),
                       file_name="production_filtree.csv", mime="text/csv")


def page_conclusion(df):
    st.caption("Les conclusions portent sur l'ensemble de la période, indépendamment des filtres. "
               "TRS et cadence sont calculés hors maintenance préventive planifiée.")
    hors_maint = df[~df.arret_planifie]
    nuit, reste = df[df.equipe == "Nuit"], df[df.equipe != "Nuit"]
    k_n, k_r = kpis(nuit), kpis(reste)
    z_n, p_n = test_deux_proportions(nuit.vehicules_defectueux.sum(), nuit.vehicules_produits.sum(),
                                     reste.vehicules_defectueux.sum(), reste.vehicules_produits.sum())
    ok1 = k_n["trs"] < k_r["trs"] and k_n["taux_defaut"] > k_r["taux_defaut"] and p_n < 0.05
    trs_matin_brut = df[df.equipe == "Matin"].TRS_pct.mean()
    part_aucun = (hors_maint.groupby("equipe", observed=True).cause_arret
                  .apply(lambda s: (s == "Aucun").mean() * 100))

    par_ligne = comparer(df, "ligne", True)
    pire = par_ligne.taux_defaut.astype(float).idxmax()
    pe, autres = df[df.ligne == "Peinture"], df[df.ligne != "Peinture"]
    z_p, p_p = test_deux_proportions(pe.vehicules_defectueux.sum(), pe.vehicules_produits.sum(),
                                     autres.vehicules_defectueux.sum(), autres.vehicules_produits.sum())
    ok2 = pire == "Peinture" and p_p < 0.05

    res = tendance_defauts(df)
    sem, pente, _, r = res
    ok3 = pente < 0 and r < -0.5
    mensuel = comparer(df, "mois", True).taux_defaut.astype(float)

    maint = df[df.arret_planifie]
    dim_matin = (maint.jour_semaine == "Dimanche").all() and maint.heure.between(6, 13).all()
    sans_prod = (maint.vehicules_produits == 0).all()
    ok4 = bool(dim_matin and sans_prod)
    h_plan = maint.temps_arret_min.sum() / 60
    part_plan = maint.temps_arret_min.sum() / df.temps_arret_min.sum() * 100
    capacite_perdue = maint.cadence_cible_h.sum()
    part_capacite = capacite_perdue / df.vehicules_produits.sum() * 100

    def verdict(ok):
        return ('<span class="verdict-ok">Confirmé</span>' if ok
                else '<span class="verdict-ko">Non confirmé</span>')

    st.subheader("Ce que révèlent les données")
    st.markdown(f"""<div class="constat">{verdict(ok1)} · <b>L'équipe de nuit est moins performante.</b><br>
TRS de {fr(k_n["trs"], 1)} % contre {fr(k_r["trs"], 1)} % pour les deux autres équipes ; taux de défauts de
{fr(k_n["taux_defaut"], 2)} % contre {fr(k_r["taux_defaut"], 2)} % ({txt_p(p_n)}, test z de comparaison de proportions) ;
{fr(k_n["arret_moy_min"], 1)} min d'arrêt subi par ligne-heure contre {fr(k_r["arret_moy_min"], 1)} min.
La nuit n'a une heure sans aucun arrêt que dans {fr(part_aucun["Nuit"], 0)} % des cas, contre
{fr(part_aucun["Matin"], 0)} % le matin.<br>
<i>Piège de lecture :</i> en incluant la maintenance du dimanche matin (TRS = 0), le TRS du Matin tombe à
{fr(trs_matin_brut, 1)} % et semble le pire. C'est un artefact : il faut isoler les arrêts planifiés.</div>""",
                unsafe_allow_html=True)
    st.markdown(f"""<div class="constat">{verdict(ok2)} · <b>La ligne Peinture a le taux de défauts le plus élevé.</b><br>
{fr(par_ligne.taux_defaut[pire], 2)} % pour la ligne {pire}, contre
{fr(autres.vehicules_defectueux.sum() / autres.vehicules_produits.sum() * 100, 2)} % pour l'ensemble des autres lignes
({txt_p(p_p)}). Elle consomme aussi le plus d'énergie par véhicule
({fr(kpis(pe)["energie_veh"], 1)} kWh).</div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="constat">{verdict(ok3)} · <b>Le taux de défauts baisse avec le temps.</b><br>
De {fr(sem.taux.iloc[0], 2)} % (première semaine complète) à {fr(sem.taux.iloc[-1], 2)} % (dernière) ;
pente de {fr(pente * 4.345, 2)} point de % par mois (r = {fr(r, 2)}, {len(sem)} semaines).
Mensuellement : {fr(mensuel.iloc[0], 2)} % en mars, {fr(mensuel.iloc[-2], 2)} % en juillet.
La corrélation décrit une évolution, elle n'identifie pas la cause de l'amélioration.</div>""",
                unsafe_allow_html=True)
    st.markdown(f"""<div class="constat">{verdict(ok4)} · <b>Le dimanche matin est réservé à la maintenance préventive.</b><br>
{fr(len(maint))} heures-ligne de maintenance préventive, toutes le dimanche entre 6 h et 13 h, avec production nulle
partout. Cela représente {fr(h_plan)} h d'arrêt, soit {fr(part_plan, 0)} % du temps d'arrêt total, et environ
{fr(capacite_perdue)} véhicules de capacité théorique non utilisée (l'équivalent de {fr(part_capacite, 1)} %
de la production réalisée).</div>""", unsafe_allow_html=True)

    subi = hors_maint[hors_maint.cause_arret != "Aucun"].groupby("cause_arret").temps_arret_min.sum().sort_values(ascending=False)
    top2 = subi.head(2)
    part_top2 = top2.sum() / subi.sum() * 100
    pn = (nuit[nuit.cause_arret != CAUSE_PLANIFIEE].cause_arret.value_counts(normalize=True) * 100)
    pr = (reste[reste.cause_arret != CAUSE_PLANIFIEE].cause_arret.value_counts(normalize=True) * 100)
    taux_pe_nuit = kpis(pe[pe.equipe == "Nuit"])["taux_defaut"]
    taux_pe_jour = kpis(pe[pe.equipe != "Nuit"])["taux_defaut"]
    k_pe = kpis(pe)["energie_veh"]
    k_mo = kpis(df[df.ligne == "Montage final"])["energie_veh"]
    vide = df[df.vehicules_produits == 0].energie_kwh.mean()

    st.subheader("Recommandations")
    st.markdown(f"""
1. **Traiter le poste de nuit en priorité.** Les arrêts y sont fréquents : panne machine dans
   {fr(pn.get("Panne machine", 0), 0)} % des heures contre {fr(pr.get("Panne machine", 0), 0)} % pour les autres équipes.
   Actions : maintenance de premier niveau avant le poste, présence d'un référent maintenance et qualité,
   passation de consignes structurée. Objectif : ramener le TRS de nuit au niveau des autres équipes
   (+{fr(k_r["trs"] - k_n["trs"], 1)} point).
2. **Fiabiliser la ligne Peinture, surtout la nuit.** Son taux de défauts est de {fr(taux_pe_nuit, 2)} % la nuit
   contre {fr(taux_pe_jour, 2)} % le reste du temps. Suivre la maîtrise statistique du procédé (température,
   humidité, poussières en cabine) et analyser les défauts par type pour trouver l'origine de l'écart.
3. **Attaquer le Pareto des arrêts subis.** « {top2.index[0]} » et « {top2.index[1]} » représentent
   {fr(part_top2, 0)} % des minutes d'arrêt subi : maintenance conditionnelle sur les équipements critiques et
   sécurisation de l'approvisionnement en pièces (stocks tampons, suivi fournisseurs).
4. **Optimiser la maintenance préventive du dimanche.** Elle mobilise {fr(part_plan, 0)} % du temps d'arrêt, avec
   toutes les lignes et les deux sites à l'arrêt en même temps. Hypothèse à tester : échelonner les interventions
   par ligne ou par site, et vérifier que les pannes diminuent bien après chaque intervention.
5. **Réduire la consommation à vide et sur la Peinture.** La Peinture consomme {fr(k_pe, 1)} kWh par véhicule
   contre {fr(k_mo, 1)} kWh au montage final ; une heure sans production consomme encore {fr(vide, 0)} kWh en moyenne
   par ligne. Mise en veille pendant les arrêts planifiés et optimisation des cycles de séchage.
6. **Capitaliser sur la baisse des défauts.** Identifier ce qui a changé depuis mars (formation, réglages,
   fournisseurs) pour le généraliser, notamment au poste de nuit.
""")
    st.info("Limites : données observationnelles sur 5 mois ; les liens relevés sont des associations. "
            "Le TRS est celui fourni dans le fichier. Les tests z supposent des véhicules indépendants, "
            "ce qui est une approximation.")


# ====================================================================
# 7. CONNEXION ET NAVIGATION
# ====================================================================
def identifiants_valides(utilisateur: str, mot_de_passe: str) -> bool:
    ok_u = hmac.compare_digest(utilisateur.strip().lower().encode(), IDENTIFIANT.encode())
    ok_m = hmac.compare_digest(mot_de_passe.encode(), MOT_DE_PASSE.encode())
    return ok_u and ok_m


def page_connexion() -> None:
    _, centre, _ = st.columns([1, 1.1, 1])
    with centre:
        st.write("")
        st.write("")
        st.markdown('<div class="connexion-titre">Production automobile</div>'
                    '<div class="connexion-sous">Suivi des sites de Tanger et de Kénitra. '
                    'Connectez-vous pour accéder au tableau de bord.</div>', unsafe_allow_html=True)
        with st.form("connexion"):
            utilisateur = st.text_input("Identifiant")
            mot_de_passe = st.text_input("Mot de passe", type="password")
            envoye = st.form_submit_button("Se connecter")
        if envoye:
            if identifiants_valides(utilisateur, mot_de_passe):
                st.session_state["auth"] = True
                st.session_state["utilisateur"] = utilisateur.strip().capitalize()
                st.rerun()
            else:
                st.error("Identifiant ou mot de passe incorrect.")


PAGES = {
    "Vue d'ensemble": page_vue_ensemble,
    "Performance": page_performance,
    "Qualité et défauts": page_qualite,
    "Arrêts et Pareto": page_arrets,
    "Énergie": page_energie,
}


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    style_graphiques()

    if not st.session_state.get("auth"):
        page_connexion()
        return

    if not CHEMIN_DONNEES.exists():
        st.error(f"Fichier introuvable : {CHEMIN_DONNEES.name}. Placez-le à côté de app.py.")
        st.stop()
    df, rapport, manquantes = charger_donnees(str(CHEMIN_DONNEES))

    # ---- barre latérale : navigation et filtres
    st.sidebar.markdown(f"**Connecté : {st.session_state.get('utilisateur', '')}**")
    if st.sidebar.button("Se déconnecter"):
        st.session_state.clear()
        st.rerun()
    st.sidebar.divider()
    page = st.sidebar.radio("Navigation", [*PAGES, "Données et nettoyage", "Conclusion"])

    if page == "Conclusion":
        bandeau(df.date.min(), df.date.max(), page)
        page_conclusion(df)
        return

    st.sidebar.divider()
    st.sidebar.subheader("Filtres")
    dmin, dmax = df.date.min().date(), df.date.max().date()
    plage = st.sidebar.date_input("Période", (dmin, dmax), min_value=dmin, max_value=dmax,
                                  format="DD/MM/YYYY")
    sites = st.sidebar.multiselect("Sites", sorted(df.site.unique()), default=sorted(df.site.unique()))
    lignes = st.sidebar.multiselect("Lignes", ORDRE_LIGNES, default=ORDRE_LIGNES)
    equipes = st.sidebar.multiselect("Équipes", ORDRE_EQUIPES, default=ORDRE_EQUIPES)
    modeles = st.sidebar.multiselect("Modèles", sorted(df.modele.unique()), default=sorted(df.modele.unique()))
    excl = st.sidebar.checkbox("Exclure la maintenance planifiée du TRS", value=True,
                               help="La maintenance préventive du dimanche matin a un TRS de 0 : "
                                    "l'inclure fausse la comparaison entre équipes.")

    if not (isinstance(plage, (tuple, list)) and len(plage) == 2):
        st.info("Sélectionnez une date de début et une date de fin.")
        st.stop()
    masque = (df.date >= pd.Timestamp(plage[0])) & (df.date <= pd.Timestamp(plage[1])) \
        & df.site.isin(sites) & df.ligne.isin(lignes) & df.equipe.isin(equipes) & df.modele.isin(modeles)
    d = df[masque]
    if d.empty:
        bandeau(dmin, dmax, page)
        st.warning("Aucune donnée ne correspond aux filtres. Élargissez la sélection dans la barre latérale.")
        st.stop()
    st.sidebar.caption(f"{fr(len(d))} lignes sélectionnées sur {fr(len(df))}")

    bandeau(plage[0], plage[1], page)
    if page == "Données et nettoyage":
        page_donnees(d, df, rapport, manquantes)
    else:
        PAGES[page](d, excl)


def bandeau(debut, fin, page: str) -> None:
    st.markdown(f'<div class="bandeau"><div class="b-titre">{page}</div>'
                f'<div class="b-sous">Production automobile à Tanger et à Kénitra, '
                f'du {pd.Timestamp(debut):%d/%m/%Y} au {pd.Timestamp(fin):%d/%m/%Y}</div></div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
