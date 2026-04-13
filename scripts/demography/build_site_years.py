"""Map each site in the study data to (ISO3, WPP reference year).

Study periods come from Battle et al. 2016 (Dryad doi:10.5061/dryad.j0kv0), the
GRIFFIN=="Yes" subset -- i.e. exactly the records Griffin 2014 fitted. The WPP
reference year is the study-period midpoint rounded to the nearest year (WPP
estimates are as of 1 July).
"""
import collections
import csv
import math
import os
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STUDY = os.path.join(ROOT, "artifacts/numpyro_jamie_100x4/study_data_2014_age_le20.csv")
BATTLE = os.environ.get("BATTLE_CSV", os.path.join(ROOT, "battle.csv"))

# site_name -> Battle (INC_AUTHOR, ACD_LOCATION). Site names in the study data are
# slugged variants of Battle's ACD_LOCATION.
SITE_TO_BATTLE = {
    "Burkina_faso_Ye_Nouna":          ("Ye", "Nouna"),
    "Cameroon_Bonnet_Ebolakonou":     ("Bonnet", "Ebolakounou"),
    "Cameroon_Bonnet_Koundou":        ("Bonnet", "Koundou"),
    "dielmo":                         ("Trape", "Dielmo"),
    "ndiop":                          ("Trape", "Ndiop"),
    "Ghana_Chandramohan_KND":         ("Chandramohan", "Kassena-Nankana District"),
    "Ivory_coast_Henry_R0_all_ages":  ("Henry", "Katiola - R0"),
    "Ivory_coast_Henry_R1_all_ages":  ("Henry", "Korhogo - R1"),
    "Ivory_coast_Henry_R2_all_ages":  ("Henry", "Korhogo - R2"),
    "Kenya_Mwangi_Chonyi":            ("Mwangi", "Chonyi"),
    "Kenya_Mwangi_Ngerebya":          ("Mwangi", "Ngerenya"),
    "Mali_Dicko_Doneguebougou":       ("Dicko", "Doneguebougou"),
    "Mali_Dicko_Sotuba":              ("Dicko", "Sotuba"),
    "Mozambique_Guinovart_Manhica":   ("Guinovart", "Manhica"),
    "Mozambique_Saute_Manhica":       ("Saute", "Manhica"),
    "Mozambique_Thompson_Matola":     ("Thompson", "Matola"),
    "Senegal_Diallo_Dakar_south":     ("Diallo", "South Dakar"),   # Diallo 1998, 6/1994-5/1995
    "Senegal_Diallo_Dakar_central":   ("Diallo", "Dakar"),         # Diallo 2000, 3/1996-2/1997
    "Tanzania_Henning_Ifakara":       ("Henning", "Ifakara"),
    "Tanzania_Lusingu_Magamba":       ("Lusingu", "Magamba"),
    "Tanzania_Lusingu_Mgome":         ("Lusingu", "Mgome"),
    "Tanzania_Lusingu_Ubiri":         ("Lusingu", "Ubiri"),
    "Tanzania_Schellenberg_Ifakara":  ("Schellenberg", "Ifakara"),
}

ISO3 = {
    "Burkina Faso": "BFA", "Cameroon": "CMR", "Ghana": "GHA", "Ivory Coast": "CIV",
    "Kenya": "KEN", "Mali": "MLI", "Mozambique": "MOZ", "Senegal": "SEN",
    "Tanzania": "TZA",
}

# Dicko has two distinct survey years (1999, 2000) for the same two villages; the
# study data carries them as duplicate site_index pairs. Lower site_index -> 1999.
DICKO_YEARS = {False: 1999, True: 2000}


def norm(s):
    """Battle's CSV is Mac-Roman; fold accents and whitespace so keys match."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).strip()


def decimal_year(month, year):
    if month in ("NA", ""):
        return float(year) + 0.5
    return float(year) + (float(month) - 0.5) / 12.0


def main():
    with open(BATTLE, encoding="mac_roman") as fh:
        battle = [r for r in csv.DictReader(fh) if r["GRIFFIN"].strip() == "Yes"]

    # (author, location) -> set of (start, end) decimal-year spans
    spans = collections.defaultdict(set)
    for r in battle:
        key = (norm(r["INC_AUTHOR"]), norm(r["ACD_LOCATION"]))
        spans[key].add((decimal_year(r["START_MONTH"], r["START_YEAR"]),
                        decimal_year(r["END_MONTH"], r["END_YEAR"])))

    with open(STUDY) as fh:
        study = list(csv.DictReader(fh))
    sites = {}
    for r in study:
        sites.setdefault(int(r["site_index"]), (r["country_name"], r["site_name"]))

    # For Dicko, the two site_index blocks are the two survey years.
    dicko_idx = sorted(i for i, (_, s) in sites.items() if "Dicko" in s)
    dicko_second = set(dicko_idx[len(dicko_idx) // 2:])

    out = []
    for idx in sorted(sites):
        country, site = sites[idx]
        key = SITE_TO_BATTLE[site]
        if key not in spans:
            raise SystemExit(f"no Battle match for {site} -> {key}")
        sp = sorted(spans[key])
        if key[0] == "Dicko":
            year = DICKO_YEARS[idx in dicko_second]
            lo, hi = year, year + 1.0
        else:
            lo = min(s for s, _ in sp)
            hi = max(e for _, e in sp)
            # WPP year Y is referenced to 1 July (decimal year Y.5), so the Y
            # whose reference date is nearest the study midpoint is just the
            # calendar year the midpoint falls in.
            year = math.floor((lo + hi) / 2.0)
        out.append({
            "site_index": idx,
            "site_name": site,
            "country_name": country,
            "iso3": ISO3[country],
            "battle_author": key[0],
            "battle_location": key[1],
            "study_start": round(lo, 2),
            "study_end": round(hi, 2),
            "wpp_year": year,
        })

    with open("site_years.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['site_index']:>3} {r['site_name']:<32} {r['iso3']} "
              f"{r['study_start']}-{r['study_end']} -> {r['wpp_year']}")


if __name__ == "__main__":
    main()
