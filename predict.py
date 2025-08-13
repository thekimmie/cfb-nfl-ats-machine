# ----- Script-model debug (raw output from predict.py) -----
@app.get("/debug/model-script")
async def debug_model_script(limit: int = 10, reload: bool = False):
    """
    Build a minimal games[] payload from current odds and call predict.predict_games(games).
    Use ?reload=1 to hot-reload predict.py after edits.
    """
    pm = load_predict_module(reload=bool(reload))
    if not (pm and hasattr(pm, "predict_games")):
        return {
            "enabled": MODEL_SOURCE == "script",
            "path": MODEL_PATH,
            "has_predict_games": False
        }

    # Build the same minimal inputs we pass in build_rows()
    odds = await fetch_all_odds()
    games: List[Dict[str, Any]] = []
    for ev in odds:
        lg = league_label_from_odds_key(ev.get("sport_key") or ev.get("sport_title", ""))
        if not lg:
            continue
        commence_time = ev.get("commence_time")
        home = ev.get("home_team")
        away = ev.get("away_team")

        # Pull current best available lines for spread/total/ML (same helpers as elsewhere)
        m_spreads, _ = pick_market(ev.get("bookmakers") or [], "spreads")
        m_totals,  _ = pick_market(ev.get("bookmakers") or [], "totals")
        m_h2h,     _ = pick_market(ev.get("bookmakers") or [], "h2h")

        ml_home = next((o for o in (m_h2h or {}).get("outcomes", []) if o.get("name")==home), {})
        ml_away = next((o for o in (m_h2h or {}).get("outcomes", []) if o.get("name")==away), {})
        over    = next((o for o in (m_totals or {}).get("outcomes", []) if (o.get("name") or "").lower()=="over"), {})
        under   = next((o for o in (m_totals or {}).get("outcomes", []) if (o.get("name") or "").lower()=="under"), {})
        sp_home = next((o for o in (m_spreads or {}).get("outcomes", []) if o.get("name")==home), {})

        games.append({
            "league": lg,
            "game_time": commence_time,
            "home_team": home,
            "away_team": away,
            "sportsbook_spread": sp_home.get("point"),
            "sportsbook_total": over.get("point") or under.get("point"),
            "sportsbook_ml_odds_home": ml_home.get("price"),
            "sportsbook_ml_odds_away": ml_away.get("price"),
        })

    # Call the script model with inputs; fall back to no-arg if needed
    try:
        try:
            out = pm.predict_games(games[:max(0, limit)])  # primary: requires games list
        except TypeError:
            out = pm.predict_games()                       # fallback: no-arg signature
    except Exception as e:
        return {"enabled": True, "call_error": str(e)}

    if isinstance(out, list):
        keys = sorted({k for r in out if isinstance(r, dict) for k in r.keys()})
        return {"enabled": True, "count": len(out), "keys": keys, "sample": out[:max(0, limit)]}
    else:
        return {"enabled": True, "note": "predict_games returned non-list", "type": str(type(out))}
