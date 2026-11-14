import numpy as np
from collections import defaultdict
from elo import elos, expected_score

# ── R32 BRACKET ───────────────────────────────────────────────────────────────
R32_BRACKET = [
    # LEFT SIDE OF DRAW
    ['Germany',      'Paraguay'],
    ['France',       'Sweden'],
    ['South Africa', 'Canada'],
    ['Netherlands',  'Morocco'],
    ['Portugal',     'Croatia'],
    ['Spain',        'Austria'],
    ['USA',          'Bosnia and Herzegovina'],
    ['Belgium',      'Senegal'],

    # RIGHT SIDE OF DRAW
    ['Brazil',       'Japan'],
    ['Ivory Coast',  'Norway'],
    ['Mexico',       'Ecuador'],
    ['England',      'DR Congo'],
    ['Argentina',    'Cape Verde'],
    ['Australia',    'Egypt'],
    ['Switzerland',  'Algeria'],
    ['Colombia',     'Ghana'],
]


# ── HELPERS ───────────────────────────────────────────────────────────────────
def simulate_ko_match(team_a, team_b):
    """Knockout match — no draws, returns winner."""
    e = expected_score(elos[team_a], elos[team_b])
    return team_a if np.random.random() < e else team_b

def pair_up(teams):
    """Converts flat list of winners into next round pairs."""
    return [[teams[i], teams[i + 1]] for i in range(0, len(teams), 2)]

def simulate_knockout_round(pairs, round_name):
    """Simulates one knockout round, prints results, returns winners."""
    print(f"\n{'═'*55}")
    print(f"  {round_name}")
    print(f"{'═'*55}")
    winners = []
    for a, b in pairs:
        winner = simulate_ko_match(a, b)
        e      = expected_score(elos[a], elos[b])
        prob   = e if winner == a else 1 - e
        print(f"  {a:<25} vs {b:<25}")
        print(f"  → {winner} advances  (win prob: {prob*100:.0f}%)")
        print()
        winners.append(winner)
    return winners

def print_team_ratings(bracket=None):
    """Prints all teams in the bracket ranked by ELO."""
    if bracket is None:
        bracket = R32_BRACKET

    all_teams = [team for pair in bracket for team in pair]
    ranked    = sorted(all_teams, key=lambda t: -elos[t])

    print(f"\n{'═'*50}")
    print(f"  32 TEAMS RANKED BY ELO")
    print(f"{'═'*50}")
    print(f"  {'Rank':<5} {'Team':<25} {'ELO':>7}")
    print(f"  {'─'*40}")
    for i, team in enumerate(ranked, 1):
        bar = '█' * int((elos[team] - 1500) / 30)
        print(f"  {i:<5} {team:<25} {elos[team]:>7.1f}  {bar}")


# ── FULL TOURNAMENT (SINGLE RUN) ──────────────────────────────────────────────
def run_full_tournament(bracket=None, verbose=True):
    """
    Runs the tournament from R32 through to the Final.
    bracket = list of 16 pairs (32 teams).
    Falls back to R32_BRACKET if none provided.
    """
    if bracket is None:
        bracket = R32_BRACKET

    # Validate all teams exist in ELO system
    missing = []
    for pair in bracket:
        for team in pair:
            if team not in elos:
                missing.append(team)
    if missing:
        print(f"⚠ These teams were not found in ELO ratings: {missing}")
        print("  Check spelling and fix before running.")
        return

    # Validate bracket size
    if len(bracket) != 16:
        print(f"⚠ Bracket needs exactly 16 pairs, got {len(bracket)}.")
        return

    print(f"\n{'═'*55}")
    print(f"  FIFA WORLD CUP 2026 — KNOCKOUT STAGE")
    print(f"{'═'*55}")

    # --- Round of 32 ---
    r32_winners = simulate_knockout_round(bracket, "ROUND OF 32")

    # --- Round of 16 ---
    r16_pairs   = pair_up(r32_winners)
    r16_winners = simulate_knockout_round(r16_pairs, "ROUND OF 16")

    # --- Quarter-finals ---
    qf_pairs    = pair_up(r16_winners)
    qf_winners  = simulate_knockout_round(qf_pairs, "QUARTER-FINALS")

    # --- Semi-finals ---
    sf_pairs    = pair_up(qf_winners)
    sf_winners  = simulate_knockout_round(sf_pairs, "SEMI-FINALS")
    sf_losers   = [
        sf_pairs[0][0] if sf_winners[0] == sf_pairs[0][1] else sf_pairs[0][1],
        sf_pairs[1][0] if sf_winners[1] == sf_pairs[1][1] else sf_pairs[1][1],
    ]

    # --- Third-place playoff ---
    print(f"\n{'═'*55}")
    print(f"  THIRD-PLACE PLAYOFF")
    print(f"{'═'*55}")
    third  = simulate_ko_match(*sf_losers)
    fourth = sf_losers[1] if third == sf_losers[0] else sf_losers[0]
    print(f"  {sf_losers[0]} vs {sf_losers[1]}")
    print(f"  → {third} finishes 3rd")

    # --- Final ---
    print(f"\n{'═'*55}")
    print(f"  ★  THE FINAL  ★")
    print(f"{'═'*55}")
    finalist_a, finalist_b = sf_winners
    champion  = simulate_ko_match(finalist_a, finalist_b)
    runner_up = finalist_b if champion == finalist_a else finalist_a
    e    = expected_score(elos[finalist_a], elos[finalist_b])
    prob = e if champion == finalist_a else 1 - e
    print(f"  {finalist_a} vs {finalist_b}")
    print(f"\n  🏆 CHAMPION:  {champion}  (win prob: {prob*100:.0f}%)")
    print(f"  Runner-up:   {runner_up}")
    print(f"  3rd place:   {third}")
    print(f"  4th place:   {fourth}")

    return champion


# ── MONTE CARLO ───────────────────────────────────────────────────────────────
def simulate_many(bracket=None, n=10_000):
    """
    Runs n Monte Carlo simulations and shows the probability
    of each team reaching every stage of the tournament.
    Also prints the single most likely bracket path.
    """
    if bracket is None:
        bracket = R32_BRACKET

    all_teams = [team for pair in bracket for team in pair]

    # Track how many times each team reaches each stage
    stage_counts = {
        team: {
            'R32':   0,
            'R16':   0,
            'QF':    0,
            'SF':    0,
            'Final': 0,
            'Winner':0,
        }
        for team in all_teams
    }

    print(f"Running {n:,} simulations...")

    for _ in range(n):

        # Every team starts in R32
        for team in all_teams:
            stage_counts[team]['R32'] += 1

        # --- Round of 32 ---
        r32_w = [simulate_ko_match(*p) for p in bracket]
        for team in r32_w:
            stage_counts[team]['R16'] += 1

        # --- Round of 16 ---
        r16_w = [simulate_ko_match(*p) for p in pair_up(r32_w)]
        for team in r16_w:
            stage_counts[team]['QF'] += 1

        # --- Quarter-finals ---
        qf_w = [simulate_ko_match(*p) for p in pair_up(r16_w)]
        for team in qf_w:
            stage_counts[team]['SF'] += 1

        # --- Semi-finals ---
        sf_pairs_i = pair_up(qf_w)
        sf_w       = [simulate_ko_match(*p) for p in sf_pairs_i]
        for team in sf_w:
            stage_counts[team]['Final'] += 1

        # --- Final ---
        champion = simulate_ko_match(*sf_w)
        stage_counts[champion]['Winner'] += 1

    # ── PRINT PROBABILITY TABLE ───────────────────────────────────────────────
    ranked = sorted(all_teams, key=lambda t: -stage_counts[t]['Winner'])

    print(f"\n{'═'*90}")
    print(f"  MONTE CARLO RESULTS — {n:,} SIMULATIONS")
    print(f"{'═'*90}")
    print(f"  {'Team':<25} {'R16':>6}  {'QF':>6}  {'SF':>6}  {'Final':>6}  {'Winner':>6}  {'Odds':>6}")
    print(f"  {'─'*80}")

    for team in ranked:
        c     = stage_counts[team]
        r16   = c['R16']    / n * 100
        qf    = c['QF']     / n * 100
        sf    = c['SF']     / n * 100
        final = c['Final']  / n * 100
        win   = c['Winner'] / n * 100
        odds  = f"{round(n / c['Winner'])}:1" if c['Winner'] > 0 else "---"
        bar   = '█' * int(win / 2)
        print(f"  {team:<25} {r16:>5.1f}%  {qf:>5.1f}%  {sf:>5.1f}%  {final:>5.1f}%  {win:>5.1f}%  {odds:>6}  {bar}")

    # ── MOST LIKELY BRACKET ───────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  MOST LIKELY BRACKET (favourite wins every game)")
    print(f"{'═'*55}")

    # R32 most likely winners
    print(f"\n  ROUND OF 32")
    print(f"  {'─'*52}")
    r32_likely = []
    for a, b in bracket:
        e      = expected_score(elos[a], elos[b])
        winner = a if e >= 0.5 else b
        prob   = e if e >= 0.5 else 1 - e
        r32_likely.append(winner)
        print(f"  {a:<22} vs {b:<22} → {winner} ({prob*100:.0f}%)")

    # R16 most likely winners
    print(f"\n  ROUND OF 16")
    print(f"  {'─'*52}")
    r16_likely = []
    for a, b in pair_up(r32_likely):
        e      = expected_score(elos[a], elos[b])
        winner = a if e >= 0.5 else b
        prob   = e if e >= 0.5 else 1 - e
        r16_likely.append(winner)
        print(f"  {a:<22} vs {b:<22} → {winner} ({prob*100:.0f}%)")

    # QF most likely winners
    print(f"\n  QUARTER-FINALS")
    print(f"  {'─'*52}")
    qf_likely = []
    for a, b in pair_up(r16_likely):
        e      = expected_score(elos[a], elos[b])
        winner = a if e >= 0.5 else b
        prob   = e if e >= 0.5 else 1 - e
        qf_likely.append(winner)
        print(f"  {a:<22} vs {b:<22} → {winner} ({prob*100:.0f}%)")

    # SF most likely winners
    print(f"\n  SEMI-FINALS")
    print(f"  {'─'*52}")
    sf_likely = []
    for a, b in pair_up(qf_likely):
        e      = expected_score(elos[a], elos[b])
        winner = a if e >= 0.5 else b
        prob   = e if e >= 0.5 else 1 - e
        sf_likely.append(winner)
        print(f"  {a:<22} vs {b:<22} → {winner} ({prob*100:.0f}%)")

    # Third place
    print(f"\n  THIRD-PLACE PLAYOFF")
    print(f"  {'─'*52}")
    third_teams = [
        sf_likely[0] if sf_likely[0] != sf_likely[0] else
        [t for t in pair_up(qf_likely)[0] if t != sf_likely[0]][0],
        [t for t in pair_up(qf_likely)[1] if t != sf_likely[1]][0],
    ]
    a, b   = third_teams
    e      = expected_score(elos[a], elos[b])
    third  = a if e >= 0.5 else b
    prob   = e if e >= 0.5 else 1 - e
    print(f"  {a:<22} vs {b:<22} → {third} finishes 3rd ({prob*100:.0f}%)")

    # Final
    print(f"\n  ★  THE FINAL")
    print(f"  {'─'*52}")
    a, b   = sf_likely[0], sf_likely[1]
    e      = expected_score(elos[a], elos[b])
    winner = a if e >= 0.5 else b
    prob   = e if e >= 0.5 else 1 - e
    print(f"  {a:<22} vs {b:<22} → 🏆 {winner} ({prob*100:.0f}%)")

    return stage_counts
