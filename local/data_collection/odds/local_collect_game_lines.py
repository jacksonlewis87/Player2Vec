import pyperclip


def parse_roto_wire_content(raw_text: str):
    raw_text = raw_text.split("Add FilterLegend\r\n")[1].split("\r\nCaesars Sports")[0]
    game_count = raw_text.count("@")
    raw_text = raw_text.split("\r\n")
    teams = raw_text[:game_count]
    dates = raw_text[game_count : 2 * game_count]
    seasons = raw_text[2 * game_count : 3 * game_count]
    spreads = raw_text[5 * game_count : 6 * game_count]
    games = {}
    for i in range(game_count):
        games[
            f"{teams[i]}_{' '.join(dates[i].split(' ')[:2])}_{seasons[i]}"
        ] = f"{teams[i]}\t{' '.join(dates[i].split(' ')[:2])}\t{seasons[i]}\t{spreads[i]}"

    return games


def do_work():
    existing_games = []
    while True:
        raw_text = pyperclip.paste()
        new_games = parse_roto_wire_content(raw_text=raw_text)
        for game_id, game in new_games.items():
            if game_id not in existing_games:
                print(game)
                existing_games += [game_id]


if __name__ == "__main__":
    # hacky way to quickly scrap odds off of https://www.rotowire.com/betting/nba/archive.php
    # use in tandem with local_automate_scroll_and_copy.py
    do_work()
