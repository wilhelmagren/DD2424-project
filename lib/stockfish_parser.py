"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 23/04-2021
"""
import chess.engine
import chess.pgn
from tqdm import tqdm
import pandas as pd

DATA_FILEPATH = '../data/'
STOCKFISH_FILEPATH = '../stockfish/stockfish_13_win_x64_avx2.exe'
CSV_FILEPATH = '../parsed_data/1000games_TRIMMED2019_batch8.csv.gz'
ERROR_FEN = '<| ERROR: incorrect FEN string format, '
ERROR_FEATURE = '<| ERROR: incorrect feature length, '
ERROR_CSV = '<| ERROR: incorrect length of csv row, '
ERROR_EVALUATION = '<| ERROR: could not evaluate position, '
NUM_GAMES = 1000
SKIP_GAMES = 7000
SKIP_FIRST_POSITIONS = 10
NUM_FEATURES = 7

pgn_list = ['ficsgamesdb_2020_chess_nomovetimes_201347.pgn',
            'ficsgamesdb_2019_chess_nomovetimes_201348.pgn',
            'ficsgamesdb_2018_chess_nomovetimes_201349.pgn',
            'ficsgamesdb_2017_chess_nomovetimes_201350.pgn',
            'ficsgamesdb_2016_chess_nomovetimes_201351.pgn']
PIECE_IDX = {
    'p': 0,
    'P': 0,
    'b': 1,
    'B': 1,
    'n': 2,
    'N': 2,
    'r': 3,
    'R': 3,
    'q': 4,
    'Q': 4,
    'k': 5,
    'K': 5
}

CLR_MOVE = {
    'b': -1,
    'w': 1
}

ZERO_FEATURE = [0 for _ in range(6)]

MATED_VALUES = [-255.0, 255.0]


def is_upper_case(s):
    return 1 if s.isupper() else -1


def parse_FEN(s):
    """
    func parse_FEN/1
    @spec parse_FEN(string()) :: string()
        Parse the FEN-string 's' and returns the 8x8x7 representation as a string.
        This will later be written to .csv file? Or maybe here already. We find out later.
        TODO: decide on when to write to .csv
    """
    s_list = s.split(' ')

    assert len(s_list) >= 2, print(ERROR_FEN)

    board_rep_list, clr_move = s_list[0].split('/'), s_list[1]

    assert len(board_rep_list) == 8, print(ERROR_FEN + f'num rows on board={len(board_rep_list)}')

    fen_board = [[[0, 0, 0, 0, 0, 0, CLR_MOVE[clr_move]] for _ in range(8)] for _ in range(8)]

    for idx, row in enumerate(board_rep_list):
        cnt = 0
        for pce in row:
            if pce.isdigit():
                cnt += int(pce)
                continue
            else:
                fen_board[idx][cnt][PIECE_IDX[pce]] = is_upper_case(pce)
            cnt += 1

    csv_format_s = ''
    for idx in range(8):
        for jdx in range(8):
            feature_row = fen_board[idx][jdx]

            assert len(feature_row) == NUM_FEATURES, print(ERROR_FEATURE)

            for kdx, feature in enumerate(feature_row):
                csv_format_s += ',' + str(feature)

    assert len(csv_format_s.split(',')) - 1 == 8 * 8 * 7, \
        print(ERROR_CSV + f" the row has length {len(csv_format_s.split(',')) - 1}")

    return csv_format_s


def write_data_to_csv(dataframe):
    print(dataframe)
    dataframe.to_csv(CSV_FILEPATH, compression='gzip')


def parse_data():
    column_list = ['y']
    for x in range(NUM_FEATURES*8*8):
        column_list.append(f'x{x}')
    main_df = pd.DataFrame(columns=column_list)
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_FILEPATH)

    with open(DATA_FILEPATH + pgn_list[1]) as pgn:
        print('<| Skipping all pre-parsed games ...')
        for _ in range(SKIP_GAMES):
            game = chess.pgn.read_game(pgn)
        print('<| Done! Parsing all games now ... \n')
        for _ in tqdm(range(NUM_GAMES)):
            game = chess.pgn.read_game(pgn)
            board = game.board()
            dic_list = []
            for idx, move in enumerate(game.mainline_moves()):
                board.push(move)
                fen = board.fen()
                dic = {}
                to_move = CLR_MOVE[fen.split(' ')[1]]
                eval = ''
                score = engine.analyse(board, chess.engine.Limit(depth=15))['score'].white()
                try:
                    eval = str(str(score.score() / 100))
                except:
                    if score.mate() > 0:
                        eval = MATED_VALUES[1]
                    elif score.mate() < 0:
                        eval = MATED_VALUES[0]
                    else:
                        if to_move > 0:
                            eval = MATED_VALUES[0]
                        else:
                            eval = MATED_VALUES[1]
                if -10 < float(eval) < 10:
                    continue
                df_l = (str(eval) + parse_FEN(fen)).split(',')
                for (col, val) in zip(column_list, df_l):
                    dic[col] = val
                dic_list.append(dic)
            for dicc in dic_list:
                main_df = main_df.append(dicc, ignore_index=True)
        write_data_to_csv(main_df)
    engine.quit()
    ###


def main():
    parse_data()


if __name__ == '__main__':
    main()
