"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 23/04-2021
"""
import chess.engine
import chess.pgn
import random
from tqdm import tqdm

DATA_FILEPATH = '../data/'
STOCKFISH_FILEPATH = '../stockfish/stockfish_13_win_x64_avx2.exe'
CSV_FILEPATH = '../parsed_data/parsed_games_test.csv'
ERROR_FEN = '<| ERROR: incorrect FEN string format, '
ERROR_FEATURE = '<| ERROR: incorrect feature length, '
ERROR_CSV = '<| ERROR: incorrect length of csv row, '
ERROR_EVALUATION = '<| ERROR: could not evaluate position, '
NUM_GAMES = 10
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

    # Ok now we have the one hot encoding
    # Append the player which turn it is to move. This might act as some type of bias?
    # So maybe we don't have to add bias later... Food for thought.
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


def write_data_to_csv(data_list):
    with open(CSV_FILEPATH, 'a') as fd:
        for position in data_list:
            fd.write(position + '\n')
        fd.close()


def parse_data():
    with open(CSV_FILEPATH, 'a') as fd:
        fd.write('y')
        for x in range(8*8*7):
            fd.write(f',x{x}')
        fd.write('\n')
    fd.close()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_FILEPATH)
    with open(DATA_FILEPATH + pgn_list[0]) as pgn:
        csv_list = []
        for _ in tqdm(range(NUM_GAMES)):
            game = chess.pgn.read_game(pgn)
            board = game.board()
            for idx, move in enumerate(game.mainline_moves()):
                # print(f'\t\tNumber of moves processed [{idx + 1}]')
                board.push(move)
                fen = board.fen()
                # If we randomly sample smaller than 0.5 we will skipp this position. Should speed up the parsing
                # time by 50% on average. Lets hope this works.
                if random.random() < 0.5:
                    continue
                # Ok we got the goods, but now how to take the FEN string and model it according to our data specification?
                # We want to structure the data such that it has 8*8*7 features, i.e. dimensionality of the data is 448. The
                # first 8x8 are to represent the board. And at each index in this, what we pretend to be a 2D matrix,
                # we will have 7 values. The first six of these are one-hot coded representations both the piece type and the
                # color of the piece. The final value dictates what colors turn it is to move. -1 meaning BLACK to move, and 1
                # means that it is WHITE to move.
                try:
                    eval = str((engine.analyse(board, chess.engine.Limit(depth=20))['score'].white().score() / 100))
                    csv_list.append(eval + parse_FEN(fen))
                except:
                    print(ERROR_EVALUATION + 'moving on to next game ...')
                    break
                # According to Forsyth-Edwards Notation (FEN) on Wikipedia
                # (https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)
                # everything coming after either 'b' or 'w' is trivial to us. We only care about the current board state and
                # what player it is next to move. Castling rights, en-passant squares and 50-move rule are not interesting.
            write_data_to_csv(csv_list)

    ###
    # info = engine.analyse(board, chess.engine.Limit(depth=20))
    # print('score:', float((info['score'].white().score()/100)))
    engine.quit()
    ###


def main():
    parse_data()


if __name__ == '__main__':
    main()
