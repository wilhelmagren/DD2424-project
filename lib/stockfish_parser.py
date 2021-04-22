import chess
import chess.engine

#pgn = open('../data/pgn_data.pgn')

fen = "N1b2bnr/pp1pk1pp/4pp2/1B2Nn2/4P3/8/PPP2PPP/R1B1K2R b KQ - 0 10"

engine = chess.engine.SimpleEngine.popen_uci('../stockfish/stockfish_13_win_x64_avx2.exe')

board = chess.Board(fen)

info = engine.analyse(board, chess.engine.Limit(depth=20))
print('score:', float((info['score'].white().score()/100)))
engine.quit()
