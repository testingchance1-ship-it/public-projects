
class ConnectFour:
    def __init__(self):
        self.board = [[' ' for _ in range(7)] for _ in range(6)]
        self.current_player = 'X'

    def print_board(self):
        print('\n'.join([' '.join(row) for row in self.board[::-1]]))
        print(' '.join(str(i) for i in range(1, 8)))

    def drop_piece(self, column):
        column -= 1
        for row in range(len(self.board)):
            if self.board[row][column] == ' ':
                self.board[row][column] = self.current_player
                return

    def is_win(self):
        # Check horizontal locations for win
        for c in range(7-3):
            for r in range(6):
                if self.board[r][c] == self.current_player and self.board[r][c+1] == self.current_player and self.board[r][c+2] == self.current_player and self.board[r][c+3] == self.current_player:
                    return True

        # Check vertical locations for win
        for c in range(7):
            for r in range(6-3):
                if self.board[r][c] == self.current_player and self.board[r+1][c] == self.current_player and self.board[r+2][c] == self.current_player and self.board[r+3][c] == self.current_player:
                    return True

        # Check positively sloped diagonals
        for c in range(7-3):
            for r in range(6-3):
                if self.board[r][c] == self.current_player and self.board[r+1][c+1] == self.current_player and self.board[r+2][c+2] == self.current_player and self.board[r+3][c+3] == self.current_player:
                    return True

        # Check negatively sloped diagonals
        for c in range(7-3):
            for r in range(3, 6):
                if self.board[r][c] == self.current_player and self.board[r-1][c+1] == self.current_player and self.board[r-2][c+2] == self.current_player and self.board[r-3][c+3] == self.current_player:
                    return True

    def is_board_full(self):
        for row in self.board:
            if ' ' in row:
                return False
        return True


def main():
    game = ConnectFour()
    while True:
        game.print_board()
        column = int(input(f"Player {game.current_player}, choose a column: ")) - 1
        if column < 0 or column > 6:
            print("Invalid column choice. Try again.")
            continue

        game.drop_piece(column)
        if game.is_win():
            game.print_board()
            print(f"Player {game.current_player} wins!")
            break
        elif game.is_board_full():
            game.print_board()
            print("It's a tie!")
            break
        game.current_player = 'O' if game.current_player == 'X' else 'X'


if __name__ == "__main__":
    main()
