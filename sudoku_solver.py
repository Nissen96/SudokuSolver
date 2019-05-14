import argparse

from itertools import product


BOARDS = ["easy", "medium", "hard"]  # Available Sudoku boards


class SudokuError(Exception):
    """
    An application specific error.
    """
    pass


class SudokuBase:
    """
    Base Sudoku methods used by multiple subclasses
    """
    @staticmethod
    def get_row(board, row):
        return board[row]

    @staticmethod
    def get_column(board, column):
        return (board[row][column] for row in range(9))

    @staticmethod
    def get_square(board, row, column):
        return (
            board[r][c]
            for r in range(row * 3, (row + 1) * 3)
            for c in range(column * 3, (column + 1) * 3)
        )


class SudokuBoard(SudokuBase):
    """
    Sudoku Board representation
    """
    def __init__(self, board_file):
        self.board = self.__create_board(board_file)
        self.__validate()

    @staticmethod
    def __create_board(board_file):
        # Create an initial matrix
        board = []

        # Iterate over each line
        for line in board_file:
            line = line.strip()

            # Line must be 9 characters
            if len(line) != 9:
                raise SudokuError("Each line in the puzzle must be 9 characters long")

            board.append([])

            # Iterate over each character
            for c in line:
                # Character must be an int
                try:
                    digit = int(c)
                    if not 0 <= digit <= 9:
                        raise ValueError
                except ValueError:
                    raise SudokuError("Valid characters for a sudoku puzzle are 0-9")
                board[-1].append(digit)

        # Puzzle must have 9 lines
        if len(board) != 9:
            raise SudokuError("Each sudoku puzzle must be 9 lines long")

        return board

    def __validate(self):
        for row in range(9):
            if not self.__validate_row(row):
                raise SudokuError(f"Row ({row + 1}) is invalid")
        for column in range(9):
            if not self.__validate_column(column):
                raise SudokuError(f"Column ({column + 1}) is invalid")
        for row, column in product(range(3), repeat=2):
            if not self.__validate_square(row, column):
                raise SudokuError(f"Block ({row + 1}, {column + 1}) is invalid")

    @staticmethod
    def __validate_block(block):
        non_empty = [n for n in block if n]
        return len(non_empty) == len(set(non_empty))

    def __validate_row(self, row):
        return self.__validate_block(self.get_row(self.board, row))

    def __validate_column(self, column):
        return self.__validate_block(self.get_column(self.board, column))

    def __validate_square(self, row, column):
        return self.__validate_block(self.get_square(self.board, row, column))


class SudokuGame(SudokuBase):
    """
    A Sudoku game, in charge of storing the state of the board
    and checking whether the puzzle is completed.
    A strict game disallows any illegal number placements.
    """
    def __init__(self, board_file, strict=True):
        self.board_file = board_file
        self.strict = strict

        self.start_puzzle = SudokuBoard(board_file).board
        self.puzzle = None  # Defined upon every start
        self.fixed_positions = {(r, c) for r, c in product(range(9), repeat=2) if self.start_puzzle[r][c] != 0}
        self.game_over = False

    def start(self):
        self.game_over = False
        self.puzzle = self.start_puzzle.copy()

    def get_cell(self, row, column):
        return self.puzzle[row][column]

    def set_cell(self, row, column, number):
        if self.__is_illegal_position(row, column):
            raise SudokuError("Illegal position - fixed from starting board.")
        if self.strict and self.__is_impossible_position(row, column, number):
            raise SudokuError("Impossible position - other numbers interfere")
        self.puzzle[row][column] = number

    def check_win(self):
        for row in range(9):
            if not self.__check_row(row):
                return False
        for column in range(9):
            if not self.__check_column(column):
                return False
        for row, column in product(range(3), repeat=2):
            if not self.__check_square(row, column):
                return False
        self.game_over = True
        return True

    def __is_illegal_position(self, row, column):
        return (row, column) in self.fixed_positions

    def __is_impossible_position(self, row, column, number):
        if number == 0:
            return False  # Clearing a cell is always possible
        if number in self.get_row(self.puzzle, row):
            return True
        if number in self.get_column(self.puzzle, column):
            return True
        if number in self.get_square(self.puzzle, row // 3, column // 3):
            return True
        return False

    @staticmethod
    def __check_block(block):
        return set(block) == set(range(1, 10))

    def __check_row(self, row):
        return self.__check_block(self.get_row(self.puzzle, row))

    def __check_column(self, column):
        return self.__check_block(self.get_column(self.puzzle, column))

    def __check_square(self, row, column):
        return self.__check_block(self.get_square(self.puzzle, row, column))

    def __str__(self):
        output = ""
        for i, row in enumerate(self.puzzle):
            output += ("{} {} {}|" * 3).format(*[x if x else " " for x in row])
            output = output[:-1] + "\n"

            if i % 3 == 2 and i != 8:
                output += "-" * 5 + ("+" + "-" * 5) * 2 + "\n"

        return output


class SudokuCLI:
    """
    Command-Line Interface, responsible for drawing the board
    and accepting user input.
    """
    def __init__(self, game):
        self.game = game
        self.__print_puzzle()
        self.__get_move()

    def __print_puzzle(self):
        print()
        print(self.game)

    def __clear_answers(self):
        self.game.start()
        self.__print_puzzle()

    def __get_move(self):
        if self.game.game_over:
            return

        row = self.__get_input("Row: ") - 1
        column = self.__get_input("Column: ") - 1
        cur_num = self.game.get_cell(row, column)
        msg = "Number{}: ".format(f" (currently {cur_num})" if cur_num else "")
        num = self.__get_input(msg)

        try:
            self.game.set_cell(row, column, num)
        except SudokuError as e:
            print(e)

        self.__print_puzzle()
        if self.game.check_win():
            print("Congratulations! Puzzle solved correctly!")
            return

        self.__get_move()

    def __get_input(self, msg):
        try:
            user_input = input_int(msg, min=0, max=9)
        except SudokuError as e:
            print(e)
            return self.__get_input(msg)

        if user_input == -1:
            print("Thanks for playing!")
            exit()
        return user_input


def input_int(msg, min=None, max=None):
    user_input = input(msg).lower()
    if user_input in ("q", "quit"):
        return -1
    try:
        user_input = int(user_input)
    except ValueError:
        raise SudokuError("Input must be an integer")

    if min is None and max is None:
        return user_input
    if min is not None and user_input < min:
        if max is not None:
            raise SudokuError(f"Input must be between {min}-{max}")
        raise SudokuError(f"Input must be greater than {min}")
    if max is not None and user_input > max:
        raise SudokuError(f"Input must be less than {max}")

    return user_input


def parse_arguments():
    """
    Pases arguments of the form:
        sudoku.py <board name>
    Where `board name` must be in the `BOARD` list
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--board",
                            help="Board name",
                            type=str,
                            choices=BOARDS,
                            required=True)
    arg_parser.add_argument("--strict",
                            help="Strict positioning",
                            dest="strict",
                            action="store_true")
    arg_parser.set_defaults(strict=False)

    # Creates a dictionary of keys = argument flag, and value = argument
    args = arg_parser.parse_args()
    return args


def main():
    args = parse_arguments()
    board_name = args.board
    strict = args.strict

    with open(f"sudokus/{board_name}.sudoku", "r") as board_file:
        try:
            game = SudokuGame(board_file, strict=strict)
        except SudokuError as e:
            print(e)
            return

        game.start()

        SudokuCLI(game)


if __name__ == "__main__":
    main()
