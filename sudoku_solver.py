import argparse

from itertools import combinations, product, tee


BOARDS = ["easy", "medium", "medium2", "hard"]  # Available Sudoku boards


class SudokuError(Exception):
    """
    An application specific error.
    """
    pass


class SudokuBase:
    """
    Base Sudoku methods used by multiple subclasses
    for getting and setting blocks from two-dimensional arrays
    in some way representing the Sudoku board
    """
    @staticmethod
    def get_row(board, row):
        return board[row]

    @staticmethod
    def get_rows(board):
        return (SudokuBase.get_row(board, row) for row in range(9))

    @staticmethod
    def get_column(board, column):
        return (board[row][column] for row in range(9))

    @staticmethod
    def get_columns(board):
        return (SudokuBase.get_column(board, column) for column in range(9))

    @staticmethod
    def get_square(board, row, column):
        return (
            board[r][c]
            for r in range(row * 3, (row + 1) * 3)
            for c in range(column * 3, (column + 1) * 3)
        )

    @staticmethod
    def get_squares(board):
        return (SudokuBase.get_square(board, row, column) for row, column in product(range(3), repeat=2))


class SudokuBoard(SudokuBase):
    """
    Sudoku Board representation
    """
    def __init__(self, board_file):
        self.board = self.__create_board(board_file)
        self.validate(self.board)

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

    @staticmethod
    def validate(board):
        for row in range(9):
            if not SudokuBoard.__validate_row(board, row):
                raise SudokuError(f"Row ({row + 1}) is invalid")
        for column in range(9):
            if not SudokuBoard.__validate_column(board, column):
                raise SudokuError(f"Column ({column + 1}) is invalid")
        for row, column in product(range(3), repeat=2):
            if not SudokuBoard.__validate_square(board, row, column):
                raise SudokuError(f"Block ({row + 1}, {column + 1}) is invalid")

        return True

    @staticmethod
    def __validate_block(block):
        non_empty = [n for n in block if n]
        return len(non_empty) == len(set(non_empty))

    @staticmethod
    def __validate_row(board, row):
        return SudokuBoard.__validate_block(SudokuBase.get_row(board, row))

    @staticmethod
    def __validate_column(board, column):
        return SudokuBoard.__validate_block(SudokuBase.get_column(board, column))

    @staticmethod
    def __validate_square(board, row, column):
        return SudokuBoard.__validate_block(SudokuBase.get_square(board, row, column))


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


class SudokuSolver(SudokuBase):
    def __init__(self, game):
        self.game = game
        self.board = self.game.puzzle
        self.remainders = []
        self.__init_constraints()

    def start(self):
        # Run until finished solving
        while not self.game.game_over:
            print(self.game)
            # Check for any naked singles that can be set directly
            if self.__naked_single():
                self.game.check_win()
                continue

            # Run variety of strategies
            if not self.__run_strategies():
                break

        print(self.game)

    def __run_strategies(self):
        strategies = [
            self.__hidden_single,
            self.__naked_pair,
            self.__hidden_pair,
            self.__naked_triple,
            self.__hidden_triple,
            self.__naked_quad,
            self.__hidden_quad,
        ]

        # Run technique and return if anything was changed
        for strategy in strategies:
            print(strategy.__name__)
            if strategy():
                return True

        # No changes with any technique - failed to solve
        return False

    def __get_cell_remainders(self, row, column):
        if self.board[row][column] != 0:
            return set()

        remainders = set(range(1, 10))
        remainders -= (set(self.get_row(self.board, row)) |
                       set(self.get_column(self.board, column)) |
                       set(self.get_square(self.board, row // 3, column // 3))
                       )
        return remainders

    def __init_constraints(self):
        for r, row in enumerate(self.board):
            self.remainders.append([])
            for c, cell in enumerate(row):
                remainders = self.__get_cell_remainders(r, c)
                self.remainders[r].append(remainders)

    def __update_surrounding(self, row, column, number):
        # Iterate over the row, column and square
        # and remove the number
        start_row, start_column = (row // 3) * 3, (column // 3) * 3
        for i in range(9):
            self.remainders[row][i].discard(number)
            self.remainders[i][column].discard(number)
            self.remainders[start_row + i // 3][start_column + i % 3].discard(number)

    @staticmethod
    def __find_n_identical_in_n_cells(block, n):
        match_cells = set()
        match_numbers = set()
        start_idx = 0

        while start_idx < 7:
            if len(block[start_idx]) == n:
                match_cells.add(start_idx)
                match_numbers = block[start_idx]
                idx = start_idx + 1
                while idx < 9 and len(match_cells) < n:
                    if block[idx] == match_numbers:
                        match_cells.add(idx)
                    idx += 1

                if len(match_cells) == n:
                    break

            start_idx += 1
            match_cells = set()

        return match_cells, match_numbers

    def __naked(self, count):
        updated = False

        for r, row in enumerate(self.get_rows(self.remainders)):
            cells, numbers = self.__find_n_identical_in_n_cells(row, count)
            if len(cells) > 0:
                for c in set(range(9)) - cells:
                    if len(self.remainders[r][c] & numbers) != 0:
                        updated = True
                        self.remainders[r][c] -= numbers

        for c, column in enumerate(self.get_columns(self.remainders)):
            cells, numbers = self.__find_n_identical_in_n_cells(list(column), count)
            if len(cells) > 0:
                for r in set(range(9)) - cells:
                    if len(self.remainders[r][c] & numbers) != 0:
                        updated = True
                        self.remainders[r][c] -= numbers

        for s, square in enumerate(self.get_squares(self.remainders)):
            cells, numbers = self.__find_n_identical_in_n_cells(list(square), count)
            if len(cells) > 0:
                r_offset = (s // 3) * 3
                c_offset = (s % 3) * 3
                for n in set(range(9)) - cells:
                    r = r_offset + n // 3
                    c = c_offset + n % 3
                    if len(self.remainders[r][c] & numbers) != 0:
                        updated = True
                        self.remainders[r][c] -= numbers

        return updated

    def __naked_single(self):
        """
        Checks all cells for a naked single and sets each in the game.
        Naked singles are when only a single number can be in the cell
        Thus, the number is forced into this position.
        :return: whether a cell was found
        """
        updated = False
        for row, column in product(range(9), repeat=2):
            cell = self.remainders[row][column]
            if len(cell) == 1:
                single = cell.pop()
                self.game.set_cell(row, column, single)
                print("Single:", row, column, single)
                print(self.game)
                self.__update_surrounding(row, column, single)
                updated = True
        return updated

    def __naked_pair(self):
        return self.__naked(2)

    def __naked_triple(self):
        return self.__naked(3)

    def __naked_quad(self):
        return self.__naked(4)

    @staticmethod
    def __find_number_occurrences(block):
        block_counter = dict()

        # Count occurrences of each remainder in all blocks
        for cell_num, cell in enumerate(block):
            for num in cell:
                block_counter.setdefault(num, set()).add(cell_num)

        return block_counter

    def __get_numbers_occurring_n(self, block, n):
        matches = dict()

        number_occurrences = self.__find_number_occurrences(block)
        for num, occurrences in number_occurrences.items():
            if len(occurrences) == n:
                matches[num] = occurrences
        return matches

    @staticmethod
    def __get_n_cells_with_hidden(cell_pool, n):
        if len(cell_pool) == 0:
            return

        # Iterate over all possible pairings
        if n == 1:
            number, occurrences = cell_pool.popitem()
            return {number}, occurrences

        for potential_match in combinations(cell_pool, n):
            match = True
            for num, next_num in pairwise(potential_match):
                if cell_pool[num] != cell_pool[next_num]:
                    match = False
                    break

            if match:
                numbers = {*potential_match}
                occurrences = cell_pool[potential_match[0]]
                return numbers, occurrences

    def __isolate_hidden(self, hidden, row=None, column=None):
        if row is None and column is None:
            raise AttributeError("Either row or column must be specified")

        numbers, occurrences = hidden
        any_isolated = False
        for cell in occurrences:
            # Square input
            if not (row is None or column is None):
                r = row * 3 + cell // 3
                c = column * 3 + cell % 3
            # Row or column input
            else:
                r = cell if row is None else row
                c = cell if column is None else column

            if self.remainders[r][c] != numbers:
                self.remainders[r][c] = numbers
                any_isolated = True

        return any_isolated

    def __hidden_block(self, block, count, row=None, column=None):
        matches = self.__get_numbers_occurring_n(block, count)
        hidden = self.__get_n_cells_with_hidden(matches, count)

        if hidden is not None:
            return self.__isolate_hidden(hidden, row=row, column=column)

        return False

    def __hidden(self, count):
        """
        Checks each block (row, column, square) for a hidden
        A hidden single is when only a single cell in the block can hold a specific number
        but other numbers are also possibilities for the cell.
        These can then be removed as possibilities.
        """
        # Check in rows
        for r, row in enumerate(self.get_rows(self.remainders)):
            if self.__hidden_block(row, count, row=r):
                return True

        # Check in columns
        for c, column in enumerate(self.get_columns(self.remainders)):
            if self.__hidden_block(column, count, column=c):
                return True

        # Check in square
        for s, square in enumerate(self.get_squares(self.remainders)):
            if self.__hidden_block(square, count, row=s // 3, column=s % 3):
                return True

        return False

    def __hidden_single(self):
        return self.__hidden(1)

    def __hidden_pair(self):
        return self.__hidden(2)

    def __hidden_triple(self):
        return self.__hidden(3)

    def __hidden_quad(self):
        return self.__hidden(4)


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
            user_input = input_int(msg, min_val=0, max_val=9)
        except SudokuError as e:
            print(e)
            return self.__get_input(msg)

        if user_input == -1:
            print("Thanks for playing!")
            exit()
        return user_input


def pairwise(t):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(t)
    next(b, None)
    return zip(a, b)


def input_int(msg, min_val=None, max_val=None):
    user_input = input(msg).lower()
    if user_input in ("q", "quit"):
        return -1
    try:
        user_input = int(user_input)
    except ValueError:
        raise SudokuError("Input must be an integer")

    if min_val is None and max_val is None:
        return user_input
    if min_val is not None and user_input < min_val:
        if max_val is not None:
            raise SudokuError(f"Input must be between {min_val}-{max_val}")
        raise SudokuError(f"Input must be greater than {min_val}")
    if max_val is not None and user_input > max_val:
        raise SudokuError(f"Input must be less than {max_val}")

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

        solver = SudokuSolver(game)
        solver.start()

        # SudokuCLI(game)


if __name__ == "__main__":
    main()
