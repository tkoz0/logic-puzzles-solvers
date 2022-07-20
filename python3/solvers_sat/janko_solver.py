'''
Usage: python3 janko_solver.py <file> [special options ...]
Expects .jsonl files (uncompressed) from the janko.at-puzzle-scraping repository
They should have original filenames for determining puzzle type
The special options part is for testing/debugging purposes
'''

from functools import reduce
import json
import os
from statistics import stdev
import sys
import time
from tqdm import tqdm
import traceback
from typing import Any, Callable, Dict, List, Tuple

from cnf_utils import cnf_stats
from sat_puzzle import *

# get part of filename before the .jsonl
input_file = sys.argv[1]
base,ext = os.path.splitext(input_file)
assert ext == '.jsonl'
puzzle_dir = os.path.basename(base)

sys.stderr.write(f'input_file = {input_file}\n')
sys.stderr.write(f'puzzle = {puzzle_dir}\n')

objects: List[Dict[str,Any]] = [json.loads(line) for line in open(input_file,'r')]
sys.stderr.write(f'loaded {len(objects)} objects\n')

if len(sys.argv) > 2: # extra option
    option = sys.argv[2]
    if option == 'list_params':
        values = reduce(lambda x,y: x|y, (set(object['data'].keys()) for object in objects))
        sys.stderr.write(f'parameters found = {values}\n')
    elif option == 'list_param_values':
        values = set(object['data'][sys.argv[3]] for object in objects if sys.argv[3] in object['data'])
        count_missing = sum(1 for object in objects if sys.argv[3] not in object['data'])
        sys.stderr.write(f'values found for parameter {sys.argv[3]} = {values}\n')
        if count_missing > 0:
            sys.stderr.write(f'counted {count_missing} objects without this parameter\n')
    elif option == 'write_json':
        sys.stderr.write(f'{json.dumps(objects[int(sys.argv[3])-1],indent=4)}\n')
    else:
        sys.stderr.write(f'invalid option = {sys.argv[2]}\n')
    quit()

def gridnum2int(x: str) -> int:
    return 0 if x in '-.' else int(x)
def gridnum2int2(x: str) -> int:
    # also converts letters
    return 0 if x in '-.' else (int(x) if x.isdigit() else '_abcdefghijklmnopqrstuvwxyz'.index(x.lower()))
def grid2numlist(x: List[List[str]]) -> List[List[int]]:
    return list(map(lambda row: list(map(gridnum2int,row)), x))
def grid2numlist2(x: List[List[str]]) -> List[List[int]]:
    # also convert letters (aA -> 1, bB -> 2, ..)
    return list(map(lambda row: list(map(gridnum2int2,row)), x))

# map category -> puzzle timings
solving_times: Dict[str,List[float]] = dict()

def insert_timing(category: str, runtime: float):
    global solving_times
    if category not in solving_times:
        solving_times[category] = []
    solving_times[category].append(runtime)

def check_solution(solver: SatPuzzleBase, solution: Any, category: str):
    global solving_times
    cnf = solver.toCnf()
    variables,clauses = cnf_stats(cnf)
    tqdm.write(f'generated CNF with {variables} variables and {clauses} clauses')
    start = time.perf_counter()
    solutions = list(solver.solveAll())
    solving_time = time.perf_counter()-start
    tqdm.write(f'solved in {solving_time} seconds')
    #if len(solutions) > 1: print('a',[''.join(map(str,row)) for row in solutions[0]]);print('b',[''.join(map(str,row)) for row in solutions[1]])
    if 0: # debug
        for s in solutions:
            print('\n'.join(map(str,s))+'\n-----')
    assert len(solutions) == 1, f'found {len(solutions)} solutions'
    assert solutions[0] == solution
    insert_timing(category,solving_time)

def _not_implemented(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    sys.stderr.write(f'not implemented\n')
    quit()

def _sudoku_standard(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    blockR = 0
    blockC = 0
    if 'patternx' in data:
        blockR = data['patterny']
        blockC = data['patternx']
    elif 'size' in data:
        blockR = blockC = {16:4,9:3,4:2}[data['size']]
    elif 'rows' in data:
        assert data['rows'] == data['cols']
        blockR = blockC = {16:4,9:3,4:2}[data['rows']]
    else: # assume standard 9x9
        blockR = blockC = 3
    givens = grid2numlist(data['problem'])
    solution = grid2numlist(data['solution'])
    if data['puzzle'] == 'sudoku':
        solver = SatPuzzleSudokuStandard(blockR,blockC,givens)
        x = False
    else:
        assert data['puzzle'] == 'sudoku, diagonals'
        solver = SatPuzzleSudokuX(blockR,blockC,givens)
        x = True
    return solver, solution, f'{blockR}x{blockC}{"_diag" if x else ""}'

def _latinsquare_x(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    N = int(data['size'])
    givens = grid2numlist(data['problem'])
    solution = grid2numlist(data['solution'])
    solver = SatPuzzleLatinSquareX(N,givens)
    return solver, solution, f'{N}'

def _sudoku_butterfly(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    givens = grid2numlist(data['problem'])
    solution = grid2numlist(data['solution'])
    solver = SatPuzzleSudokuButterfly(givens)
    return solver, solution, 'butterfly'

def _sudoku_jigsaw(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    N = data['size']
    givens = grid2numlist2(data['problem'])
    areas = data['areas']
    solution = grid2numlist2(data['solution'])
    solver = SatPuzzleSudokuJigsaw(N,givens,areas)
    return solver, solution, f'{N}'

def _sudoku_clueless1(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    givens = grid2numlist(data['problem'])
    solution = grid2numlist(data['solution'])
    solver = SatPuzzleSudokuClueless1(givens)
    return solver, solution, 'clueless1'

def _sudoku_clueless2(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    givens = grid2numlist(data['problem'])
    solution = grid2numlist(data['solution'])
    solver = SatPuzzleSudokuClueless2(givens)
    return solver, solution, 'clueless1'

def _sudoku_flower(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    givens = grid2numlist(data['problem'])
    solution = grid2numlist(data['solution'])
    solver = SatPuzzleSudokuFlower(givens)
    return solver, solution, 'flower'

def _sudoku_gattai8(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    givens = grid2numlist(data['problem'])
    solution = grid2numlist(data['solution'])
    solver = SatPuzzleSudokuGattai8(givens)
    return solver, solution, 'gattai8'

def _sudoku_consecutive(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    givens = grid2numlist(data['problem'])
    N = len(givens)
    solution = grid2numlist(data['solution'])
    # parse solution into consecutives
    consecH = [[solution[r][c] == solution[r+1][c]+1 or solution[r][c]+1 == solution[r+1][c] for c in range(N)] for r in range(N-1)]
    consecV = [[solution[r][c] == solution[r][c+1]+1 or solution[r][c]+1 == solution[r][c+1] for c in range(N-1)] for r in range(N)]
    solver = SatPuzzleSudokuConsecutive(N,givens,consecH,consecV)
    return solver, solution, f'{N}'

# convert the data object to a solver object, provided solution, and category
parsers: Dict[str,Callable[[Dict[str,Any]],Tuple[SatPuzzleBase,Any,str]]] = \
{
    'Sudoku': _sudoku_standard,
    'Sudoku_2D': _latinsquare_x,
    'Sudoku_Butterfly': _sudoku_butterfly,
    'Sudoku_Chaos': _sudoku_jigsaw,
    'Sudoku_Clueless-1': _sudoku_clueless1,
    'Sudoku_Clueless-2': _sudoku_clueless2,
    'Sudoku_Flower': _sudoku_flower,
    'Sudoku_Gattai-8': _sudoku_gattai8,
    'Sudoku_Killer': _not_implemented,
    'Sudoku_Konsekutiv': _sudoku_consecutive
}

global_start = time.perf_counter()
category2nums: Dict[str,List[int]] = dict()
i = 0
sys.stderr.write('\n')
for object in tqdm(objects):
    i += 1
    object_file = object['file']
    tqdm.write(f'processing object {i} = {object_file}')
    try:
        solver,solution,category = parsers[puzzle_dir](object['data'])
        if category not in category2nums:
            category2nums[category] = []
        category2nums[category].append(i)
        check_solution(solver,solution,category)
    except Exception as e:
        sys.stderr.write(f'{json.dumps(object,indent=4)}\n')
        sys.stderr.write(f'ERROR ON THIS OBJECT = {type(e)}: {e}\n')
        traceback.print_exc()
        quit()
    tqdm.write('')
global_time = time.perf_counter()-global_start
sys.stderr.write('\n')
sys.stderr.write(f'solved {len(objects)} puzzles in {global_time} seconds\n')
sys.stderr.write(f'average solving time is {global_time/len(objects)} seconds\n')
sys.stderr.write('\n')

for category in solving_times:
    times = solving_times[category]
    sys.stderr.write(f'category {category} ({len(times)} puzzles)\n')
    sys.stderr.write(f'puzzles = {category2nums[category]}\n')
    sys.stderr.write(f'min = {min(times)}\n')
    sys.stderr.write(f'max = {max(times)}\n')
    sys.stderr.write(f'avg = {sum(times)/len(times)}\n')
    sys.stderr.write(f'stddev = {stdev(times)}\n')
    sys.stderr.write('\n')