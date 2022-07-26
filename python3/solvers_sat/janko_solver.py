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
    solver = SatPuzzleLatinSquareX(givens)
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
    solver = SatPuzzleSudokuJigsaw(givens,areas)
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
    consec_pairs: Set[Tuple[Tuple[int,int],Tuple[int,int]]] = set()
    for r in range(N-1):
        for c in range(N):
            a,b = solution[r][c],solution[r+1][c]
            if abs(a-b) == 1:
                consec_pairs.add(((r,c),(r+1,c)))
    for r in range(N):
        for c in range(N-1):
            a,b = solution[r][c],solution[r][c+1]
            if abs(a-b) == 1:
                consec_pairs.add(((r,c),(r,c+1)))
    solver = SatPuzzleSudokuConsecutive(givens,consec_pairs)
    return solver, solution, f'{N}'

def _sudoku_kropki(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    if 'patternx' in data:
        br,bc = data['patterny'],data['patternx']
    else: # if not specified by above then use square root
        assert data['size'] == 9
        br,bc = 3,3
    solution = grid2numlist(data['solution'])
    N = len(solution)
    white: Set[Tuple[Tuple[int,int],Tuple[int,int]]] = set()
    black: Set[Tuple[Tuple[int,int],Tuple[int,int]]] = set()
    for r in range(N-1): # vertical pairs
        for c in range(N):
            a,b = solution[r][c],solution[r+1][c]
            if a*2==b or a==b*2: # 1,2 is marked as black so prioritize this
                black.add(((r,c),(r+1,c)))
            elif abs(a-b) == 1:
                white.add(((r,c),(r+1,c)))
    for r in range(N): # horizontal pairs
        for c in range(N-1):
            a,b = solution[r][c],solution[r][c+1]
            if a*2==b or a==b*2:
                black.add(((r,c),(r,c+1)))
            elif abs(a-b) == 1:
                white.add(((r,c),(r,c+1)))
    solver = SatPuzzleSudokuKropki(br,bc,white,black)
    return solver, solution, f'{N}'

def _sudoku_magicnumberx(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    givens = grid2numlist(data['problem'])
    N = len(givens)
    clues = grid2numlist(data['clues'])
    solution = grid2numlist(data['solution'])
    pairs: Set[Tuple[Tuple[int,int],Tuple[int,int]]] = set()
    for r in range(N):
        for c in range(N):
            if clues[r][c] == 1 or clues[r][c] == 3: # right edge
                pairs.add(((r,c),(r,c+1)))
            if clues[r][c] == 2 or clues[r][c] == 3: # bottom edge
                pairs.add(((r,c),(r+1,c)))
    solver = SatPuzzleSudokuMagicNumberX(givens,data['magic'],pairs)
    return solver, solution, f'{N}'

def _sudoku_samurai(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    givens = grid2numlist(data['problem'])
    solution = grid2numlist(data['solution'])
    solver = SatPuzzleSudokuSamurai(givens)
    return solver, solution, 'samurai'

def _sudoku_oddeven(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    givens = grid2numlist(data['problem'])
    solution = grid2numlist(data['solution'])
    odds = [[n % 2 == 1 for n in row] for row in solution]
    if 'samurai' in data['puzzle']:
        solver = SatPuzzleSudokuOddEvenSamurai(givens,odds)
    else:
        try:
            blockR,blockC = data['patterny'],data['patternx']
        except:
            if data['size'] == 9:
                blockR,blockC = 3,3
            else:
                blockR,blockC = 0,0
                assert 0
        solver = SatPuzzleSudokuOddEven(blockR,blockC,givens,odds)
    return solver, solution, f'{len(givens)}'

def _sudoku_marginalsum(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    blockR,blockC = data['patterny'],data['patternx']
    N = blockR*blockC
    solution = grid2numlist(data['solution'])
    top = [sum(solution[r][c] for r in range(blockR)) for c in range(N)]
    bottom = [sum(solution[r][c] for r in range(N-blockR,N)) for c in range(N)]
    left = [sum(solution[r][c] for c in range(blockC)) for r in range(N)]
    right = [sum(solution[r][c] for c in range(N-blockC,N)) for r in range(N)]
    solver = SatPuzzleSudokuMarginalSum(blockR,blockC,top,bottom,left,right)
    return solver, solution, f'{blockR}x{blockC}'

def _sudoku_shogun(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    givens = grid2numlist(data['problem'])
    solution = grid2numlist(data['solution'])
    solver = SatPuzzleSudokuShogun(givens)
    return solver, solution, 'shogun'

def _sudoku_sohei(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    givens = grid2numlist(data['problem'])
    solution = grid2numlist(data['solution'])
    solver = SatPuzzleSudokuSohei(givens)
    return solver, solution, 'shogun'

def _sudoku_sumo(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    givens = grid2numlist(data['problem'])
    solution = grid2numlist(data['solution'])
    solver = SatPuzzleSudokuSumo(givens)
    return solver, solution, 'sumo'

def _sudoku_windmill(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    givens = grid2numlist(data['problem'])
    solution = grid2numlist(data['solution'])
    solver = SatPuzzleSudokuWindmill(givens)
    return solver, solution, 'windmill'

def _sudoku_comparison(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    solution = grid2numlist(data['solution'])
    try:
        blockR,blockC = data['patterny'],data['patternx']
    except:
        assert data['size'] == 9
        blockR,blockC = 3,3
    N = blockR*blockC
    relations = set()
    for r in range(N):
        for bc in range(blockR):
            for i in range(blockC-1):
                p1 = (r,bc*blockC+i)
                p2 = (r,bc*blockC+i+1)
                if solution[p1[0]][p1[1]] < solution[p2[0]][p2[1]]:
                    relations.add((p1,p2))
                else:
                    relations.add((p2,p1))
    for c in range(N):
        for br in range(blockC):
            for i in range(blockR-1):
                p1 = (br*blockR+i,c)
                p2 = (br*blockR+i+1,c)
                if solution[p1[0]][p1[1]] < solution[p2[0]][p2[1]]:
                    relations.add((p1,p2))
                else:
                    relations.add((p2,p1))
    solver = SatPuzzleSudokuComparison(blockR,blockC,relations)
    return solver, solution, f'{blockR}x{blockC}'

def _suguru(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    givens = grid2numlist(data['problem'])
    areas = grid2numlist(data['areas'])
    solution = grid2numlist(data['solution'])
    solver = SatPuzzleSuguruStandard(givens,areas)
    return solver, solution, f'{len(givens)}x{len(givens[0])}'

def _hakyuu(data: Dict[str,Any]) -> Tuple[SatPuzzleBase,Any,str]:
    givens = grid2numlist(data['problem'])
    areas = grid2numlist(data['areas'])
    solution = grid2numlist(data['solution'])
    solver = SatPuzzleHakyuu(givens,areas)
    return solver, solution, f'{len(givens)}x{len(givens[0])}'

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
    'Sudoku_Konsekutiv': _sudoku_consecutive,
    'Sudoku_Kropki': _sudoku_kropki,
    'Sudoku_Magic-Number': _sudoku_magicnumberx,
    'Sudoku-Odd-Even': _sudoku_oddeven,
    'Sudoku_Odd-Even': _sudoku_oddeven,
    'Sudoku_Samurai': _sudoku_samurai,
    'Sudoku-Randsummen': _sudoku_marginalsum,
    'Sudoku_Randsummen': _sudoku_marginalsum,
    'Sudoku_Shogun': _sudoku_shogun,
    'Sudoku_Sohei': _sudoku_sohei,
    'Sudoku_Sumo': _sudoku_sumo,
    'Sudoku_Windmill': _sudoku_windmill,
    'Sudoku_Vergleich': _sudoku_comparison,
    'Sudoku_Wolkenkratzer': _not_implemented,
    'Suguru': _suguru,
    'Hakyuu': _hakyuu
}

# puzzles to skip due to issues that make them not work with the main solver
skip_puzzles: Dict[str,str] = \
{
    '/Hakyuu/469.a.x-janko': 'no solution provided'
}

global_start = time.perf_counter()
category2nums: Dict[str,List[int]] = dict()
skip_count = 0
i = 0
sys.stderr.write('\n')
for object in tqdm(objects):
    i += 1
    object_file = object['file']
    if object_file in skip_puzzles:
        tqdm.write(f'SKIPPING OBJECT {i} = {object_file}')
        tqdm.write(f'reason = {skip_puzzles[object_file]}')
        skip_count += 1
    else:
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
sys.stderr.write(f'solved {len(objects)-skip_count} puzzles in {global_time} seconds\n')
sys.stderr.write(f'average solving time is {global_time/len(objects)} seconds\n')
sys.stderr.write('\n')

for category in solving_times:
    times = solving_times[category]
    sys.stderr.write(f'category {category} ({len(times)} puzzles)\n')
    sys.stderr.write(f'puzzles = {category2nums[category]}\n')
    sys.stderr.write(f'min = {min(times)}\n')
    sys.stderr.write(f'max = {max(times)}\n')
    sys.stderr.write(f'avg = {sum(times)/len(times)}\n')
    if len(times) >= 2:
        sys.stderr.write(f'stddev = {stdev(times)}\n')
    sys.stderr.write('\n')
